"""
Hamner vs moco EMG excitation copmarison 

Outputs (results/):
  emg_vs_moco_percycle.csv   track x cycle x shift x muscle -> r, rmse (the raw stats)
  emg_vs_moco_summary.csv    track x shift x muscle -> mean +/- SD over the 5 cycles
Figures (plots/): one per track x shift, spaghetti over cycles.

Run: conda run -n osim46 python compare_emg_moco.py [track ...]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE)) # repo root (validation/emg -> repo)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PKG, "scripts"))
from hamner_emg import SubjectEMG, EMG_GROUPS
from registration import register
from moco_solution import read_controls, read_f0m, aggregate_activation
import rstats
from lib.paper1io import CYCLES, crop_window

PLOTS = os.path.join(HERE, "plots")
RESULTS = os.path.join(HERE, "results")
os.makedirs(PLOTS, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)
DATA_EMG = os.path.join(PKG, "data", "emg") # raw Hamner EMG + GRF archive
EMG_FOLDER = DATA_EMG
 
TRACKS = {
    "davis": dict(label="Davis/step14", track_dir="davis",
                  model=os.path.join(PKG, "models", "Davis2026_subject01_DGF.osim"),
                  model_key="model_davis"),
    "raj": dict(label="Rajagopal2023 (vanilla)", track_dir="raj",
                model=os.path.join(PKG, "models", "Rajagopal2023_subject01_DGF.osim"),
                model_key="model_raj"),
}
SPEED = 3.0
N = 101

# electromechancial delay (Hamner cf)
EMD_MS = 75.0
EMD_CONDITIONS = {0.0: "raw / unshifted (PRIMARY)", EMD_MS: f"+{EMD_MS:.0f} ms EMD (secondary)"}

PANELS = [
    ("soleus_r", False, "soleus"),
    ("gasmed_r", False, "gastroc med"),
    ("gaslat_r", False, "gastroc lat"),
    ("tibant_r", False, "tib ant"),
    ("vaslat_r", False, "vastus lat"),
    ("vasmed_r", False, "vastus med"),
    ("recfem_r", False, "rectus fem"),
    ("bflh_r", False, "biceps fem lh"),
    ("hamstring_med", True, "semimem+ten (1 electrode)"),
    ("glmax", True, "glute max (1 electrode)"),
    ("glmed", True, "glute med (1 electrode)"),
]

#R corr
def pearson(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def solution(cfg, cycle):
    return os.path.join(PKG, "results", cfg["track_dir"],
                        f"cycle{cycle}_compliant_solution.sto")


def registered_pair(env_norm, act, f0m, key, is_group, model_key, t0, t1, shift):
    emg_ch = EMG_GROUPS[key]["emg_rep"] if is_group else key
    _, emg = register(env_norm.index.values + shift, env_norm[emg_ch].values, t0, t1, n=N)
    mt = act.index.values
    if not is_group:
        _, aw = register(mt, act[key].values, t0, t1, n=N)
        return emg, aw, None, key
    tracts = [t for t in EMG_GROUPS[key][model_key] if t in act.columns]
    _, aw = register(mt, aggregate_activation(act, tracts, f0m=f0m, method="f0m").values,
                     t0, t1, n=N)
    _, am = register(mt, aggregate_activation(act, tracts, method="mean").values,
                     t0, t1, n=N)
    return emg, aw, am, "+".join(tracts)


def collect(track_key, cfg, env_norm):
    """Per-cycle r and RMSE for every panel x EMD shift. No averaging anywhere."""
    f0m = read_f0m(cfg["model"])
    rows, curves = [], {}
    for cycle in CYCLES:
        act = read_controls(solution(cfg, cycle))
        t0, t1 = crop_window(cycle)
        for emd_ms in EMD_CONDITIONS:
            for key, is_group, title in PANELS:
                emg, aw, am, tracts = registered_pair(
                    env_norm, act, f0m, key, is_group, cfg["model_key"], t0, t1,
                    emd_ms / 1000.0)
                rows.append(dict(track=track_key, cycle=cycle, emd_ms=emd_ms,
                                 muscle=title, model_tracts=tracts,
                                 r=pearson(emg, aw), rmse=rmse(emg, aw),
                                 r_meanagg=(pearson(emg, am) if am is not None else np.nan),
                                 rmse_meanagg=(rmse(emg, am) if am is not None else np.nan)))
                curves.setdefault((emd_ms, title), []).append((emg, aw))
    return pd.DataFrame(rows), curves


def summarize(percycle):
    """FISHER-z mean + dispersion ACROSS CYCLES of the per-cycle statistics.

    r_mean is tanh(mean(arctanh(r))) -- see rstats.py for why the arithmetic mean is
    biased toward zero. r_sd is the plain SD of the per-cycle r, kept as a DESCRIPTIVE
    spread; r_lo68/r_hi68 are the coherent (asymmetric) +/-1 SD interval in z-space.
    RMSE is a plain mean -- it is not a correlation and needs no transform.
    """
    # NB group/filter on "emd_ms", never an attribute named `shift` -- DataFrame.shift is
    # a method, so `df.shift == x` silently yields a scalar False and empties the filter.
    g = percycle.groupby(["track", "emd_ms", "muscle"], sort=False)
    out = g.agg(n_cycles=("cycle", "nunique"),
                r_mean=("r", rstats.fisher_mean),
                r_sd=("r", rstats.raw_sd),
                r_lo68=("r", lambda v: rstats.fisher_interval(v)[0]),
                r_hi68=("r", lambda v: rstats.fisher_interval(v)[1]),
                r_mean_arith_RETIRED=("r", rstats.arith_mean),
                r_min=("r", "min"), r_max=("r", "max"),
                rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std")).reset_index()
    return out.round(4)


def figure(track_key, cfg, curves, summary, emd_ms, stance_pct):
    from matplotlib.lines import Line2D
    phase = np.linspace(0, 100, N)
    fig, axes = plt.subplots(4, 3, figsize=(12, 12), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, (key, is_group, title) in zip(axes, PANELS):
        pairs = curves[(emd_ms, title)]
        for emg, mod in pairs:                       # the individual cycles
            ax.plot(phase, emg, color="k", lw=0.6, alpha=0.35)
            ax.plot(phase, mod, color="C3", lw=0.6, alpha=0.35)
        ax.plot(phase, np.mean([e for e, _ in pairs], axis=0), color="k", lw=2.4)
        ax.plot(phase, np.mean([m for _, m in pairs], axis=0), color="C3", lw=2.2)
        s = summary[(summary.track == track_key) & (summary["emd_ms"] == emd_ms)
                    & (summary.muscle == title)]
        assert len(s) == 1, f"no summary row for {track_key}/{emd_ms}/{title}"
        s = s.iloc[0]
        ax.set_title(f"{title}\nr = {s.r_mean:.2f} ± {s.r_sd:.2f}   "
                     f"RMSE = {s.rmse_mean:.2f}", fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 100)
        ax.tick_params(labelsize=8)
        ax.axvline(stance_pct, color="0.4", ls=":", lw=1)
    for r_ in range(4):
        axes[3 * r_].set_ylabel("excitation / norm. EMG", fontsize=9)
    for ax in (axes[8], axes[9], axes[10]):
        ax.set_xlabel("% gait cycle", fontsize=9)
    axes[-1].axis("off")
    axes[-1].legend(
        [Line2D([0], [0], color="k", lw=2.4), Line2D([0], [0], color="C3", lw=2.2),
         Line2D([0], [0], color="0.5", lw=0.6), Line2D([0], [0], color="0.4", ls=":", lw=1)],
        ["EMG (measured, subject-max normalised)", "Moco excitation",
         "individual gait cycles (n=5)", f"toe-off (~{stance_pct:.0f}%)"],
        loc="center", fontsize=10, frameon=False)
    shift_txt = ("EMG NOT shifted — PRIMARY" if emd_ms == 0 else
                 f"EMG shifted +{emd_ms:.0f} ms for electromechanical delay — SECONDARY")
    fig.suptitle(f"subject01 Run_30002 (3 m/s), 5 gait cycles — measured EMG vs "
                 f"{cfg['label']} MocoInverse excitation\n{shift_txt};  r and RMSE "
                 f"computed PER CYCLE then averaged (mean ± SD)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    tag = "raw" if emd_ms == 0 else f"emd{emd_ms:.0f}ms"
    out = os.path.join(PLOTS, f"emg_vs_moco_5cyc_{track_key}_{tag}.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main(track_keys=None):
    track_keys = track_keys or list(TRACKS)
    subj = SubjectEMG.from_folder(EMG_FOLDER, subject="subject01")
    trial = subj.trial(SPEED)
    env_norm = trial.envelope / subj.norm
    stance_pct = float(np.mean([c.stance_frac for c in trial.cycles]) * 100)

    # the EMG cycle detection and the solved windows must describe the same strides
    for k, c in enumerate(trial.cycles[:len(CYCLES)]):
        t0, _ = crop_window(CYCLES[k])
        assert abs(c.t_start - t0) < 2e-3, (
            f"cycle {CYCLES[k]}: EMG IC {c.t_start:.4f} != solved window start {t0:.4f}")

    percyc, summ = [], []
    for k in track_keys:
        pc, curves = collect(k, TRACKS[k], env_norm)
        percyc.append(pc)
        s = summarize(pc)
        summ.append(s)
        for emd_ms in EMD_CONDITIONS:
            print(f"[ok] wrote {os.path.relpath(figure(k, TRACKS[k], curves, s, emd_ms, stance_pct), HERE)}")
    percyc = pd.concat(percyc, ignore_index=True)
    summary = pd.concat(summ, ignore_index=True)
    percyc.round(4).to_csv(os.path.join(RESULTS, "emg_vs_moco_percycle.csv"), index=False)
    summary.to_csv(os.path.join(RESULTS, "emg_vs_moco_summary.csv"), index=False)

    for k in track_keys:
        for emd_ms, desc in EMD_CONDITIONS.items():
            s = summary[(summary.track == k) & (summary["emd_ms"] == emd_ms)]
            print(f"\n--- {k} ({TRACKS[k]['label']}), EMD = {emd_ms:.0f} ms [{desc}] — "
                  f"per-cycle stats, mean ± SD over {int(s.n_cycles.iloc[0])} cycles ---")
            print(f"  {'muscle':<28}{'r':>16}{'RMSE':>16}")
            for _, r_ in s.iterrows():
                print(f"  {r_.muscle:<28}{r_.r_mean:>9.2f} ±{r_.r_sd:<5.2f}"
                      f"{r_.rmse_mean:>9.3f} ±{r_.rmse_sd:<5.3f}")
            print(f"  {'MEAN over muscles':<28}"
                  f"{rstats.fisher_mean(s.r_mean):>9.2f}      "
                  f"{s.rmse_mean.mean():>9.3f}")
    print("\n  r is the primary statistic (scale-invariant). RMSE is descriptive: the EMG "
          "side is a\n  subject-max-normalised envelope, so RMSE partly measures that "
          "normalisation, not model error.")
    print(f"\nsaved: {os.path.join('results', 'emg_vs_moco_percycle.csv')}")
    print(f"saved: {os.path.join('results', 'emg_vs_moco_summary.csv')}")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
