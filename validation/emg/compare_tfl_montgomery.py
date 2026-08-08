"""
TFL vs lit 

"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PKG, "scripts"))
from registration import register                    # noqa: E402
from moco_solution import read_controls              # noqa: E402
import rstats                                        # noqa: E402
from lib.paper1io import RESULTS, CYCLES, ICS, TOE, crop_window   # noqa: E402
import compare_tfl_cappellini as CAP                 # noqa: E402  (constants + loaders only)

PLOTS = os.path.join(HERE, "plots")
RES = os.path.join(HERE, "results")
SRC = os.path.join(HERE, "digitized", "montgomery1994-tfl-running-4ms.csv")
TRACT = "tfl12_ITB_r"
N = 101
EMD_MS = 75.0
EMD_CONDITIONS = {0.0: "raw (PRIMARY)", EMD_MS: f"+{EMD_MS:.0f} ms EMD (secondary)"}
MONTGOMERY_SPEED, OUR_SPEED = 4.0, 3.0
OUR_STANCE = float(np.mean([(TOE[c] - ICS[c]) / (ICS[c + 1] - ICS[c]) for c in range(5)]) * 100)
CAPPELLINI_STANCE_APPROX = 34.0


def load_montgomery():
    bins, vals, phases = [], [], []
    with open(SRC) as f:
        for r in csv.DictReader(f):
            bins.append(int(r["bin"]))
            phases.append(r["phase"].strip())
            vals.append(float(r["activation_pct_mmt"]))
    n = len(bins)
    assert bins == sorted(bins) and bins[0] == 1, "bins must be 1..N in order"
    assert phases[:phases.count("Stance")] == ["Stance"] * phases.count("Stance"), \
        "stance bins must lead the table"
    n_stance = sum(1 for p in phases if p.lower() == "stance")
    edges = np.arange(n + 1) / n * 100.0          # 21 edges over 0-100%
    return edges, np.array(vals, float), n_stance / n


def warp_phase(x_pct, mg_stance):
    x = np.asarray(x_pct, float) / 100.0
    s = mg_stance
    out = np.where(x <= s,
                   x / s * (OUR_STANCE / 100.0),
                   (OUR_STANCE / 100.0) + (x - s) / (1.0 - s) * (1.0 - OUR_STANCE / 100.0))
    return out * 100.0


def step_on_grid(edges, vals, grid):
    idx = np.clip(np.searchsorted(edges, grid, side="right") - 1, 0, len(vals) - 1)
    return vals[idx]

def montgomery_on_our_phase(edges, vals, mg_stance, warp):
    e = warp_phase(edges, mg_stance) if warp else edges.copy()
    return e, step_on_grid(e, vals, np.linspace(0, 100, N))

def pearson(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def main():
    phase = np.linspace(0, 100, N)
    edges, mmt, mg_stance = load_montgomery()
    print(f"Montgomery et al. (1994) TFL, {len(mmt)} equal-duration bins "
          f"({100.0 / len(mmt):.0f}% of the cycle each), %MMT")
    print(f"  its stance = {mg_stance * 100:.1f}% of the cycle (at {MONTGOMERY_SPEED} m/s)")
    print(f"  ours       = {OUR_STANCE:.1f}% (at {OUR_SPEED} m/s)  -> warped onto ours for leg (a)\n")

    ref = {w: montgomery_on_our_phase(edges, mmt, mg_stance, w) for w in (True, False)}

    cx, cy = CAP.load_cappellini()
    cap_ref, cap_dev = CAP.spline_to_grid(cx, cy, phase)
    print(f"Cappellini et al. (2006) TFL at {CAP.CAPPELLINI_SPEED:.2f} m/s, splined to "
          f"{N} points (spline-vs-PCHIP dev {cap_dev:.4f})")
    print(f"  its stance ~ {CAPPELLINI_STANCE_APPROX:.0f}% (Fig. 1C graph-read; NOT applied "
          f"as a warp)\n")

    rows, ours_reg = [], {}
    for cycle in CYCLES:
        act = read_controls(os.path.join(RESULTS, "davis",
                                            f"cycle{cycle}_compliant_solution.sto"))
        t0, t1 = crop_window(cycle)
        for emd_ms in EMD_CONDITIONS:
            _, a = register(act.index.values + emd_ms / 1000.0, act[TRACT].values,
                            t0, t1, n=N)
            ours_reg[(cycle, emd_ms)] = a
            for warp in (True, False):
                rows.append(dict(cycle=cycle, emd_ms=emd_ms,
                                 alignment=("stance-warped" if warp else "uniform"),
                                 r=round(pearson(ref[warp][1], a), 4)))

    os.makedirs(RES, exist_ok=True)
    os.makedirs(PLOTS, exist_ok=True)
    out_csv = os.path.join(RES, "tfl_montgomery_comparison.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("(a) our tfl12_ITB vs MONTGOMERY\n")
    print(f"{'alignment':<16}{'EMD':>10}{'r (mean +/- SD over 5 cycles)':>34}")
    print("-" * 60)
    summary = {}
    for align in ("stance-warped", "uniform"):
        for emd_ms, desc in EMD_CONDITIONS.items():
            v = [r["r"] for r in rows if r["alignment"] == align and r["emd_ms"] == emd_ms]
            summary[(align, emd_ms)] = (rstats.fisher_mean(v), rstats.raw_sd(v))
            print(f"{align:<16}{desc:>10}{rstats.fisher_mean(v):>24.2f} "
                  f"+/- {rstats.raw_sd(v):.2f}")
    ref_rows = [
        dict(comparison="Montgomery1994_vs_Cappellini2006", role="primary",
             alignment="raw (neither warped)",
             r=round(pearson(ref[False][1], cap_ref), 4),
             note="both on their own phase axis; their stance fractions "
                  f"({mg_stance * 100:.0f}% vs ~{CAPPELLINI_STANCE_APPROX:.0f}%) agree to "
                  "within a percentage point"),
        dict(comparison="Montgomery1994_vs_Cappellini2006", role="sensitivity",
             alignment="Montgomery stance-warped onto ours",
             r=round(pearson(ref[True][1], cap_ref), 4),
             note="warps one side only; do not quote"),
    ]
    ref_csv = os.path.join(RES, "tfl_reference_disagreement.csv")
    with open(ref_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ref_rows[0].keys()))
        w.writeheader()
        w.writerows(ref_rows)

    print("\n(c) MONTGOMERY vs CAPPELLINI  (reference vs reference, no cycles, no EMD)\n")
    for r in ref_rows:
        print(f"  {r['role'].upper():<12}{r['alignment']:<38} r = {r['r']:+.2f}")
    r_ref = [r for r in ref_rows if r["role"] == "primary"][0]["r"]

    cap_csv = os.path.join(RES, "tfl_cappellini_comparison.csv")
    r_cap = None
    if os.path.exists(cap_csv):
        with open(cap_csv) as f:
            crows = list(csv.DictReader(f))
        r_cap = {e: rstats.fisher_mean([float(r["r"]) for r in crows
                                        if float(r["emd_ms"]) == e]) for e in EMD_CONDITIONS}
        print("\n(b) our tfl12_ITB vs CAPPELLINI  (from compare_tfl_cappellini.py)\n")
        for e, d in EMD_CONDITIONS.items():
            print(f"  {d:<38} r = {r_cap[e]:+.2f}")
    else:
        print("\n(b) run compare_tfl_cappellini.py first to complete the three-way table")

    print("\nTHREE-WAY SUMMARY (raw / unshifted, the condition all three share)")
    print(f"  ours vs Montgomery   r = {summary[('stance-warped', 0.0)][0]:+.2f}")
    if r_cap is not None:
        print(f"  ours vs Cappellini   r = {r_cap[0.0]:+.2f}")
    ours_r = [summary[("stance-warped", 0.0)][0]] + ([r_cap[0.0]] if r_cap else [])
    if r_ref > max(ours_r):
        print(f"  Montgomery vs Cappellini  r = {r_ref:+.2f}   <- the two references agree "
              "with EACH OTHER better than either agrees with us.")
        print("     => do NOT write 'the references do not agree with one another'. Both "
              "put the TFL burst in\n        early stance; our model puts it in mid-swing. "
              "One disagreement, two independent instances.")
    else:
        print(f"  Montgomery vs Cappellini  r = {r_ref:+.2f}   <- the references disagree "
              "with each other too")

    # ---- figure 
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True)
    for ax, emd_ms in zip(axes, EMD_CONDITIONS):
        for c in CYCLES:
            ax.plot(phase, ours_reg[(c, emd_ms)], color="C3", lw=0.7, alpha=0.40)
        ax.plot(phase, np.mean([ours_reg[(c, emd_ms)] for c in CYCLES], axis=0),
                color="C3", lw=2.6, label=f"our {TRACT} excitation")
        ax.axvline(OUR_STANCE, color="0.4", ls=":", lw=1.1)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, 100)
        ax.set_xlabel("% gait cycle")
        ax.set_ylabel("fraction of maximum\n(model excitation  /  Montgomery %MMT / 100)")
        m, sd = summary[("stance-warped", emd_ms)]
        ax.set_title(f"{EMD_CONDITIONS[emd_ms]}   r = {m:.2f} +/- {sd:.2f}", fontsize=11)
        e = ref[True][0]
        ax.step(np.append(e, 100.0),
                np.append(np.append(mmt, mmt[-1]), mmt[-1])[:len(e) + 1] / 100.0,
                where="post", color="k", lw=2.2, ls=":",
                label=f"Montgomery 1994, {MONTGOMERY_SPEED} m/s (%MMT/100, {len(mmt)} bins)")
        if emd_ms == 0.0:
            ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    fig.suptitle("tfl12_ITB vs Montgomery et al. (1994) group-average TFL EMG - one weak "
                 "external reference, not a validation\n"
                 f"different subjects and {MONTGOMERY_SPEED} vs {OUR_SPEED} m/s; both are "
                 f"fractions of a maximum so they share one 0-1 axis; thin = our 5 cycles",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out_png = os.path.join(PLOTS, "tfl_vs_montgomery.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    # ---- figure 2
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharex=True, sharey=True)
    for ax, emd_ms in zip(axes, EMD_CONDITIONS):
        for c in CYCLES:
            ax.plot(phase, ours_reg[(c, emd_ms)], color="C3", lw=0.7, alpha=0.35)
        ax.plot(phase, np.mean([ours_reg[(c, emd_ms)] for c in CYCLES], axis=0),
                color="C3", lw=2.8, label=f"our {TRACT} (n=5 cycles, {OUR_SPEED} m/s)")
        e = ref[True][0]
        ax.step(np.append(e, 100.0),
                np.append(np.append(mmt, mmt[-1]), mmt[-1])[:len(e) + 1] / 100.0,
                where="post", color="k", lw=2.2, ls=":",
                label=f"Montgomery 1994 ({MONTGOMERY_SPEED} m/s, %MMT/100, 20 bins)")
        ax.plot(phase, cap_ref, color="C0", lw=2.2, ls="--",
                label=f"Cappellini 2006 ({CAP.CAPPELLINI_SPEED:.2f} m/s, own-peak normalised)")
        ax.axvline(OUR_STANCE, color="0.4", ls=":", lw=1.1)
        ax.text(OUR_STANCE, 1.07, "our toe-off", fontsize=8, color="0.35", ha="center")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1.12)
        ax.set_xlabel("% gait cycle")
        bits = [f"ours-Montgomery r = {summary[('stance-warped', emd_ms)][0]:+.2f}"]
        if r_cap is not None:
            bits.append(f"ours-Cappellini r = {r_cap[emd_ms]:+.2f}")
        ax.set_title(f"{EMD_CONDITIONS[emd_ms]}\n" + "   ".join(bits), fontsize=10)
    axes[0].set_ylabel("fraction of maximum")
    axes[0].legend(fontsize=8, loc="upper left", framealpha=0.9)
    verdict = ("both references agree with each other more closely than either agrees "
               "with us" if (r_cap is None or r_ref > max(summary[('stance-warped', 0.0)][0],
                                                          r_cap[0.0]))
               else "the references disagree with each other as strongly as either "
                    "disagrees with us")
    fig.suptitle("TFL: our model against BOTH literature references, and the references "
                 "against each other\n"
                 f"Montgomery vs Cappellini r = {r_ref:+.2f} - {verdict}; both put the main "
                 "burst in early stance, our model puts it in mid-swing",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out_png3 = os.path.join(PLOTS, "tfl_three_way.png")
    fig.savefig(out_png3, dpi=150)
    plt.close(fig)

    for p in (out_csv, ref_csv, out_png, out_png3):
        print(f"saved: {os.path.relpath(p, HERE)}")


if __name__ == "__main__":
    main()
