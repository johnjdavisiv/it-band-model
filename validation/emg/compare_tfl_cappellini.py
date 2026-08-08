"""
TFL vs lit estimates 

"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PKG, "scripts"))
from registration import register                    # noqa: E402
from moco_solution import read_controls              # noqa: E402
from lib.paper1io import RESULTS, CYCLES, ICS, TOE, crop_window   # noqa: E402
import rstats                                        # noqa: E402

PLOTS = os.path.join(HERE, "plots")
RES = os.path.join(HERE, "results")
SRC = os.path.join(HERE, "digitized", "cappellini-emg-direct-raw.csv")
TRACT = "tfl12_ITB_r"
N = 101
EMD_MS = 75.0
EMD_CONDITIONS = {0.0: "raw (PRIMARY)", EMD_MS: f"+{EMD_MS:.0f} ms EMD (secondary)"}
CAPPELLINI_KMH = 12.0
CAPPELLINI_SPEED = CAPPELLINI_KMH / 3.6
OUR_SPEED = 3.0
OUR_STANCE = float(np.mean([(TOE[c] - ICS[c]) / (ICS[c + 1] - ICS[c]) for c in range(5)]) * 100)
SPLINE_TOL = 0.03


def load_cappellini():
    x, y = [], []
    with open(SRC) as f:
        for r in csv.DictReader(f):
            x.append(float(r["percent_stance"]))
            y.append(float(r["muscle_activation"]))
    x, y = np.asarray(x, float), np.asarray(y, float)
    assert np.all(np.diff(x) > 0), "digitized phase must be strictly increasing"
    assert 0.0 <= x[0] and x[-1] <= 1.0 and x[-1] > 0.9, \
        f"expected a 0-1 cycle fraction, got [{x[0]}, {x[-1]}]"
    return x * 100.0, y


def spline_to_grid(x_pct, y, grid):
    yp = y.copy()
    yp[0] = yp[-1] = 0.5 * (y[0] + y[-1])
    out = CubicSpline(x_pct, yp, bc_type="periodic")(grid)
    ref = PchipInterpolator(x_pct, y)(grid)
    dev = float(np.abs(out - ref).max())
    if dev > SPLINE_TOL:
        raise AssertionError(f"spline deviates {dev:.4f} from PCHIP (> {SPLINE_TOL}) "
                             "-- it is not interpolating, it is inventing shape")
    if out.min() < 0.0:
        raise AssertionError(f"spline undershoots to {out.min():.4f}; an EMG envelope "
                             "cannot be negative")
    return out, dev

def pearson(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def main():
    x_pct, y_raw = load_cappellini()
    phase = np.linspace(0, 100, N)
    ref, dev = spline_to_grid(x_pct, y_raw, phase)

    print(f"Cappellini et al. (2006) TFL, running at {CAPPELLINI_KMH:.0f} km/h "
          f"= {CAPPELLINI_SPEED:.2f} m/s (ours {OUR_SPEED:.1f} m/s)")
    print(f"  {len(x_pct)} digitized points, spacing {np.diff(x_pct).min():.2f}-"
          f"{np.diff(x_pct).max():.2f}% of the cycle")
    print(f"  -> periodic cubic spline on {N} even points; max |spline - PCHIP| = {dev:.4f}")
    print(f"  endpoint wrap closed by {abs(y_raw[-1] - y_raw[0]):.4f} "
          f"(y(0)={y_raw[0]:.4f}, y(100)={y_raw[-1]:.4f})")
    print(f"  reference peak {ref.max():.3f} at {phase[ref.argmax()]:.0f}% of the cycle")
    print("  plotted RAW: no phase warp (its duty factor is not ours) and no EMD shift "
          "(stride duration unknown)\n")
    print(f"  our stance ends at {OUR_STANCE:.1f}% of the cycle at {OUR_SPEED:.1f} m/s\n")

    rows, ours_reg = [], {}
    for cycle in CYCLES:
        act = read_controls(os.path.join(RESULTS, "davis",
                                            f"cycle{cycle}_compliant_solution.sto"))
        t0, t1 = crop_window(cycle)
        for emd_ms in EMD_CONDITIONS:
            # shifting EMG later == shifting the model earlier; the reference here has no
            # clock, so the delay is applied to OUR window instead (equivalent, one signal)
            _, a = register(act.index.values + emd_ms / 1000.0, act[TRACT].values,
                            t0, t1, n=N)
            ours_reg[(cycle, emd_ms)] = a
            rows.append(dict(cycle=cycle, emd_ms=emd_ms, r=round(pearson(ref, a), 4)))

    os.makedirs(RES, exist_ok=True)
    os.makedirs(PLOTS, exist_ok=True)
    out_csv = os.path.join(RES, "tfl_cappellini_comparison.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    ref_csv = os.path.join(RES, "tfl_cappellini_reference.csv")
    with open(ref_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pct_gait_cycle", "tfl_emg_normalized"])
        w.writerows([[round(p, 4), round(v, 6)] for p, v in zip(phase, ref)])

    print(f"{'EMD':>12}{'r (mean ± SD over 5 cycles)':>34}")
    print("-" * 46)
    summary = {}
    for emd_ms, desc in EMD_CONDITIONS.items():
        v = [r["r"] for r in rows if r["emd_ms"] == emd_ms]
        summary[emd_ms] = (rstats.fisher_mean(v), rstats.raw_sd(v))
        print(f"{desc:>12}{rstats.fisher_mean(v):>24.2f} ± {rstats.raw_sd(v):.2f}")

    # ---- figure -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharex=True, sharey=True)
    for ax, emd_ms in zip(axes, EMD_CONDITIONS):
        for c in CYCLES:
            ax.plot(phase, ours_reg[(c, emd_ms)], color="C3", lw=0.7, alpha=0.40)
        ax.plot(phase, np.mean([ours_reg[(c, emd_ms)] for c in CYCLES], axis=0),
                color="C3", lw=2.6, label=f"our {TRACT} excitation")
        ax.plot(phase, ref, color="k", lw=2.2, ls=":",
                label=f"Cappellini 2006 TFL EMG, {CAPPELLINI_KMH:.0f} km/h "
                      f"({CAPPELLINI_SPEED:.2f} m/s)\nraw phase, NOT EMD-shifted")
        ax.axvline(OUR_STANCE, color="0.4", ls=":", lw=1.1)
        ax.text(OUR_STANCE, 1.10, "our toe-off", fontsize=8, color="0.35",
                ha="center", va="bottom")
        ax.set_ylim(0, 1.09)
        ax.set_xlim(0, 100)
        ax.set_xlabel("% gait cycle  (0 = ipsilateral foot contact)")
        m, sd = summary[emd_ms]
        ax.set_title(f"our excitation: {EMD_CONDITIONS[emd_ms]}   r = {m:.2f} ± {sd:.2f}",
                     fontsize=11, pad=16)
    axes[0].set_ylabel("fraction of maximum\n(model excitation  /  normalized TFL EMG)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.085))
    fig.suptitle(
        "tfl12_ITB vs Cappellini et al. (2006) running TFL EMG — one weak external "
        "reference, not a validation\n"
        f"different subjects and {CAPPELLINI_SPEED:.2f} vs {OUR_SPEED:.1f} m/s; both are "
        "fractions of a maximum so they share one 0-1 axis; thin = our 5 cycles",
        fontsize=11)
    fig.text(0.5, 0.015,
             "The reference is drawn RAW. It is NOT phase-warped — its duty factor at "
             f"{CAPPELLINI_KMH:.0f} km/h is not ours, so only 0% (foot contact) is a shared "
             "landmark and the two phase axes drift apart after it.\nIt is NOT advanced by "
             "75 ms for electromechanical delay as the other EMG in this validation is: "
             "that shift is in seconds and the source reports no stride duration to convert "
             "it. Read timing qualitatively.",
             ha="center", va="bottom", fontsize=8.5, color="0.25")
    fig.tight_layout(rect=(0, 0.145, 1, 0.93))
    out_png = os.path.join(PLOTS, "tfl_vs_cappellini.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    print(f"\nsaved: {os.path.relpath(out_csv, HERE)}"
          f"\nsaved: {os.path.relpath(ref_csv, HERE)}"
          f"\nsaved: {os.path.relpath(out_png, HERE)}")

if __name__ == "__main__":
    main()
