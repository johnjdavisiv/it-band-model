#!/usr/bin/env python
"""
Process raw web-plot-digitized Eng et al. (2015) *SIMM MODEL* moment-arm curves
into a uniform tidy ("long") table -- the companion to the cadaver bands produced
by process_eng_moment_arms.py.

These 9 files (3 model tracts x 3 DOFs) are what Eng's own ORIGINAL SIMM model
reported for the moment arms of glmax12-itb / glmax34-itb / tfl12-itb about hip
flexion, hip adduction, and knee flexion. Unlike the cadaver data (upper/lower
uncertainty band per sub-tract), each SIMM file is a SINGLE model curve, so there
is one deg-4 fit per (tract, DOF) -- no band, no intersection grid.

Method (per tract, per DOF), matching the cadaver processing:
  1. fit a 4th-degree polynomial (matching Eng's own experimental fits),
  2. resample onto a 201-pt grid spanning the digitized x-range (never extrapolated).

CONVENTION: identical to the cadaver data -- Eng is already in our model's
convention on all three axes (hip +flexion / +adduction, knee +flexion), so NO
sign flips are applied.

Outputs (next to this script):
  eng-simm-model-moment-arms-long.csv   tidy long table (single line per tract/DOF)
  eng-simm-fit-provenance.json          poly coeffs, residuals, x-range, anomaly flags
  plots/eng-simm-model-fits.png         3x3 grid: digitized points vs deg-4 fit
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from numpy.polynomial import polynomial as P

HERE = os.path.dirname(os.path.abspath(__file__))
POLY_DEG = 4
N_GRID = 201

GROUP_TO_MODEL_TRACT = {
    "glmax12": "glmax12_ITB",
    "glmax34": "glmax34_ITB",
    "tfl12": "tfl12_ITB",
}
COORD_ORDER = ["hip_flexion", "hip_adduction", "knee_flexion"]
GROUP_ORDER = ["glmax12", "glmax34", "tfl12"]

OUTLIER_ABS_CM = 0.20
OUTLIER_REL = 4.0


def parse_file(path):
    """Return (group, model_tract, coord_key, dataframe with std columns).
    Tolerant of the '-itb' being present or absent in the filename."""
    base = os.path.basename(path)
    m = re.match(r"eng-(glmax12|glmax34|tfl12)-.*-SIMM-raw\.csv", base)
    group = m.group(1)
    df = pd.read_csv(path)
    angle_col = [c for c in df.columns if c.endswith("_angle_deg")][0]
    ma_col = [c for c in df.columns if c.endswith("_moment_arm_cm")][0]
    coord_key = angle_col[: -len("_angle_deg")]  # e.g. hip_adduction
    df = df.rename(columns={angle_col: "angle_deg", ma_col: "moment_arm_cm"})
    return group, GROUP_TO_MODEL_TRACT[group], coord_key, df


def fit_series(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    series = P.Polynomial.fit(x, y, POLY_DEG)
    resid = y - series(x)
    rms = float(np.sqrt(np.mean(resid ** 2)))
    outliers = []
    thr = max(OUTLIER_ABS_CM, OUTLIER_REL * rms)
    for xi, yi, ri in zip(x, y, resid):
        if abs(ri) > thr:
            outliers.append(dict(angle_deg=round(float(xi), 3),
                                 moment_arm_cm=round(float(yi), 4),
                                 residual_cm=round(float(ri), 4)))
    return series, x, y, rms, float(np.max(np.abs(resid))), outliers


def main():
    files = sorted(glob.glob(os.path.join(HERE, "eng-*-SIMM-raw.csv")))
    long_rows = []
    provenance = {
        "description": "Eng et al. (2015) ORIGINAL SIMM MODEL moment-arm curves "
                       "(3 tracts x 3 DOFs), deg-4 poly fit + 201-pt resample over "
                       "the digitized x-range, in our model's convention (no sign "
                       "flips -- Eng data is natively compatible; verified vs step7).",
        "poly_degree": POLY_DEG, "n_grid": N_GRID,
        "grid_rule": "per (tract, DOF): linspace over the single curve's digitized x-range",
        "coeff_basis": "power, ascending (moment_arm_cm = sum c[i]*angle_deg**i)",
        "sign_flips": "none (angle x1, moment_arm x1 on all axes)",
        "source": "digitized SIMM-model lines from Eng et al. (2015) figures",
        "panels": {},
    }
    fit_store = {}
    anomalies = []

    for path in files:
        group, model_tract, coord_key, df = parse_file(path)
        sub = df["muscle"].iloc[0]  # single tract per file, e.g. 'gmax12-itb'
        series, x, y, rms, mx, outs = fit_series(df["angle_deg"], df["moment_arm_cm"])
        grid = np.linspace(x.min(), x.max(), N_GRID)
        g = series(grid)

        for o in outs:
            anomalies.append(f"[{group} {coord_key}] outlier pt angle={o['angle_deg']} "
                             f"ma={o['moment_arm_cm']} resid={o['residual_cm']} cm")

        for i in range(N_GRID):
            long_rows.append(dict(
                model_tract=model_tract, muscle=sub,
                angle_type=coord_key, moment_arm_type=coord_key,
                angle_deg=round(float(grid[i]), 6),
                moment_arm_cm=round(float(g[i]), 6),
                grid_idx=i))

        pkey = f"{group}|{coord_key}"
        provenance["panels"][pkey] = dict(
            raw_file=os.path.basename(path), model_tract=model_tract, coord=coord_key,
            muscle=sub, n_pts=int(len(x)),
            x_range=[round(float(x.min()), 3), round(float(x.max()), 3)],
            coeffs=series.convert().coef.tolist(),
            resid_rms_cm=round(rms, 5), resid_max_cm=round(mx, 5))
        fit_store[(group, coord_key)] = dict(x=x, y=y, grid=grid, g=g)
        print(f"{group:7s} {coord_key:13s} {sub:10s}  n={len(x):2d}  "
              f"RMS={rms:.3f}  max={mx:.3f}  x=[{x.min():.1f},{x.max():.1f}]")

    out_df = pd.DataFrame(long_rows, columns=[
        "model_tract", "muscle", "angle_type", "moment_arm_type",
        "angle_deg", "moment_arm_cm", "grid_idx"])
    csv_path = os.path.join(HERE, "eng-simm-model-moment-arms-long.csv")
    out_df.to_csv(csv_path, index=False)
    provenance["anomaly_report"] = anomalies
    with open(os.path.join(HERE, "eng-simm-fit-provenance.json"), "w") as f:
        json.dump(provenance, f, indent=2)

    print(f"\nwrote {csv_path}  ({len(out_df)} rows)")
    print("\n===== ANOMALY REPORT =====")
    print("  " + ("\n  ".join(anomalies) if anomalies
                  else "(none: all deg-4 fits clean, no outlier points)"))

    # ---- 3x3 verification figure -------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for r, group in enumerate(GROUP_ORDER):
        for c, coord in enumerate(COORD_ORDER):
            ax = axes[r][c]
            fs = fit_store.get((group, coord))
            if fs:
                ax.plot(fs["x"], fs["y"], "o", ms=5, color="#333", alpha=0.7, label="digitized")
                ax.plot(fs["grid"], fs["g"], "-", lw=1.8, color="#d62728", label="deg-4 fit")
                ax.legend(fontsize=7, loc="best")
            ax.axhline(0, color="0.8", lw=0.8, zorder=0)
            ax.grid(alpha=0.2)
            if r == 2:
                ax.set_xlabel(f"{coord.replace('_', ' ')} angle (deg)")
            if c == 0:
                ax.set_ylabel(f"{GROUP_TO_MODEL_TRACT[group]}\nmoment arm (cm)")
            ax.set_title(f"{GROUP_TO_MODEL_TRACT[group]} — {coord}", fontsize=9)
    fig.suptitle("Eng SIMM-model moment-arm curves: digitized points vs deg-4 fits "
                 "(model convention)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plotdir = os.path.join(HERE, "plots")
    os.makedirs(plotdir, exist_ok=True)
    fig_path = os.path.join(plotdir, "eng-simm-model-fits.png")
    fig.savefig(fig_path, dpi=130)
    print(f"\nwrote {fig_path}")


if __name__ == "__main__":
    main()
