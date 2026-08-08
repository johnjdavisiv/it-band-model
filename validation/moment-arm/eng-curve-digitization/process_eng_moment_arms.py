#!/usr/bin/env python
"""
Process raw web-plot-digitized Eng et al. (2015) ITB moment-arm bands into a
uniform tidy ("long") table, compatible with the Blemker long-table conventions.

Unlike Blemker (one aggregate band per panel over a fixed ROM), Eng plots a
separate uncertainty band per *individual tract*, and each tract's x-range depends
on how the experiment sampled it. Our model collapses tract pairs into one tract
(gmax1-itb + gmax2-itb -> glmax12_ITB, gmax3/4 -> glmax34_ITB, tfl1/2 -> tfl12_ITB),
so most panels show TWO partially-overlapping bands. Exception: tfl1-itb never
crossed the knee, so the tfl knee panel has only the tfl2-itb band.

Method (per tract, per bound):
  1. fit a 4th-degree polynomial (matching Eng's own experimental fits),
  2. build a common grid per tract = INTERSECTION of its upper & lower x-ranges
     (both bounds land on the same 201-pt grid so they can shade a band; using the
     intersection means neither bound is ever extrapolated -- the trimmed ends are
     sub-0.1-deg digitization fuzz),
  3. resample both bounds onto that 201-pt grid.

CONVENTION: verified against Eng_2015_model_step7.osim -- Eng's data is already in
our model's convention on ALL THREE axes (hip +flexion / +adduction, knee
+flexion), so NO sign flips are applied (contrast Blemker's extension->flexion x-1).
Confirmed empirically: model hip/knee MAs of every tract match Eng's signs.

Outputs (next to this script):
  eng-itb-moment-arms-long.csv     tidy long table
  eng-itb-fit-provenance.json      poly coeffs, residuals, grids, anomaly flags
  plots/eng-itb-moment-arm-fits.png  3x3 grid: digitized points vs fitted bands
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

# eng-<group>-itb-...  ->  model tract name
GROUP_TO_MODEL_TRACT = {
    "glmax12": "glmax12_ITB",
    "glmax34": "glmax34_ITB",
    "tfl12": "tfl12_ITB",
}
# display order
COORD_ORDER = ["hip_flexion", "hip_adduction", "knee_flexion"]
GROUP_ORDER = ["glmax12", "glmax34", "tfl12"]

# outlier flag: |residual| above this AND above 4*series-RMS -> candidate misclick
OUTLIER_ABS_CM = 0.20
OUTLIER_REL = 4.0


def parse_file(path):
    """Return (group, model_tract, coord_key, dataframe with std columns)."""
    base = os.path.basename(path)
    m = re.match(r"eng-(\w+?)-itb-(.+)-raw\.csv", base)
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
    # NB exclude the *-SIMM-raw.csv files: those are Eng's own SIMM-MODEL curves (single lines, no
    # uncertainty band) handled by process_eng_simm_moment_arms.py, and they carry no `bound_type`
    # column. The bare glob matches them (8 of 17 files here) -> KeyError: 'bound_type'. (fixed 2026-07-16)
    files = sorted(f for f in glob.glob(os.path.join(HERE, "eng-*-itb-*-raw.csv"))
                   if "SIMM" not in os.path.basename(f))
    long_rows = []
    provenance = {
        "description": "Eng et al. (2015) ITB tract moment-arm bands, deg-4 poly fit "
                       "+ 201-pt resample on a per-tract intersection grid, in our "
                       "model's convention (no sign flips applied -- Eng data is "
                       "natively compatible; verified vs step7).",
        "poly_degree": POLY_DEG, "n_grid": N_GRID,
        "grid_rule": "per tract: linspace over intersection of upper & lower x-ranges",
        "coeff_basis": "power, ascending (moment_arm_cm = sum c[i]*angle_deg**i)",
        "sign_flips": "none (angle x1, moment_arm x1 on all axes)",
        "bound_type_note": "upper/lower = numerically higher/lower moment arm (Eng's "
                           "band edges). No axis is sign-flipped, so upper always "
                           "plots on top (unlike the Blemker hip_flexion panel).",
        "panels": {},
    }
    fit_store = {}   # (group, coord) -> {subtract: {...}}
    anomalies = []

    for path in files:
        group, model_tract, coord_key, df = parse_file(path)
        subtracts = list(dict.fromkeys(df["muscle"]))  # preserve order
        pkey = f"{group}|{coord_key}"
        provenance["panels"][pkey] = dict(
            raw_file=os.path.basename(path), model_tract=model_tract,
            coord=coord_key, subtracts={})
        fit_store[(group, coord_key)] = {}

        for sub in subtracts:
            sdf = df[df["muscle"] == sub]
            up = sdf[sdf["bound_type"] == "upper"]
            lo = sdf[sdf["bound_type"] == "lower"]
            fit_u = fit_series(up["angle_deg"], up["moment_arm_cm"])
            fit_l = fit_series(lo["angle_deg"], lo["moment_arm_cm"])
            su, xu, yu, rmsu, maxu, outu = fit_u
            sl, xl, yl, rmsl, maxl, outl = fit_l

            xlo = max(xu.min(), xl.min())
            xhi = min(xu.max(), xl.max())
            if xhi <= xlo:
                anomalies.append(f"[{group} {coord_key} {sub}] DEGENERATE grid: "
                                 f"upper x[{xu.min():.2f},{xu.max():.2f}] vs "
                                 f"lower x[{xl.min():.2f},{xl.max():.2f}] do not overlap")
                continue
            grid = np.linspace(xlo, xhi, N_GRID)
            gu, gl = su(grid), sl(grid)

            # band-inversion check (upper should stay >= lower)
            inv = gu < gl - 1e-6
            if inv.any():
                bad = grid[inv]
                anomalies.append(f"[{group} {coord_key} {sub}] BAND INVERSION "
                                 f"(upper<lower) over {bad.min():.1f}..{bad.max():.1f} deg")
            for tag, out in (("upper", outu), ("lower", outl)):
                for o in out:
                    anomalies.append(f"[{group} {coord_key} {sub} {tag}] outlier "
                                     f"pt angle={o['angle_deg']} ma={o['moment_arm_cm']} "
                                     f"resid={o['residual_cm']} cm")

            for tag, gvals in (("upper", gu), ("lower", gl)):
                for i in range(N_GRID):
                    long_rows.append(dict(
                        model_tract=model_tract, muscle=sub,
                        angle_type=coord_key, moment_arm_type=coord_key,
                        angle_deg=round(float(grid[i]), 6),
                        moment_arm_cm=round(float(gvals[i]), 6),
                        bound_type=tag, grid_idx=i))

            provenance["panels"][pkey]["subtracts"][sub] = dict(
                grid_deg=[round(xlo, 4), round(xhi, 4)],
                upper=dict(n_pts=int(len(xu)), coeffs=su.convert().coef.tolist(),
                           resid_rms_cm=round(rmsu, 5), resid_max_cm=round(maxu, 5),
                           x_range=[round(float(xu.min()), 3), round(float(xu.max()), 3)]),
                lower=dict(n_pts=int(len(xl)), coeffs=sl.convert().coef.tolist(),
                           resid_rms_cm=round(rmsl, 5), resid_max_cm=round(maxl, 5),
                           x_range=[round(float(xl.min()), 3), round(float(xl.max()), 3)]))
            fit_store[(group, coord_key)][sub] = dict(
                xu=xu, yu=yu, xl=xl, yl=yl, grid=grid, gu=gu, gl=gl)
            print(f"{group:7s} {coord_key:13s} {sub:10s}  "
                  f"n(u/l)={len(xu):2d}/{len(xl):2d}  "
                  f"RMS(u/l)={rmsu:.3f}/{rmsl:.3f}  "
                  f"max(u/l)={maxu:.3f}/{maxl:.3f}  grid=[{xlo:.1f},{xhi:.1f}]")

    out_df = pd.DataFrame(long_rows, columns=[
        "model_tract", "muscle", "angle_type", "moment_arm_type",
        "angle_deg", "moment_arm_cm", "bound_type", "grid_idx"])
    csv_path = os.path.join(HERE, "eng-itb-moment-arms-long.csv")
    out_df.to_csv(csv_path, index=False)
    provenance["anomaly_report"] = anomalies
    with open(os.path.join(HERE, "eng-itb-fit-provenance.json"), "w") as f:
        json.dump(provenance, f, indent=2)

    print(f"\nwrote {csv_path}  ({len(out_df)} rows)")
    print("\n===== ANOMALY REPORT =====")
    if anomalies:
        for a in anomalies:
            print("  " + a)
    else:
        print("  (none: all fits clean, no band inversions, no outlier points)")

    # ---- 3x3 verification figure -------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub_colors = {
        "gmax1-itb": "#1f77b4", "gmax2-itb": "#ff7f0e",
        "gmax3-itb": "#2ca02c", "gmax4-itb": "#d62728",
        "tfl1-itb": "#9467bd", "tfl2-itb": "#8c564b",
    }
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for r, group in enumerate(GROUP_ORDER):
        for c, coord in enumerate(COORD_ORDER):
            ax = axes[r][c]
            store = fit_store.get((group, coord), {})
            for sub, fs in store.items():
                col = sub_colors.get(sub, "0.3")
                ax.plot(fs["xu"], fs["yu"], "o", ms=4, color=col, alpha=0.7)
                ax.plot(fs["xl"], fs["yl"], "s", ms=4, color=col, alpha=0.7,
                        markerfacecolor="none")
                ax.plot(fs["grid"], fs["gu"], "-", lw=1.6, color=col)
                ax.plot(fs["grid"], fs["gl"], "-", lw=1.6, color=col)
                ax.fill_between(fs["grid"], fs["gl"], fs["gu"], color=col,
                                alpha=0.15, label=sub)
            ax.axhline(0, color="0.8", lw=0.8, zorder=0)
            ax.grid(alpha=0.2)
            if store:
                ax.legend(fontsize=7, loc="best")
            if r == 2:
                ax.set_xlabel(f"{coord.replace('_', ' ')} angle (deg)")
            if c == 0:
                ax.set_ylabel(f"{GROUP_TO_MODEL_TRACT[group]}\nmoment arm (cm)")
            ax.set_title(f"{GROUP_TO_MODEL_TRACT[group]} — {coord}", fontsize=9)
    fig.suptitle("Eng ITB tract moment-arm bands: digitized points (o=upper, "
                 "□=lower) vs deg-4 fits (model convention)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plotdir = os.path.join(HERE, "plots")
    os.makedirs(plotdir, exist_ok=True)
    fig_path = os.path.join(plotdir, "eng-itb-moment-arm-fits.png")
    fig.savefig(fig_path, dpi=130)
    print(f"\nwrote {fig_path}")


if __name__ == "__main__":
    main()
