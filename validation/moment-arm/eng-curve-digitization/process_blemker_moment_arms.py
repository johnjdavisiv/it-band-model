#!/usr/bin/env python
"""
Process raw web-plot-digitized Blemker gluteus-maximus moment-arm bounds into a
uniform, tidy ("long") table, converted into OUR MODEL's coordinate convention.

Source: Blemker & Delp 3D-model gluteus-maximus moment-arm panels. Each panel is
the upper/lower envelope of the cyan "3D Model Gluteus Maximus Fibers" band,
manually digitized in WebPlotDigitizer -> different, irregular #points per bound.

For each (coordinate, bound) we:
  1. fit a 4th-degree polynomial to the digitized points
     (degree 4 chosen for consistency with Eng et al.'s experimental fits),
  2. re-sample the polynomial onto a uniform 201-point grid over Blemker's ROM,
  3. cast into our model's sign convention.

CONVENTION MAPPING (verified empirically against Eng_2015_model_step7.osim, whose
hip coords are +flexion / +adduction / +internal-rotation, the Rajagopal lineage):

  raw file                       Blemker axis            -> model coord   angle  MA
  blemker-...-hip-extension-raw  hip EXTENSION MA         hip_flexion      x1     x-1
  blemker-...-hip-adduction-raw  hip ADDUCTION MA         hip_adduction    x1     x1
  blemker-...-hip-rotation-raw   hip INTERNAL-ROTATION MA hip_rotation     x1     x1

Only the extension panel flips sign (extension MA = -flexion MA). The angle axes
already match our model on all three; internal-rotation & adduction MAs are
already in our convention.

Outputs (written next to this script):
  blemker-gmax-moment-arms-long.csv   tidy long table (3 coords x 2 bounds x 201)
  blemker-gmax-fit-provenance.json    poly coeffs + fit residuals + mapping
  plots/blemker-gmax-moment-arm-fits.png  raw points vs fitted curves (model conv.)
"""
import json
import os

import numpy as np
import pandas as pd
from numpy.polynomial import polynomial as P

HERE = os.path.dirname(os.path.abspath(__file__))
POLY_DEG = 4
N_GRID = 201

# One entry per digitized panel.
PANELS = [
    dict(
        key="hip_flexion",
        raw="blemker-gmax-hip-extension-raw.csv",
        angle_col="hip_flexion_angle_deg",
        ma_col="hip_extension_moment_arm_cm",
        blemker_angle_label="hip flexion angle (deg)",
        blemker_ma_label="hip extension moment arm (cm)",
        angle_sign=1.0,   # Blemker flexion angle == model hip_flexion
        ma_sign=-1.0,     # Blemker EXTENSION MA == -1 * model flexion MA
        grid=(0.0, 100.0),
    ),
    dict(
        key="hip_adduction",
        raw="blemker-gmax-hip-adduction-raw.csv",
        angle_col="hip_adduction_angle_deg",
        ma_col="hip_adduction_moment_arm_cm",
        blemker_angle_label="hip adduction angle (deg)",
        blemker_ma_label="hip adduction moment arm (cm)",
        angle_sign=1.0,
        ma_sign=1.0,
        grid=(-40.0, 40.0),
    ),
    dict(
        key="hip_rotation",
        raw="blemker-gmax-hip-rotation-raw.csv",
        angle_col="hip_internal_rotation_angle_deg",
        ma_col="hip_internal_rotation_moment_arm_cm",
        blemker_angle_label="hip internal rotation angle (deg)",
        blemker_ma_label="hip internal rotation moment arm (cm)",
        angle_sign=1.0,
        ma_sign=1.0,
        grid=(-30.0, 30.0),
    ),
]

BOUNDS = ["lower", "upper"]


def fit_and_resample(angle_native, ma_native, panel):
    """Convert to model convention, fit deg-4 poly, resample to the uniform grid.

    Returns (grid_angle, grid_ma, coeffs_ascending, resid_rms, resid_max,
             n_points, model_angle_pts, model_ma_pts).
    """
    ang = panel["angle_sign"] * np.asarray(angle_native, float)
    ma = panel["ma_sign"] * np.asarray(ma_native, float)
    order = np.argsort(ang)
    ang, ma = ang[order], ma[order]

    # numpy.polynomial.Polynomial.fit auto-scales the domain -> well-conditioned.
    series = P.Polynomial.fit(ang, ma, POLY_DEG)
    fit_at_pts = series(ang)
    resid = ma - fit_at_pts
    resid_rms = float(np.sqrt(np.mean(resid ** 2)))
    resid_max = float(np.max(np.abs(resid)))

    lo, hi = panel["grid"]
    grid_angle = np.linspace(lo, hi, N_GRID)
    grid_ma = series(grid_angle)

    coeffs_ascending = series.convert().coef.tolist()  # standard power basis, c0..c4
    return (grid_angle, grid_ma, coeffs_ascending, resid_rms, resid_max,
            int(len(ang)), ang, ma)


def main():
    long_rows = []
    provenance = {
        "description": "Blemker gluteus-maximus moment-arm bounds, fit with a "
                       "4th-degree polynomial (per Eng et al.) and resampled onto "
                       "a uniform 201-point grid, in our model's coordinate "
                       "convention (hip +flexion/+adduction/+internal-rotation).",
        "poly_degree": POLY_DEG,
        "n_grid": N_GRID,
        "coeff_basis": "power, ascending (moment_arm_cm = sum c[i]*angle_deg**i)",
        "bound_type_note": (
            "GOTCHA: bound_type ('lower'/'upper') labels Blemker's ORIGINAL fiber-"
            "band edge (min/max moment arm across fibers), NOT the visual top/bottom "
            "of the plotted curve. For the hip_flexion panel the MA is sign-flipped "
            "(extension->flexion), so the 'upper' bound (largest extension MA = most "
            "extensor) plots at the BOTTOM in model convention. adduction & rotation "
            "are not flipped, so their 'upper' stays on top."
        ),
        "panels": {},
    }
    fit_store = {}  # for plotting

    for panel in PANELS:
        raw_path = os.path.join(HERE, panel["raw"])
        df = pd.read_csv(raw_path)
        provenance["panels"][panel["key"]] = {
            "raw_file": panel["raw"],
            "blemker_angle_axis": panel["blemker_angle_label"],
            "blemker_moment_arm_axis": panel["blemker_ma_label"],
            "angle_sign_to_model": panel["angle_sign"],
            "moment_arm_sign_to_model": panel["ma_sign"],
            "grid_deg": list(panel["grid"]),
            "bounds": {},
        }
        fit_store[panel["key"]] = {}

        for bound in BOUNDS:
            sub = df[df["bound_type"] == bound]
            (grid_angle, grid_ma, coeffs, rms, rmax, npts,
             ang_pts, ma_pts) = fit_and_resample(
                sub[panel["angle_col"]].values,
                sub[panel["ma_col"]].values,
                panel,
            )
            for i in range(N_GRID):
                long_rows.append(dict(
                    angle_type=panel["key"],
                    moment_arm_type=panel["key"],
                    angle_deg=round(float(grid_angle[i]), 6),
                    moment_arm_cm=round(float(grid_ma[i]), 6),
                    bound_type=bound,
                    grid_idx=i,
                ))
            provenance["panels"][panel["key"]]["bounds"][bound] = dict(
                n_digitized_points=npts,
                poly_coeffs=coeffs,
                resid_rms_cm=round(rms, 5),
                resid_max_cm=round(rmax, 5),
            )
            fit_store[panel["key"]][bound] = dict(
                ang_pts=ang_pts, ma_pts=ma_pts,
                grid_angle=grid_angle, grid_ma=grid_ma,
            )
            print(f"{panel['key']:14s} {bound:5s}  n={npts:2d}  "
                  f"RMS={rms:.3f} cm  max={rmax:.3f} cm")

    out_df = pd.DataFrame(long_rows, columns=[
        "angle_type", "moment_arm_type", "angle_deg",
        "moment_arm_cm", "bound_type", "grid_idx"])
    csv_path = os.path.join(HERE, "blemker-gmax-moment-arms-long.csv")
    out_df.to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}  ({len(out_df)} rows)")

    prov_path = os.path.join(HERE, "blemker-gmax-fit-provenance.json")
    with open(prov_path, "w") as f:
        json.dump(provenance, f, indent=2)
    print(f"wrote {prov_path}")

    # ---- verification figure -------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plotdir = os.path.join(HERE, "plots")
    os.makedirs(plotdir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    titles = {
        "hip_flexion": "Hip flexion MA vs hip flexion angle\n(Blemker extension MA x -1)",
        "hip_adduction": "Hip adduction MA vs hip adduction angle",
        "hip_rotation": "Hip internal-rot MA vs hip internal-rot angle",
    }
    colors = {"lower": "#1b7837", "upper": "#762a83"}
    for ax, panel in zip(axes, PANELS):
        key = panel["key"]
        for bound in BOUNDS:
            fs = fit_store[key][bound]
            ax.plot(fs["ang_pts"], fs["ma_pts"], "o", ms=5,
                    color=colors[bound], alpha=0.55,
                    label=f"{bound} digitized (n={len(fs['ang_pts'])})")
            ax.plot(fs["grid_angle"], fs["grid_ma"], "-", lw=2,
                    color=colors[bound], label=f"{bound} deg-4 fit")
        ax.axhline(0, color="0.7", lw=0.8, zorder=0)
        ax.set_title(titles[key], fontsize=10)
        ax.set_xlabel(f"{key.replace('hip_','hip ')} angle (deg)")
        ax.set_ylabel("moment arm (cm), model convention")
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.25)
    fig.suptitle("Blemker gluteus-maximus moment-arm bounds: digitized points vs "
                 "deg-4 fits (our model convention)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = os.path.join(plotdir, "blemker-gmax-moment-arm-fits.png")
    fig.savefig(fig_path, dpi=140)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
