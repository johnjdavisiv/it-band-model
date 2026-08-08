"""
Numeric moment-arm validation of the ITB model against digitized literature data

"""
import os
import math
import warnings

import numpy as np
import pandas as pd
import opensim as osim

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(HERE, "..", "..", "models")
DIGI = os.path.join(HERE, "eng-curve-digitization")

MODELS = {  # key -> (path, one-line role)
    # Shipped names: Eng2015_replica is the exact replica of Eng's SIMM model (build step 2);
    # Davis2026 is the final modernized model (build step 14).
    "step2":  (os.path.join(CLEAN, "Eng2015_replica.osim"),
               "exact replica (pre-modernization baseline)"),
    "step14": (os.path.join(CLEAN, "Davis2026.osim"),
               "current best (bilateral graft on RajagopalLaiUhlrich2023)"),
}

# step14's grafted tracts carry an _r suffix; step2 (single-limb) does not.
def muscle_name(model_key, tract):
    return tract + "_r" if model_key == "step14" else tract

COORD_OSIM = {
    "hip_flexion": "hip_flexion_r",
    "hip_adduction": "hip_adduction_r",
    "hip_rotation": "hip_rotation_r",
    "knee_flexion": "knee_angle_r",
}
NEUTRAL = ["hip_flexion_r", "hip_adduction_r", "hip_rotation_r", "knee_angle_r"]

ENG_TRACTS = ["glmax12_ITB", "glmax34_ITB", "tfl12_ITB"]
ENG_DOFS = ["hip_flexion", "hip_adduction", "knee_flexion"]

# Blemker gmax band applies to every glute-max tract (femoral + ITB); TFL is qualitative.
BLEMKER_TRACTS = ["glmax1", "glmax2", "glmax3", "glmax4", "glmax12_ITB", "glmax34_ITB"]
BLEMKER_DOFS = ["hip_flexion", "hip_adduction", "hip_rotation"]

N_UNION = 401 # envelope common grid
N_SWEEP = 201 # saved model sweep resolution

def load_model(path):
    m = osim.Model(path)
    for c in NEUTRAL:
        try:
            m.getCoordinateSet().get(c).set_clamped(False)
        except Exception:
            pass
    s = m.initSystem()
    return m, s

def model_ma(m, s, muscle, coord, angles_deg):
    for c in NEUTRAL:
        m.getCoordinateSet().get(c).setValue(s, 0.0, False)
    mu = m.getMuscles().get(muscle)
    co = m.getCoordinateSet().get(coord)
    out = np.empty(len(angles_deg))
    for i, a in enumerate(angles_deg):
        co.setValue(s, math.radians(float(a)), False)
        m.realizePosition(s)
        out[i] = mu.computeMomentArm(s, co) * 100.0
    co.setValue(s, 0.0, False)
    return out

def subtract_band(df, muscle):
    up = df[(df.muscle == muscle) & (df.bound_type == "upper")].sort_values("angle_deg")
    lo = df[(df.muscle == muscle) & (df.bound_type == "lower")].sort_values("angle_deg")
    a, b = lo.moment_arm_cm.values, up.moment_arm_cm.values
    return up.angle_deg.values, np.minimum(a, b), np.maximum(a, b)


def interp_masked(xq, x, y):
    out = np.interp(xq, x, y)
    out[(xq < x.min()) | (xq > x.max())] = np.nan
    return out


def panel_rmse(m, s, muscle, coord, bands):
    gmin = max(b[0].min() for b in bands)
    gmax = min(b[0].max() for b in bands)
    grid = np.linspace(gmin, gmax, N_UNION)
    mids = np.vstack([np.interp(grid, g, (blo + bhi) / 2.0) for g, blo, bhi in bands])
    ref = mids.mean(axis=0)
    mma = model_ma(m, s, muscle, coord, grid)
    rmse = float(np.sqrt(np.mean((mma - ref) ** 2)))
    half = float(np.sqrt(np.mean(np.mean((mids - ref) ** 2, axis=0))))
    pooled = float(np.sqrt(np.mean([np.mean((mma - c) ** 2) for c in mids])))
    assert abs(pooled - math.hypot(rmse, half)) < 1e-9, (
        f"decomposition (*) violated on {muscle}/{coord}: pooled {pooled} != "
        f"hypot({rmse}, {half})")
    return dict(rmse_cm=rmse, half_spread_cm=half, pooled_rmse_cm=pooled,
                n_subtracts=len(bands), angle_min=float(gmin), angle_max=float(gmax),
                n_grid=N_UNION)


def envelope_within_band(m, s, muscle, coord, bands):
    gmin = min(b[0].min() for b in bands)
    gmax = max(b[0].max() for b in bands)
    ug = np.linspace(gmin, gmax, N_UNION)
    los = np.vstack([interp_masked(ug, g, blo) for g, blo, bhi in bands])
    ups = np.vstack([interp_masked(ug, g, bhi) for g, blo, bhi in bands])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices -> NaN
        env_lo = np.nanmin(los, axis=0)
        env_up = np.nanmax(ups, axis=0)
    defined = ~np.isnan(env_lo)
    mma = model_ma(m, s, muscle, coord, ug)
    inside = (mma[defined] >= env_lo[defined]) & (mma[defined] <= env_up[defined])
    half = (env_up[defined] - env_lo[defined]) / 2.0
    return dict(within_band_pct=100.0 * inside.mean(),
                envelope_halfwidth_cm=float(half.mean()),
                angle_min=float(gmin), angle_max=float(gmax),
                n_grid=int(defined.sum()))

def main():
    eng = pd.read_csv(os.path.join(DIGI, "eng-itb-moment-arms-long.csv"))
    blem = pd.read_csv(os.path.join(DIGI, "blemker-gmax-moment-arms-long.csv"))

    persub_rows, env_rows, sweep_rows, tfl_notes, panel_rows = [], [], [], [], []

    for mkey, (path, role) in MODELS.items():
        m, s = load_model(path)

        for tract in ENG_TRACTS:
            mus = muscle_name(mkey, tract)
            for dof in ENG_DOFS:
                coord = COORD_OSIM[dof]
                sub = eng[(eng.model_tract == tract) & (eng.angle_type == dof)]
                subtracts = sorted(sub.muscle.unique())
                bands = []
                for name in subtracts:
                    g, blo, bhi = subtract_band(sub, name)
                    bands.append((g, blo, bhi))
                    mid = (blo + bhi) / 2.0
                    mma = model_ma(m, s, mus, coord, g)
                    resid = mma - mid
                    within = 100.0 * np.mean((mma >= blo) & (mma <= bhi))
                    persub_rows.append(dict(
                        model=mkey, dataset="Eng2015", model_tract=tract, dof=dof,
                        subtract=name, rmse_cm=float(np.sqrt(np.mean(resid**2))),
                        mae_cm=float(np.mean(np.abs(resid))),
                        within_band_pct_subtract=float(within),
                        band_halfwidth_cm=float(np.mean((bhi - blo) / 2.0)),
                        n=len(g), angle_min=float(g.min()), angle_max=float(g.max())))
                env = envelope_within_band(m, s, mus, coord, bands)
                env_rows.append(dict(model=mkey, dataset="Eng2015", model_tract=tract,
                                     dof=dof, n_subtracts=len(subtracts), **env))
                panel_rows.append(dict(model=mkey, dataset="Eng2015", model_tract=tract,
                                       dof=dof, within_band_pct=env["within_band_pct"],
                                       **panel_rmse(m, s, mus, coord, bands)))
                # saved sweep over the union ROM
                grid = np.linspace(env["angle_min"], env["angle_max"], N_SWEEP)
                ma = model_ma(m, s, mus, coord, grid)
                for i in range(N_SWEEP):
                    sweep_rows.append(dict(model=mkey, dataset="Eng2015",
                                           model_tract=tract, dof=dof,
                                           angle_deg=round(float(grid[i]), 6),
                                           moment_arm_cm=round(float(ma[i]), 6), grid_idx=i))

        # ---- Blemker
        for dof in BLEMKER_DOFS:
            coord = COORD_OSIM[dof]
            bsub = blem[blem.angle_type == dof]
            g = bsub[bsub.bound_type == "upper"].sort_values("angle_deg").angle_deg.values
            _a = bsub[bsub.bound_type == "lower"].sort_values("angle_deg").moment_arm_cm.values
            _b = bsub[bsub.bound_type == "upper"].sort_values("angle_deg").moment_arm_cm.values
            lo, up = np.minimum(_a, _b), np.maximum(_a, _b)  # normalize (see subtract_band)
            mid = (lo + up) / 2.0
            for tract in BLEMKER_TRACTS:
                mus = muscle_name(mkey, tract)
                mma = model_ma(m, s, mus, coord, g)
                resid = mma - mid
                within = 100.0 * np.mean((mma >= lo) & (mma <= up))
                persub_rows.append(dict(
                    model=mkey, dataset="Blemker2005", model_tract=tract, dof=dof,
                    subtract="blemker_gmax_band",
                    rmse_cm=float(np.sqrt(np.mean(resid**2))),
                    mae_cm=float(np.mean(np.abs(resid))),
                    within_band_pct_subtract=float(within),
                    band_halfwidth_cm=float(np.mean((up - lo) / 2.0)),
                    n=len(g), angle_min=float(g.min()), angle_max=float(g.max())))
                env_rows.append(dict(model=mkey, dataset="Blemker2005", model_tract=tract,
                                     dof=dof, n_subtracts=1, within_band_pct=float(within),
                                     envelope_halfwidth_cm=float(np.mean((up - lo) / 2.0)),
                                     angle_min=float(g.min()), angle_max=float(g.max()),
                                     n_grid=len(g)))
                # one band -> the midpoint IS that band's mean, so this equals the
                # per-sub-tract RMSE and the half-spread is 0 by construction.
                panel_rows.append(dict(model=mkey, dataset="Blemker2005", model_tract=tract,
                                       dof=dof, within_band_pct=float(within),
                                       **panel_rmse(m, s, mus, coord, [(g, lo, up)])))
                gg = np.linspace(g.min(), g.max(), N_SWEEP)
                ma = model_ma(m, s, mus, coord, gg)
                for i in range(N_SWEEP):
                    sweep_rows.append(dict(model=mkey, dataset="Blemker2005",
                                           model_tract=tract, dof=dof,
                                           angle_deg=round(float(gg[i]), 6),
                                           moment_arm_cm=round(float(ma[i]), 6), grid_idx=i))
            # qualitative TFL-vs-gmax-band check
            tfl = muscle_name(mkey, "tfl12_ITB")
            tma = model_ma(m, s, tfl, coord, g)
            frac_above = 100.0 * np.mean(tma > up)
            tfl_notes.append(dict(model=mkey, dof=dof, tfl_pct_above_gmax_upper=round(frac_above, 1),
                                  tfl_ma_mean_cm=round(float(tma.mean()), 3)))

    persub = pd.DataFrame(persub_rows)
    envdf = pd.DataFrame(env_rows)
    sweeps = pd.DataFrame(sweep_rows)
    panels = pd.DataFrame(panel_rows)

    # ---- aggregate
    def _agg(mkey, ds, scope, grp, pn, ps):
        return dict(model=mkey, dataset=ds, scope=scope, n_panels=len(grp),
                    mean_within_band_pct=round(grp.within_band_pct.mean(), 1),
                    min_within_band_pct=round(grp.within_band_pct.min(), 1),
                    mean_rmse_cm=round(pn.rmse_cm.mean(), 3),
                    max_rmse_cm=round(pn.rmse_cm.max(), 3),
                    mean_half_spread_cm=round(pn.half_spread_cm.mean(), 3),
                    max_half_spread_cm=round(pn.half_spread_cm.max(), 3),
                    mean_pooled_rmse_cm=round(pn.pooled_rmse_cm.mean(), 3),
                    mean_mae_persubtract_cm=round(ps.mae_cm.mean(), 3),
                    mean_rmse_persubtract_RETIRED_cm=round(ps.rmse_cm.mean(), 3))

    agg_rows = []
    for (mkey, ds), grp in envdf.groupby(["model", "dataset"]):
        sel = (panels.model == mkey) & (panels.dataset == ds)
        psel = (persub.model == mkey) & (persub.dataset == ds)
        agg_rows.append(_agg(mkey, ds, "all", grp, panels[sel], persub[psel]))
    for (mkey, ds, dof), grp in envdf.groupby(["model", "dataset", "dof"]):
        sel = (panels.model == mkey) & (panels.dataset == ds) & (panels.dof == dof)
        psel = (persub.model == mkey) & (persub.dataset == ds) & (persub.dof == dof)
        agg_rows.append(_agg(mkey, ds, dof, grp, panels[sel], persub[psel]))
    summary = pd.DataFrame(agg_rows).sort_values(["dataset", "model", "scope"])

    # ---- write 
    panels.to_csv(os.path.join(HERE, "moment_arm_validation_panel.csv"), index=False)
    persub.to_csv(os.path.join(HERE, "moment_arm_validation_persubtract.csv"), index=False)
    envdf.to_csv(os.path.join(HERE, "moment_arm_validation_envelope.csv"), index=False)
    summary.to_csv(os.path.join(HERE, "moment_arm_validation_summary.csv"), index=False)
    sweeps.to_csv(os.path.join(HERE, "model_moment_arm_sweeps_long.csv"), index=False)

    # ---- console
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
    print("\n=== SUMMARY (per model x dataset, and per DOF) ===")
    print(summary.to_string(index=False))
    print("\n=== PANEL RMSE (cm) -- THE REPORTABLE TABLE, midpoint convention ===")
    print("    rmse_cm = model vs the midpoint of the cadaveric sub-tract mean curves")
    print("    half_spread_cm = how far apart ENG'S OWN sub-tracts are (irreducible)")
    print("    pooled_rmse_cm = hypot(rmse, half_spread) = the old convention's target\n")
    print(panels[["model", "dataset", "model_tract", "dof", "n_subtracts",
                  "rmse_cm", "half_spread_cm", "pooled_rmse_cm",
                  "within_band_pct"]].round(3).to_string(index=False))
    print("\n=== ENVELOPE within-band %% per panel ===")
    print(envdf[["model", "dataset", "model_tract", "dof", "n_subtracts",
                 "within_band_pct", "envelope_halfwidth_cm"]].round(2).to_string(index=False))
    print("\n=== PER-SUB-TRACT RMSE / MAE (cm) -- DIAGNOSTIC ONLY, do not quote ===")
    print(persub[["model", "dataset", "model_tract", "dof", "subtract",
                  "rmse_cm", "mae_cm", "within_band_pct_subtract"]].round(3).to_string(index=False))
    print("\n=== TFL vs Blemker gmax band (qualitative; expect reads-above) ===")
    print(pd.DataFrame(tfl_notes).to_string(index=False))
    print("\nwrote 5 CSVs to", HERE)


if __name__ == "__main__":
    main()
