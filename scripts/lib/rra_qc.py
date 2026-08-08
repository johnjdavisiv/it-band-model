"""
Check residuals and reserves and tracking fidelity for RRA, against Hicks 2015 guidelines. 

"""
from __future__ import annotations

import csv
import json
import os

import numpy as np

from .paper1io import GRF_MOT, RESULTS
from .rra_tools import HICKS_FORCE_FRAC, HICKS_MOMENT_FRAC
from .solve_model import RESID_ROT_OPTF, RESID_TRANS_OPTF, RESID_CBOUND

FORCES, MOMENTS = ["FX", "FY", "FZ"], ["MX", "MY", "MZ"]
AXES = FORCES + MOMENTS
NGRID = 101

RAD = 180.0 / np.pi
PELVIS_TRANS = ["pelvis_tx", "pelvis_ty", "pelvis_tz"]
JOINT_ANGLES = ["pelvis_tilt", "pelvis_list", "pelvis_rotation",
                "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
                "knee_angle_r", "ankle_angle_r",
                "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
                "knee_angle_l", "ankle_angle_l",
                "lumbar_extension", "lumbar_bending", "lumbar_rotation"]
ITB_RELEVANT = ["hip_flexion_r", "hip_adduction_r", "hip_rotation_r", "knee_angle_r",
                "hip_flexion_l", "hip_adduction_l", "hip_rotation_l", "knee_angle_l"]

JOINT_RMS_LIMIT = 2.0          # deg
JOINT_MAX_LIMIT = 5.0          # deg
PELVIS_TRANS_MAX_FLAG = 40.0   # mm


# Residuals
def peak_vertical_grf():
    import opensim as osim
    t = osim.TimeSeriesTable(GRF_MOT)
    labs = set(t.getColumnLabels())
    tot = None
    for c in ("R_ground_force_vy", "L_ground_force_vy"):
        if c not in labs:
            raise SystemExit(f"{c} missing from the GRF -- cannot derive the Hicks limits")
        v = np.array(t.getDependentColumn(c).to_numpy())
        tot = v if tot is None else tot + v
    return float(np.max(np.abs(tot)))


def com_height(model_path):
    import opensim as osim
    m = osim.Model(model_path)
    s = m.initSystem()
    return float(m.calcMassCenterPosition(s)[1])


def _read_cycle_residuals(results_dir, crop):
    import opensim as osim
    hits = [f for f in os.listdir(results_dir) if f.endswith("_Actuation_force.sto")]
    if not hits:
        return None
    t = osim.TimeSeriesTable(os.path.join(results_dir, hits[0]))
    tv = np.array(t.getIndependentColumn())
    labs = set(t.getColumnLabels())
    m = (tv >= crop[0]) & (tv <= crop[1])
    if m.sum() < 5:
        return None
    frac = (tv[m] - crop[0]) / (crop[1] - crop[0]) * 100.0
    grid = np.linspace(0, 100, NGRID)
    return {a: np.interp(grid, frac, np.array(t.getDependentColumn(a).to_numpy())[m])
            for a in AXES if a in labs}


def residual_qc(root, plot=True):
    meta = json.load(open(os.path.join(root, "rra_summary.json")))
    shared_raj = os.path.join(root, "shared_massadj_raj.osim")
    gp = peak_vertical_grf()
    h = com_height(shared_raj)
    lim_f = HICKS_FORCE_FRAC * gp
    lim_m = HICKS_MOMENT_FRAC * h * gp
    print(f"\n[residual QC] peak vertical GRF {gp:.1f} N; COM height {h:.4f} m")
    print(f"[residual QC] Hicks reference: force {lim_f:.1f} N, moment {lim_m:.1f} N.m")

    rows, curves = [], {}
    for cid, info in sorted(meta["cycles_detail"].items()):
        w = info["window"]
        cur = _read_cycle_residuals(info["results_dir"], (w["crop_start"], w["crop_end"]))
        if cur is None:
            raise SystemExit(f"{cid}: no Actuation_force.sto in {info['results_dir']}")
        curves[cid] = cur
        for a, v in cur.items():
            lim = lim_f if a in FORCES else lim_m
            pk = float(np.max(np.abs(v)))
            rows.append(dict(
                cycle=cid, axis=a, kind="force" if a in FORCES else "moment",
                peak=pk, rms=float(np.sqrt(np.mean(v ** 2))), mean=float(np.mean(v)),
                pct_cycle_over_hicks=100.0 * float(np.mean(np.abs(v) > lim)),
                hicks_limit=lim, hicks_ratio=pk / lim, passes_hicks=bool(pk <= lim)))

    out_csv = os.path.join(RESULTS, "rra_residuals_qc.csv")
    os.makedirs(RESULTS, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{'axis':<5} {'peak':>9} {'RMS':>9} {'x Hicks':>9}  cycles inside")
    for a in AXES:
        d = [r for r in rows if r["axis"] == a]
        n_in = sum(1 for r in d if r["passes_hicks"])
        print(f"{a:<5} {max(r['peak'] for r in d):9.1f} {max(r['rms'] for r in d):9.1f} "
              f"{max(r['hicks_ratio'] for r in d):8.2f}x  {n_in}/{len(d)}")
    print("(Hicks is a reference for running, not a gate; see module docstring)")

    ceil_f = RESID_TRANS_OPTF * RESID_CBOUND
    ceil_m = RESID_ROT_OPTF * RESID_CBOUND
    over = [r for r in rows
            if (r["kind"] == "force" and r["peak"] > ceil_f)
            or (r["kind"] == "moment" and r["peak"] > ceil_m)]
    print(f"[residual QC] Moco representability ceiling {ceil_f:.0f} N / {ceil_m:.0f} N.m: "
          f"{'ALL CYCLES INSIDE' if not over else 'EXCEEDED -- the solve will be infeasible'}")
    for r in sorted(over, key=lambda r: -r["peak"])[:10]:
        print(f"   {r['cycle']:<10} {r['axis']} {r['peak']:.1f}")

    if plot:
        _plot_residuals(curves, lim_f, lim_m, os.path.join(root, "rra_residuals.png"))
    print(f"-> {out_csv}")
    return not over


def _plot_residuals(curves, lim_f, lim_m, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True)
    grid = np.linspace(0, 100, NGRID)
    for j, a in enumerate(AXES):
        ax = axs[j // 3][j % 3]
        band = lim_f if a in FORCES else lim_m
        ax.axhspan(-band, band, color="0.85", zorder=0)
        for cid, cur in curves.items():
            if a in cur:
                ax.plot(grid, cur[a], lw=1.0, alpha=0.8, label=cid)
        ax.axhline(0, color="k", lw=0.6, zorder=1)
        ax.set_title(a + ("  (N)" if a in FORCES else "  (N$\\cdot$m)"), fontsize=11)
        ax.grid(alpha=0.25)
        if j // 3 == 1:
            ax.set_xlabel("% of gait cycle")
    axs[0][0].legend(fontsize=7, frameon=False)
    fig.suptitle("RRA pelvis residuals, in-crop (grey band = Hicks reference)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"-> {out_png}")


# IK tracking
def _read_perr(results_dir):
    import opensim as osim
    hits = [f for f in os.listdir(results_dir) if f.endswith("_pErr.sto")]
    if not hits:
        return None, None
    t = osim.TimeSeriesTable(os.path.join(results_dir, hits[0]))
    tv = np.array(t.getIndependentColumn())
    cols = {c: np.array(t.getDependentColumn(c).to_numpy()) for c in t.getColumnLabels()}
    return tv, cols


def _argmax_key(out, keys, suffix):
    #Which coordinate isworst? 
    cand = [k for k in keys if f"{k}_{suffix}" in out]
    return max(cand, key=lambda k: out[f"{k}_{suffix}"]) if cand else ""


def _analyze_tracking_cycle(cid, info):
    tv, cols = _read_perr(info["results_dir"])
    if tv is None:
        return None
    # note - the pad is cropped away
    w = info["window"]
    win = (tv >= w["crop_start"]) & (tv <= w["crop_end"])
    if win.sum() < 3:
        win = np.ones(len(tv), bool)
    out = {"cycle": cid}
    for coord in PELVIS_TRANS:
        if coord in cols:
            e = np.abs(cols[coord][win]) * 1000.0
            out[f"{coord}_rms_mm"] = float(np.sqrt(np.mean(e ** 2)))
            out[f"{coord}_max_mm"] = float(np.max(e))
    for coord in JOINT_ANGLES:
        if coord in cols:
            e = np.abs(cols[coord][win]) * RAD
            out[f"{coord}_rms_deg"] = float(np.sqrt(np.mean(e ** 2)))
            out[f"{coord}_max_deg"] = float(np.max(e))
    ja_rms = [out[f"{k}_rms_deg"] for k in JOINT_ANGLES if f"{k}_rms_deg" in out]
    ja_max = [out[f"{k}_max_deg"] for k in JOINT_ANGLES if f"{k}_max_deg" in out]
    pt_max = [out[f"{k}_max_mm"] for k in PELVIS_TRANS if f"{k}_max_mm" in out]
    out["worst_joint_rms_deg"] = max(ja_rms) if ja_rms else np.nan
    out["worst_joint_max_deg"] = max(ja_max) if ja_max else np.nan
    out["worst_joint_rms"] = _argmax_key(out, JOINT_ANGLES, "rms_deg")
    out["worst_joint_max"] = _argmax_key(out, JOINT_ANGLES, "max_deg")
    out["worst_pelvis_trans_max_mm"] = max(pt_max) if pt_max else np.nan
    out["joint_track_fail"] = bool(out["worst_joint_rms_deg"] > JOINT_RMS_LIMIT
                                   or out["worst_joint_max_deg"] > JOINT_MAX_LIMIT)
    itb_rms = [out[f"{k}_rms_deg"] for k in ITB_RELEVANT if f"{k}_rms_deg" in out]
    itb_max = [out[f"{k}_max_deg"] for k in ITB_RELEVANT if f"{k}_max_deg" in out]
    out["worst_itb_joint_rms_deg"] = max(itb_rms) if itb_rms else np.nan
    out["worst_itb_joint_max_deg"] = max(itb_max) if itb_max else np.nan
    out["worst_itb_joint_rms"] = _argmax_key(out, ITB_RELEVANT, "rms_deg")
    out["worst_itb_joint_max"] = _argmax_key(out, ITB_RELEVANT, "max_deg")
    return out


def tracking_qc(root):
    #Sweep IK and find worst angles
    meta = json.load(open(os.path.join(root, "rra_summary.json")))
    rows = [r for r in (_analyze_tracking_cycle(cid, info)
                        for cid, info in sorted(meta["cycles_detail"].items())) if r]
    if not rows:
        raise SystemExit("no RRA pErr files found")

    keys = ["cycle",
            "worst_joint_rms", "worst_joint_rms_deg",
            "worst_joint_max", "worst_joint_max_deg",
            "worst_itb_joint_rms", "worst_itb_joint_rms_deg",
            "worst_itb_joint_max", "worst_itb_joint_max_deg",
            "worst_pelvis_trans_max_mm", "joint_track_fail"]
    extra = sorted({k for r in rows for k in r} - set(keys))
    out_csv = os.path.join(RESULTS, "rra_tracking_qc.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys + extra)
        w.writeheader()
        w.writerows(rows)

    print(f"\n[tracking QC] {'cycle':<10}{'worst by RMS':>18}{'RMS deg':>9}"
          f"{'worst by peak':>18}{'max deg':>9}{'pelvis drift':>14}")
    for r in rows:
        mark = "  <-- over screen" if r["joint_track_fail"] else ""
        print(f"              {r['cycle']:<10}{r['worst_joint_rms']:>18}"
              f"{r['worst_joint_rms_deg']:>9.2f}{r['worst_joint_max']:>18}"
              f"{r['worst_joint_max_deg']:>9.2f}{r['worst_pelvis_trans_max_mm']:>12.1f}mm{mark}")
    print(f"[tracking QC] ITB-relevant coordinates only (hip + knee -- the only joints any "
          f"tract crosses):")
    for r in rows:
        print(f"              {r['cycle']:<10}{r['worst_itb_joint_rms']:>18}"
              f"{r['worst_itb_joint_rms_deg']:>9.2f}{r['worst_itb_joint_max']:>18}"
              f"{r['worst_itb_joint_max_deg']:>9.2f}")
    fails = [r for r in rows if r["joint_track_fail"]]
    print(f"[tracking QC] joint-angle screen (RMS>{JOINT_RMS_LIMIT} or max>{JOINT_MAX_LIMIT} "
          f"deg): {len(fails)}/{len(rows)} cycles over")
    print(f"-> {out_csv}")
    return fails
