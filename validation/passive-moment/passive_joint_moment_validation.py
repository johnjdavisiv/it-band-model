"""

Total sagittal-plane passive joint moments 
"""
import os
import sys
import math

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_eval as me

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))          # the package root
OUTDIR = HERE
SILDER_MAT = os.path.join(HERE, "DigitizedPassiveMoments_Silder2007.mat")

MODELS = [
    (os.path.join(PKG, "models", "Davis2026.osim"), "Davis2026 (ITB model)", "#C0392B"),
    (os.path.join(PKG, "models", "RajagopalLaiUhlrich2023.osim"), "RajagopalLaiUhlrich2023", "#2166AC"),
    (os.path.join(PKG, "models", "Rajagopal2016.osim"), "Rajagopal2016", "#92C5DE"),
]
SILDER_LABEL = "Silder 2007 (digitized)"

N = 41
HIP_ANGLES = np.linspace(-15.0, 37.0, N)
KNEE_ANGLES = np.linspace(0.0, 75.0, N)

HIP_CONDITIONS = [
    dict(label="knee = 15°", fixed=dict(knee_angle_r=15.0, ankle_angle_r=0.0)),
    dict(label="knee = 60°", fixed=dict(knee_angle_r=60.0, ankle_angle_r=0.0)),
]
KNEE_CONDITIONS = [
    dict(label="hip = 0°,  ankle = 20°",  fixed=dict(hip_flexion_r=0.0,   ankle_angle_r=20.0)),
    dict(label="hip = 0°,  ankle = −15°", fixed=dict(hip_flexion_r=0.0,   ankle_angle_r=-15.0)),
    dict(label="hip = −15°,  ankle = 20°", fixed=dict(hip_flexion_r=-15.0, ankle_angle_r=20.0)),
]

_COORD2FIELD = {"hip_flexion_r": "hip", "knee_angle_r": "knee", "ankle_angle_r": "ankle"}

# Silder
def load_silder_digitized(matpath=SILDER_MAT):
    """Parse DigitizedPassiveMoments_Silder2007.mat into per-joint curve segments.

    File layout: PassiveM[j] (moment vector) and JAngles[j] (N x 6 pose matrix) for
    j = 0/1/2 = hip/knee/ankle; angle columns are 0=hip, 2=knee, 3=ankle (others 0).
    Each joint's arrays concatenate several fixed-condition sweeps back-to-back; a new
    sweep starts wherever the swept angle resets downward. Returns
    {'hip':[seg,...], 'knee':[...], 'ankle':[...]}, seg = dict(hip,knee,ankle, angle, moment).
    """
    m = sio.loadmat(matpath, squeeze_me=True, struct_as_record=False)
    PM, JA = m["PassiveM"], m["JAngles"]
    joints = {0: "hip", 1: "knee", 2: "ankle"}
    swept_col = {0: 0, 1: 2, 2: 3}  # which JAngles column is swept, per joint
    out = {"hip": [], "knee": [], "ankle": []}
    for ji, jname in joints.items():
        pm = np.asarray(PM[ji]).ravel().astype(float)
        ja = np.asarray(JA[ji]).astype(float)
        sw = ja[:, swept_col[ji]]
        breaks = [0] + [k for k in range(1, len(sw)) if sw[k] < sw[k - 1] - 1e-6] + [len(sw)]
        for s in range(len(breaks) - 1):
            a, b = breaks[s], breaks[s + 1]
            out[jname].append(dict(
                hip=float(ja[a, 0]), knee=float(ja[a, 2]), ankle=float(ja[a, 3]),
                angle=sw[a:b].copy(), moment=pm[a:b].copy()))
    return out


def silder_curve(silder, joint, fixed_deg, tol=0.5):
    for seg in silder[joint]:
        if all(abs(seg[_COORD2FIELD[c]] - v) <= tol for c, v in fixed_deg.items()):
            return seg["angle"], seg["moment"]
    return None, None


def make_base_pose(model, fixed_deg):
    pose = me.default_pose(model)
    for c in ("hip_adduction_r", "hip_rotation_r"):
        if c in pose:
            pose[c] = 0.0
    for name, deg in fixed_deg.items():
        pose[name] = math.radians(deg)
    return pose


def run_condition(model, state, sweep_coord, angles_deg, fixed_deg):
    base = make_base_pose(model, fixed_deg)
    df = me.sweep(model, state, sweep_coord, angles_deg,
                  base_pose=base, moment_arm_coords=[sweep_coord],
                  equilibrate=True, activation=0.0)
    return me.sum_passive_moment(df, sweep_coord), df


def collect(joint, sweep_coord, angles_deg, conditions):
    totals_rows, perm_rows = [], []
    for path, label, _ in MODELS:
        model, state = me.load_model(path)
        for cond in conditions:
            totals, perm = run_condition(model, state, sweep_coord, angles_deg, cond["fixed"])
            totals_rows.append(totals.assign(joint=joint, condition=cond["label"], model=label))
            perm_rows.append(perm.assign(joint=joint, condition=cond["label"], model=label))
        print(f"  [{label}] {joint}: {len(conditions)} condition(s) done")
    return pd.concat(totals_rows, ignore_index=True), pd.concat(perm_rows, ignore_index=True)


def silder_rows(joint, conditions, silder, window):
    rows = []
    for cond in conditions:
        ang, mom = silder_curve(silder, joint, cond["fixed"])
        if ang is None:
            print(f"  (no Silder segment for {joint} {cond['label']})")
            continue
        rows.append(pd.DataFrame(dict(joint=joint, condition=cond["label"],
                                      model=SILDER_LABEL, angle_deg=ang, total_moment_Nm=mom)))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def main():
    silder = load_silder_digitized()

    print("HIP sweeps ...")
    hip_tot, hip_perm = collect("hip", "hip_flexion_r", HIP_ANGLES, HIP_CONDITIONS)
    print("KNEE sweeps ...")
    knee_tot, knee_perm = collect("knee", "knee_angle_r", KNEE_ANGLES, KNEE_CONDITIONS)

    totals = pd.concat([hip_tot, knee_tot], ignore_index=True)
    sil = pd.concat([silder_rows("hip", HIP_CONDITIONS, silder, (-15, 37)),
                     silder_rows("knee", KNEE_CONDITIONS, silder, (0, 75))], ignore_index=True)
    totals = pd.concat([totals, sil], ignore_index=True)[
        ["joint", "condition", "model", "angle_deg", "total_moment_Nm"]]
    perm = pd.concat([hip_perm, knee_perm], ignore_index=True)

    tot_csv = os.path.join(OUTDIR, "passive_joint_moments_total.csv")
    perm_csv = os.path.join(OUTDIR, "passive_joint_moments_permuscle.csv")
    totals.to_csv(tot_csv, index=False)
    perm.to_csv(perm_csv, index=False)
    print(f"wrote {tot_csv}\nwrote {perm_csv}")
    print("(plot: make_figures.py check_F06 draws these curves)")


if __name__ == "__main__":
    main()
