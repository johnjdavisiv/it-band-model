"""Passive joint-moment agreement with Silder (2007): RMSE per model x joint x condition,
plus a machine-derived record of the exact protocol each number came from.

`passive_joint_moment_validation.py` sweeps, plots and dumps the raw moment traces but
never differences a model against Silder -- the only RMS in it compares two *reference*
sources (digitized .mat vs the analytical mean-parameter reconstruction) to justify using
the .mat. This script supplies the missing model-vs-experiment numbers.

Method. Models are sampled on a 41-point uniform angle grid; the digitized Silder curves
carry 96 (hip) / 121 (knee) points on their own grid, so Silder is INTERPOLATED onto the
model angles. This is pure interpolation with no extrapolation -- Silder's support is
WIDER than the swept range in both joints (hip -15..80 vs swept -15..37; knee 0..120 vs
swept 0..75) -- and every segment is strictly monotonic in angle with unique abscissae,
so np.interp is safe without sorting or de-duplication. Both are asserted below.

⚠️ The Silder reference is a digitized MEAN-of-subjects curve with no dispersion in the
file, so these are model-vs-mean distances with no experimental band to contextualise
them. Report them as such; do not imply a tolerance the data cannot support.

Outputs (beside this script):
  passive_moment_rmse.csv       RMSE / MAE / max error per model x joint x condition
  passive_moment_protocol.csv   the EXACT pose behind every condition: which coordinate
                                is swept over what range, and the value of every other
                                coordinate. Derived by calling the validation module's
                                own make_base_pose(), so it cannot drift from the code.

Run: conda run -n osim46 python compute_silder_rmse.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import passive_joint_moment_validation as V   # noqa: E402  (protocol source of truth)
import model_eval as me                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TOTALS = os.path.join(HERE, "passive_joint_moments_total.csv")

# (joint, swept coordinate, angle grid, conditions) -- read from the validation module
SWEEPS = [
    ("hip", "hip_flexion_r", V.HIP_ANGLES, V.HIP_CONDITIONS),
    ("knee", "knee_angle_r", V.KNEE_ANGLES, V.KNEE_CONDITIONS),
]


def rmse_table(df):
    rows = []
    for joint, sweep_coord, angles, conds in SWEEPS:
        for cond in conds:
            sub = df[(df.joint == joint) & (df.condition == cond["label"])]
            ref = sub[sub.model == V.SILDER_LABEL].sort_values("angle_deg")
            if ref.empty:
                print(f"  (no Silder segment for {joint} / {cond['label']}) -- skipped")
                continue
            rx = ref.angle_deg.to_numpy()
            ry = ref.total_moment_Nm.to_numpy()
            # guards for np.interp: strictly increasing, unique, and spanning the sweep
            assert np.all(np.diff(rx) > 0), f"{joint}/{cond['label']}: Silder x not increasing"
            assert rx[0] <= angles.min() + 1e-9 and rx[-1] >= angles.max() - 1e-9, (
                f"{joint}/{cond['label']}: sweep {angles.min():.1f}..{angles.max():.1f} "
                f"escapes Silder support {rx[0]:.1f}..{rx[-1]:.1f} -> would extrapolate")
            for _, label, _ in V.MODELS:
                m = sub[sub.model == label].sort_values("angle_deg")
                if m.empty:
                    continue
                mx = m.angle_deg.to_numpy()
                my = m.total_moment_Nm.to_numpy()
                si = np.interp(mx, rx, ry)
                err = my - si
                rows.append(dict(
                    joint=joint, swept_coordinate=sweep_coord, condition=cond["label"],
                    model=label, n_points=len(mx),
                    sweep_deg=f"{mx.min():.1f} to {mx.max():.1f}",
                    silder_support_deg=f"{rx[0]:.1f} to {rx[-1]:.1f}",
                    rmse_Nm=round(float(np.sqrt(np.mean(err ** 2))), 3),
                    mae_Nm=round(float(np.mean(np.abs(err))), 3),
                    max_abs_err_Nm=round(float(np.max(np.abs(err))), 3),
                    bias_Nm=round(float(np.mean(err)), 3),
                    silder_range_Nm=f"{ry.min():.1f} to {ry.max():.1f}",
                ))
    return pd.DataFrame(rows)


def protocol_table():
    """The exact pose behind each condition, derived by calling make_base_pose().

    Answers 'what other joints are locked where' without transcription: every coordinate
    is listed with its value and its role (swept / fixed by the condition / left at the
    model default / neutralised). Uses Davis2026; the pose construction is
    model-independent but default values are read from a real model, so the file records
    which one.
    """
    model, _ = me.load_model(os.path.join(V.PKG, "models", "Davis2026.osim"))
    defaults = me.default_pose(model)
    rows = []
    for joint, sweep_coord, angles, conds in SWEEPS:
        for cond in conds:
            pose = V.make_base_pose(model, cond["fixed"])
            for coord in sorted(pose):
                if coord == sweep_coord:
                    role, val = "SWEPT", f"{angles.min():.1f} to {angles.max():.1f}"
                elif coord in cond["fixed"]:
                    role, val = "fixed by condition", f"{np.degrees(pose[coord]):.1f}"
                elif coord in ("hip_adduction_r", "hip_rotation_r"):
                    role, val = "neutralised to 0", "0.0"
                else:
                    # translations are metres; everything else is an angle in radians
                    is_len = coord.startswith("pelvis_t")
                    role = "model default"
                    val = (f"{pose[coord]:.4f}" if is_len
                           else f"{np.degrees(pose[coord]):.1f}")
                    if abs(pose[coord] - defaults[coord]) > 1e-12:
                        role = "model default (MODIFIED?)"
                rows.append(dict(
                    joint=joint, condition=cond["label"], coordinate=coord, role=role,
                    value_deg_or_m=val,
                    units=("m" if coord.startswith("pelvis_t") else "deg")))
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(TOTALS)
    rmse = rmse_table(df)
    proto = protocol_table()

    out_r = os.path.join(HERE, "passive_moment_rmse.csv")
    out_p = os.path.join(HERE, "passive_moment_protocol.csv")
    rmse.to_csv(out_r, index=False)
    proto.to_csv(out_p, index=False)

    print("Passive joint moment: model vs digitized Silder (2007), N·m\n")
    for joint in ("hip", "knee"):
        sub = rmse[rmse.joint == joint]
        if sub.empty:
            continue
        swept = sub.swept_coordinate.iloc[0]
        print(f"{joint.upper()}  (sweep {swept} over {sub.sweep_deg.iloc[0]}°)")
        print(f"  {'condition':<26}{'model':<26}{'RMSE':>8}{'MAE':>8}"
              f"{'max':>8}{'bias':>8}")
        print("  " + "-" * 84)
        for cond in sub.condition.unique():
            for _, r in sub[sub.condition == cond].iterrows():
                print(f"  {cond:<26}{r.model:<26}{r.rmse_Nm:>8.2f}{r.mae_Nm:>8.2f}"
                      f"{r.max_abs_err_Nm:>8.2f}{r.bias_Nm:>+8.2f}")
            print()
    print("  bias = mean(model - Silder); + = model reads more flexor than Silder.")
    print("  Silder is a digitized mean-of-subjects curve with no dispersion in the "
          "file,\n  so these are distances to a mean, not to a tolerance band.\n")
    print(f"saved: {out_r}")
    print(f"saved: {out_p}  ({len(proto)} coordinate rows over "
          f"{proto.groupby(['joint', 'condition']).ngroups} conditions)")


if __name__ == "__main__":
    main()
