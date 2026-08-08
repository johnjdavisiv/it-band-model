""" 
# RRA on IK data

We do two passes, and average torso COM shift across ALL cycles at that speed (essential for fair comparison)

Then we do final phase 2 wiht adjust_com = false to get final kinematics and residuals

Note we resample and covnert to radians after this

FYI - Takes 10-15min to run

"""
import os
import sys
import json
import shutil
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.paper1io import (MODELS, INPUTS, RESULTS, WORK, CYCLES, crop_window,
                          resolve_external_loads)                              # noqa: E402
from lib import rra_tools as RT                                                # noqa: E402
from lib.rra_qc import residual_qc, tracking_qc                                # noqa: E402
from lib.solve_model import build_model, generic_dgf_model, strip_external_loads  # noqa: E402

SCALED = {"raj": os.path.join(WORK, "scale", "Rajagopal2023_subject01_scaled.osim"),
          "davis": os.path.join(WORK, "scale", "Davis2026_subject01_scaled.osim")}
IK_MOT = os.path.join(WORK, "ik", "Run_30002_ik.mot")
SOLVE_MODEL_NAME = {"raj": "Rajagopal2023_subject01.osim", "davis": "Davis2026_subject01.osim"}
DGF_NAME = {"raj": "Rajagopal2023_subject01_DGF.osim", "davis": "Davis2026_subject01_DGF.osim"}

ITB_TRACTS = ["glmax12_ITB_r", "glmax34_ITB_r", "tfl12_ITB_r"]
ENG_ITB_TENDON_STRAIN = {"glmax12_ITB_r": 0.087, "glmax34_ITB_r": 0.110, "tfl12_ITB_r": 0.053}

# Assert models are ~identical. Note Opensim does some funny rounding stuff so it's not exactly equal. 
MASS_TOL_KG = 1e-9
INERTIA_REL_TOL = 1e-4
COM_TOL_M = 1e-5


def assert_inertial_identity(osim, raj_path, davis_path):
    mr = osim.Model(raj_path); mr.initSystem()
    md = osim.Model(davis_path); md.initSystem()
    bad = []
    for b in mr.getBodySet():
        bd = md.getBodySet().get(b.getName())
        dm = abs(b.getMass() - bd.getMass())
        Ir, Id = b.get_inertia(), bd.get_inertia()
        scale = max(abs(Ir.get(i)) for i in range(6)) or 1.0
        dI = max(abs(Ir.get(i) - Id.get(i)) for i in range(6)) / scale
        cr, cd = b.get_mass_center(), bd.get_mass_center()
        dc = max(abs(cr.get(i) - cd.get(i)) for i in range(3))
        if dm > MASS_TOL_KG or dI > INERTIA_REL_TOL or dc > COM_TOL_M:
            bad.append(f"{b.getName()}: dmass={dm:.3e} kg  dI/I={dI:.3e}  dCOM={dc*1e6:.3f} um")
    if bad:
        raise SystemExit("raj/davis inertial mismatch -- the graft must not change body "
                         "properties:\n  " + "\n  ".join(bad))


def rra_all_cycles(osim, cycles, out_root):
    #First round
    os.makedirs(out_root, exist_ok=True)
    tasks = RT.ensure_tasks()
    extloads = resolve_external_loads(out_root)
    base = SCALED["raj"]
    c0_torso = RT.torso_com(base)

    acts, com = RT.residual_actuators(base)
    RT.verify_residual_actuators(base, acts)
    print(f"residual actuators: force optf {RT.RESID_FORCE_OPTF:g}, "
          f"moment optf {RT.RESID_MOMENT_OPTF:g}; pelvis COM {com}")

    # ---------------------------------------------------------------- phase 1
    print(f"\n=== phase 1 (adjust_com=true) on {os.path.basename(base)} ===", flush=True)
    dcoms, mtables = [], []
    for c in cycles:
        t0, t1 = RT.window(c)
        _, log, adjusted = RT.run_pass(f"p1_cycle{c}", base, t0, t1, extloads, IK_MOT,
                                       adjust_com=True, tasks=tasks, actuators=acts,
                                       root=out_root)
        dcoms.append(RT.torso_com(adjusted) - c0_torso)
        tab = RT.mass_table(log)
        if not tab:
            raise SystemExit(f"cycle{c}: phase 1 printed no mass table (see {log})")
        mtables.append(tab)
        print(f"  cycle{c} [{t0}, {t1}]  dCOM {dcoms[-1] * 100} cm  "
              f"total mass change {RT.mass_recommendation(log):+.4f} kg", flush=True)

    dmean = np.mean(dcoms, axis=0)
    if len(dcoms) > 1:
        d = np.array(dcoms)
        signs = [bool(np.all(np.sign(d[:, i]) == np.sign(dmean[i]))) for i in range(3)]
        print(f"\n  mean torso-COM nudge {dmean * 100} cm   SD {d.std(axis=0) * 100} cm")
        print(f"  same-sign across cycles per axis: {signs}   "
              f"(a mean over opposing shifts would be a cancellation, not a correction)")


    print("\n=== shared model (mean torso COM + mean per-body mass) ===")
    shared = {}
    for track, src in SCALED.items():
        out = os.path.join(out_root, f"shared_massadj_{track}.osim")
        before, after = RT.apply_mean_masses_and_com(src, mtables, dmean, out)
        shared[track] = out
        print(f"  {track:<6} mass {before:.4f} -> {after:.4f} kg ({after - before:+.4f})  "
              f"-> {os.path.basename(out)}")
    assert_inertial_identity(osim, shared["raj"], shared["davis"])
    print("  raj/davis inertial identity OK")

    # Torso vs pelvis COM!
    acts2, com2 = RT.residual_actuators(shared["raj"])
    RT.verify_residual_actuators(shared["raj"], acts2)

    # RRA round 2
    print("\n=== phase 2 (adjust_com=false) on shared_massadj_raj.osim ===", flush=True)
    summary = {}
    for c in cycles:
        t0, t1 = RT.window(c)
        rdir, _, _ = RT.run_pass(f"p2_cycle{c}", shared["raj"], t0, t1, extloads, IK_MOT,
                                 adjust_com=False, tasks=tasks, actuators=acts2, root=out_root)
        cr0, cr1 = crop_window(c)
        summary[f"cycle{c}"] = dict(results_dir=rdir,
                                    window=dict(sim_start=t0, sim_end=t1,
                                                crop_start=cr0, crop_end=cr1))
        print(f"  cycle{c} [{t0}, {t1}] crop [{cr0}, {cr1}]", flush=True)

    # Kinematics to prescribe to moco
    print("\n=== prescribed kinematics (uniform 250 Hz, radians) ===")
    kin_dir = os.path.join(out_root, "inputs")
    os.makedirs(kin_dir, exist_ok=True)
    conv = osim.Model(shared["raj"]) # the deg->rad map is muscle-independent
    conv.initSystem()
    for c in cycles:
        p = RT.build_cycle_kinematics(c, summary[f"cycle{c}"]["results_dir"], conv, kin_dir)
        RT.verify_against_crop(c, p)
        summary[f"cycle{c}"]["prescribed"] = p

    meta = dict(optf_force=RT.RESID_FORCE_OPTF, optf_moment=RT.RESID_MOMENT_OPTF,
                pad_s=RT.PAD_S, cycles=list(cycles), torso_com_nudge_cm=list(dmean * 100),
                cycles_detail=summary)
    with open(os.path.join(out_root, "rra_summary.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return summary, shared


def promote(osim, cycles, out_root, shared):
    print("\n=== promoting into inputs/ and models/ ===")
    for c in cycles:
        src = os.path.join(out_root, "inputs", f"cycle{c}_q_uniform250.sto")
        dst = os.path.join(INPUTS, f"cycle{c}_q_uniform250.sto")
        shutil.copyfile(src, dst)
        print(f"  {os.path.relpath(dst)}")
    for track, name in SOLVE_MODEL_NAME.items():
        shutil.copyfile(shared[track], os.path.join(MODELS, name))
        print(f"  {os.path.relpath(os.path.join(MODELS, name))}")

    print("\n=== freezing the DeGroote-Fregly read-out models ===")
    m = generic_dgf_model(os.path.join(MODELS, "Davis2026.osim"))
    m.printToXML(os.path.join(MODELS, "Davis2026_DGF.osim"))
    print(f"  Davis2026_DGF.osim: {m.getMuscles().getSize()} muscles")
    for track, name in DGF_NAME.items():
        extloads = resolve_external_loads(os.path.join(RESULTS, track))
        model = build_model(track, extloads) # the exact model the solver optimizes
        out_path = os.path.join(MODELS, name)
        model.printToXML(out_path)
        strip_external_loads(out_path) # never ship an absolute GRF path in a model
        osim.Model(out_path).initSystem() # must still load without the loads
        line = f"  {name}: {model.getMuscles().getSize()} muscles"
        if track == "davis": # the compliant ITB tendons must survive the DGF conversion
            muscles = model.getMuscles()
            bad = [nm for nm in ITB_TRACTS
                   if abs(osim.DeGrooteFregly2016Muscle.safeDownCast(muscles.get(nm))
                          .get_tendon_strain_at_one_norm_force()
                          - ENG_ITB_TENDON_STRAIN[nm]) >= 1e-3]
            line += "  (ITB tendon strains OK)" if not bad else f"  !! ITB compliance lost: {bad}"
        print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cycles", default="all", help="'all' or e.g. 1,3,5")
    ap.add_argument("--no-promote", action="store_true",
                    help="run RRA + QC but leave the frozen inputs/ and models/ untouched")
    args = ap.parse_args()
    cycles = list(CYCLES) if args.cycles == "all" else [int(x) for x in args.cycles.split(",")]

    for p in list(SCALED.values()) + [IK_MOT]:
        assert os.path.exists(p), f"missing {p} -- run 01_scale.py and 02_ik.py first"

    import opensim as osim
    osim.Logger.setLevelString("Error")
    out_root = os.path.join(RT.RRA_WORK, "build")
    summary, shared = rra_all_cycles(osim, cycles, out_root)

    representable = residual_qc(out_root)
    breaches = tracking_qc(out_root)
    if breaches:
        print("\nNOTE: a marginal tracking breach on a TRUNK coordinate (e.g. pelvis_tilt ~2.0 "
              "deg RMS) is the accepted cost of pricing the residual moments; the "
              "ITB-relevant block above is what determines whether a breach matters.")
    if not representable:
        raise SystemExit("\nresiduals exceed what MocoInverse can represent -- NOT promoting. "
                         "Fix the inputs before solving; the solve will fail!")
    if args.no_promote:
        print("\n--no-promote: products left in", out_root)
        return
    promote(osim, cycles, out_root, shared)
    print("\nDONE. Next: python -u scripts/04_moco_inverse.py --track all --cycle all")


if __name__ == "__main__":
    main()
