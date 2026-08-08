"""

Solving MocoInverse.

Note: LONG! Takes a few hours per gait cycle. 

To run: 

Run one:  python -u scripts/04_moco_inverse.py --track davis --cycle 1
Run all:  python -u scripts/04_moco_inverse.py --track all --cycle all
"""
import os
import sys
import time
import argparse

import numpy as np
import opensim as osim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.paper1io import (RESULTS, TRACKS, CYCLES, kinematics, crop_window,
                          resolve_external_loads)                          # noqa: E402
from lib.solve_model import MESH, TRIM_FRAMES, build_model                 # noqa: E402
from lib.area_effort import area_weight_muscle_effort                      # noqa: E402


def valid_full_solution(path):
    #Makes ure we can actually write solutions...
    try:
        tab = osim.TimeSeriesTable(path)
    except Exception:
        return False
    acts = [L for L in tab.getColumnLabels() if L.endswith("/activation")]
    if not acts:
        return False
    return all(np.all(np.isfinite(tab.getDependentColumn(L).to_numpy())) for L in acts)


def write_cropped_exact(full_path, crop0, crop1, out_path):
    # This is where we crop to exact gait cycel to avoid boundary effects 
    # Note that every column is interpolated at exactly crop0 and crop1, with the interior mesh nodes between, so the artifact spans the reported stride precisely. RowVector is built from a plain list, not a a Storage row-view, which segfaults!
    tab = osim.TimeSeriesTable(full_path)
    tt = np.array(tab.getIndependentColumn())
    labels = list(tab.getColumnLabels())
    data = np.column_stack([tab.getDependentColumn(L).to_numpy() for L in labels])
    interior = tt[(tt > crop0 + 1e-9) & (tt < crop1 - 1e-9)]
    grid = np.concatenate(([crop0], interior, [crop1]))
    out = osim.TimeSeriesTable()
    out.setColumnLabels(labels)
    for ti in grid:
        row = [float(np.interp(ti, tt, data[:, j])) for j in range(len(labels))]
        out.appendRow(float(ti), osim.RowVector(row))
    out.addTableMetaDataString("inDegrees", "no")
    osim.STOFileAdapter().write(out, out_path)


def build_study(track, cycle, extloads, parallel=None, max_iterations=None):
    kin_file = kinematics(cycle)
    t = np.array(osim.TimeSeriesTable(kin_file).getIndependentColumn())
    t0, t1 = float(t[TRIM_FRAMES]), float(t[-1 - TRIM_FRAMES])
    crop0, crop1 = crop_window(cycle)
    # The trim must never eat into the reported stride!
    if not (t0 < crop0 and crop1 < t1):
        raise SystemExit(f"cycle{cycle}: TRIM_FRAMES={TRIM_FRAMES} leaves window "
                         f"[{t0:.4f},{t1:.4f}], which does not contain crop [{crop0},{crop1}]")

    model = build_model(track, extloads)
    nmusc = model.getMuscles().getSize()

    inverse = osim.MocoInverse()
    inverse.setModel(osim.ModelProcessor(model))
    kin = osim.TableProcessor(kin_file)
    kin.append(osim.TabOpUseAbsoluteStateNames())
    inverse.setKinematics(kin)
    inverse.set_initial_time(t0)
    inverse.set_final_time(t1)
    inverse.set_mesh_interval(MESH)
    inverse.set_kinematics_allow_extra_columns(True)
    inverse.set_minimize_sum_squared_activations(True) # excitation + activation effort goals - scaled by effort later

    print(f"SETUP {track} cycle{cycle}: window [{t0:.4f},{t1:.4f}] crop [{crop0},{crop1}] "
          f"mesh={MESH} muscles={nmusc}", flush=True)

    study = inverse.initialize()
    solver = osim.MocoCasADiSolver.safeDownCast(study.updSolver())
    solver.set_scale_variables_using_bounds(True)   # avoids dual-infeasibility blowups
    solver.set_optim_sparsity_detection("none") # vvv     
    # CRITICAL!! exact symbolic Jacobian sparsity: the default numerical probe can NaN out and falsely report infeasibility
    
    solver.set_optim_constraint_tolerance(1e-4) # per Miller
    solver.set_optim_convergence_tolerance(1e-3) # per Miller
    solver.set_optim_max_iterations(2000 if max_iterations is None else int(max_iterations))
    # dF/dt regularization at a deliberately negligible weight (~1e-10) - may help avoid degenerate tendon force configuratoins
    solver.set_minimize_implicit_auxiliary_derivatives(True)
    solver.set_implicit_auxiliary_derivatives_weight(1e-4 / (nmusc * (t1 - t0))) #the "trick" here is normalizing by simulation time
    if parallel is not None:
        solver.set_parallel(int(parallel))

    area_weight_muscle_effort(study.updProblem(), model)
    # Note!! This is where we set our area weighted effort! 

    solver.resetProblem(study.updProblem())                # REQUIRED before createGuess

    guess = solver.createGuess()
    n = guess.getNumTimes()
    # Start every normalized_tendon_force at the safe constant 0.1 
    # pure default guess can put the implicit tendon states in a numerically bad region
    for nm in guess.getStateNames():
        if "normalized_tendon_force" in nm:
            guess.setState(nm, osim.Vector(n, 0.1))
    solver.setGuess(guess)
    return study, model, t0, t1


def solve_chain(track, cycle, parallel=None, force=False):
    # Solve one (track, cycle); on success write full + exact-cropped .sto.
    out_dir = os.path.join(RESULTS, track)
    os.makedirs(out_dir, exist_ok=True)
    full = os.path.join(out_dir, f"cycle{cycle}_compliant_solution.sto")
    cropped = os.path.join(out_dir, f"cycle{cycle}_compliant_CROPPED_solution.sto")
    failed = os.path.join(out_dir, f"cycle{cycle}_compliant_FAILED_solution.sto")
    if os.path.exists(full) and os.path.getsize(full) > 0 and not force:
        if valid_full_solution(full):
            print(f"SKIP (exists, validated): {os.path.basename(full)}", flush=True)
            return dict(full=full, cropped=cropped, success=None, skipped=True)
        print(f"RE-SOLVE (existing {os.path.basename(full)} failed validation)", flush=True)

    crop0, crop1 = crop_window(cycle)
    extloads = resolve_external_loads(out_dir)
    # `model` stays bound on purpose: it must outlive study.solve() (a GC'd Model behind a live study is a dangling reference).
    study, model, t0, t1 = build_study(track, cycle, extloads, parallel)

    t_wall0 = time.time()
    ms = study.solve()
    wall = time.time() - t_wall0
    ok = ms.success()
    obj = ms.getObjective()
    print(f"=== {track} cycle{cycle} RESULT: success={ok} objective={obj:.5f} ===", flush=True)
    print(f"WALL_TIME_SECONDS: {wall:.1f} s ({wall / 3600:.2f} h)", flush=True)

    if not ok:
        ms.unseal()
        ms.write(failed)
        for stale in (full, cropped):
            if os.path.exists(stale):
                os.remove(stale)
        print(f"  FAILED -> wrote {os.path.basename(failed)}; canonical output withheld",
              flush=True)
        return dict(full=full, cropped=cropped, success=False, objective=obj, wall=wall)

    # Success: write to a temp file, validate, then atomically publish the canonical name
    # temp name keeps the .sto extension (TimeSeriesTable dispatches its reader by extension)
    root, ext = os.path.splitext(full)
    tmp = root + ".writing" + ext
    os.makedirs(os.path.dirname(tmp) or ".", exist_ok=True)
    ms.write(tmp)
    if not os.path.exists(tmp) or not valid_full_solution(tmp):
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"{track} cycle{cycle}: solve reported success (objective {obj:.5f}) "
                           f"but wrote a missing/non-finite solution")
    os.replace(tmp, full)
    if os.path.exists(failed):
        os.remove(failed)
    write_cropped_exact(full, crop0, crop1, cropped)

    tab = osim.TimeSeriesTable(full)
    tt = np.array(tab.getIndependentColumn())
    gaps = np.array([abs(np.interp(crop0, tt, tab.getDependentColumn(L).to_numpy())
                         - np.interp(crop1, tt, tab.getDependentColumn(L).to_numpy()))
                     for L in tab.getColumnLabels() if L.endswith("/activation")])
    print(f"  cropped boundary gap |a(t0)-a(t1)| over {len(gaps)} muscles: "
          f"mean {gaps.mean():.3f} max {gaps.max():.3f}", flush=True)
    return dict(full=full, cropped=cropped, success=ok, objective=obj, wall=wall)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", default="all", choices=["davis", "raj", "all"])
    ap.add_argument("--cycle", default="all", help="1..5 or 'all'")
    ap.add_argument("--parallel", type=int, default=None,
                    help="MocoCasADiSolver.set_parallel(N). NOT a thread count -- 0 = serial, "
                         "1 = ALL cores (Moco's default), N>1 = N parallel jobs. Omit to use "
                         "Moco's default (already all cores).")
    ap.add_argument("--force", action="store_true", help="re-solve even if output exists")
    args = ap.parse_args()
    tracks = list(TRACKS) if args.track == "all" else [args.track]
    cycles = list(CYCLES) if args.cycle == "all" else [int(args.cycle)]

    t_batch = time.time()
    for track in tracks:
        for cycle in cycles:
            solve_chain(track, cycle, args.parallel, args.force)
    print(f"\nBATCH DONE ({len(tracks)} tracks x {len(cycles)} cycles) in "
          f"{(time.time() - t_batch) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
