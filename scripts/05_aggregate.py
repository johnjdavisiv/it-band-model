"""

Process the results to get what we need for results reporting. 

Notes: 

(1) muscle activations (both tracks): register each cycle's activations to 0-100% of the
    reported gait cycle.
(2) ITB tract mechanics (davis): recompute tendon force/strain/fiber length by realizing
    Davis2026_subject01_DGF.osim at each prescribed pose with the solution's muscle states,
    then register.

The full padded solution is read (not the cropped artifact) and registered over the exact
[IC_k, IC_{k+1}] stride, so the reported 0-100% cycle spans the initial contacts precisely
regardless of where the solver's mesh nodes fell (which aren't always on the cutoffs)

Outputs (results/agg/):
activations_registered_<track>.csv  cols: track,cycle,pct,muscle,activation
tracts_registered_davis.csv         cols: cycle,pct,tract,insertion,tendon_force_N,
                                    tendon_force_BW,tendon_strain,norm_fiber_length,
                                    mtu_length_m,norm_fiber_velocity
                                    (all 7 grafted tracts: 3 ITB + 4 femoral glute-max)
shared_muscle_drift.csv             per-muscle median |davis - raj| activation drift over
                                    the directly comparable (name+F0M-identical) muscles

"""
import os
import sys
import csv

import numpy as np
import opensim as osim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.paper1io import (RESULTS, MODELS, TRACKS, CYCLES, kinematics, crop_window,
                          body_weight_N, assert_tracks_same_mass)              # noqa: E402
from lib.registration import register                                          # noqa: E402

osim.Logger.setLevelString("Error")
AGG = os.path.join(RESULTS, "agg")

# All SEVEN grafted tracts, not just the 3 ITB ones! The four femoral glute-max compartments are
# the comparison group for fibre operating length: they share the graft's geometry and strength
# apportionment but insert on the femur rather than the band, so "do the ITB tracts operate on a
# sensible part of the force-length curve" is only answerable against them.
ITB = ["glmax12_ITB_r", "glmax34_ITB_r", "tfl12_ITB_r"] # ITB-inserting
FEMORAL = ["glmax1_r", "glmax2_r", "glmax3_r", "glmax4_r"] # femur-inserting
TRACTS = ITB + FEMORAL
NPT = 101

def present_cycles(track):
    out = []
    for c in CYCLES:
        p = os.path.join(RESULTS, track, f"cycle{c}_compliant_solution.sto")
        if os.path.exists(p):
            out.append((c, p))
    return out

def aggregate_activations(track):
    rows = []
    for c, path in present_cycles(track):
        t = osim.TimeSeriesTable(path)
        tt = np.array(t.getIndependentColumn())
        crop0, crop1 = crop_window(c)
        for L in t.getColumnLabels():
            if not L.endswith("/activation"):
                continue
            musc = L.split("/")[-2]
            phase, y = register(tt, t.getDependentColumn(L).to_numpy(), crop0, crop1, n=NPT)
            rows += [(track, c, round(float(phase[k]), 3), musc, float(y[k]))
                     for k in range(len(phase))]
    if not rows:
        print(f"[{track}] activations: no compliant solutions yet -- skipped")
        return None
    out = os.path.join(AGG, f"activations_registered_{track}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track", "cycle", "pct", "muscle", "activation"])
        w.writerows(rows)
    print(f"[{track}] activations: {len(set(r[1] for r in rows))} cycles, "
          f"{len(set(r[3] for r in rows))} muscles -> {os.path.basename(out)}")
    return os.path.basename(out)


def aggregate_itb():
    """Recompute each grafted tract's tendon mechanics from the solution states.

    The read-out model is realized at each prescribed pose with the solution's activation and
    normalized-tendon-force states; tendon force/strain and fibre length then come from the
    muscle's own equilibrium, exactly as the solver saw them."""
    cycles = present_cycles("davis")
    if not cycles:
        print("[davis] ITB: no compliant solutions yet -- skipped")
        return None
    bw = body_weight_N("davis")
    model = osim.Model(os.path.join(MODELS, "Davis2026_subject01_DGF.osim"))
    state = model.initSystem()
    coordset = model.getCoordinateSet()
    muscles = model.getMuscles()
    muscmap = {m: osim.DeGrooteFregly2016Muscle.safeDownCast(muscles.get(m)) for m in TRACTS}
    rows = []
    for c, path in cycles:
        ktab = osim.TimeSeriesTable(kinematics(c))
        ktimes = np.array(ktab.getIndependentColumn())
        kvals = {cc: ktab.getDependentColumn(cc).to_numpy() for cc in ktab.getColumnLabels()}
        indep = [coordset.get(i).getName() for i in range(coordset.getSize())
                 if coordset.get(i).getName() in kvals]
        t = osim.TimeSeriesTable(path)
        tt = np.array(t.getIndependentColumn())
        crop0, crop1 = crop_window(c)
        win = (tt >= crop0) & (tt <= crop1)
        clab = list(t.getColumnLabels())

        def sst(m, kind):
            for L in clab:
                if L.endswith(f"/{m}/{kind}"):
                    return t.getDependentColumn(L).to_numpy()
            return None

        act = {m: sst(m, "activation") for m in TRACTS}
        ntf = {m: sst(m, "normalized_tendon_force") for m in TRACTS}
        raw = {m: {k: np.zeros(len(tt)) for k in ("tf", "ts", "nfl", "mtu")} for m in TRACTS}
        for i, ti in enumerate(tt):
            for nm in indep:
                coordset.get(nm).setValue(state, float(np.interp(ti, ktimes, kvals[nm])), False)
            model.assemble(state)
            for m in TRACTS:
                model.setStateVariableValue(state, f"/forceset/{m}/activation",
                                            float(act[m][i]))
                model.setStateVariableValue(state, f"/forceset/{m}/normalized_tendon_force",
                                            float(ntf[m][i]))
            model.realizeDynamics(state)
            for m in TRACTS:
                mu = muscmap[m]
                raw[m]["tf"][i] = mu.getTendonForce(state)
                raw[m]["ts"][i] = mu.getTendonStrain(state)
                raw[m]["nfl"][i] = mu.getNormalizedFiberLength(state)
                raw[m]["mtu"][i] = mu.getLength(state)

        """
        What's going on here is the velocity is read fromm the solver's time grid, not the muscle's
        The state is realized with coordinate value then derivative'd to get velocity 

        """

        for m in TRACTS:
            raw[m]["nfv"] = np.gradient(raw[m]["nfl"], tt) / muscmap[m].getMaxContractionVelocity()
        for m in TRACTS:
            reg = {}
            for k in ("tf", "ts", "nfl", "mtu", "nfv"):
                phase, reg[k] = register(tt, raw[m][k], crop0, crop1, n=NPT)
            rows += [(c, round(float(phase[j]), 3), m, ("ITB" if m in ITB else "femoral"),
                      float(reg["tf"][j]), float(reg["tf"][j]) / bw, float(reg["ts"][j]),
                      float(reg["nfl"][j]), float(reg["mtu"][j]),
                      float(reg["nfv"][j])) for j in range(len(phase))]
        print(f"[davis] tracts cycle{c}: "
              + "  ".join(f"{m.split('_ITB')[0]} peakF={raw[m]['tf'][win].max():.0f}N"
                          for m in ITB))
    out = os.path.join(AGG, "tracts_registered_davis.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cycle", "pct", "tract", "insertion", "tendon_force_N", "tendon_force_BW",
                    "tendon_strain", "norm_fiber_length", "mtu_length_m",
                    "norm_fiber_velocity"])
        w.writerows(rows)
    print(f"[davis] tracts: {len(set(r[0] for r in rows))} cycles, {len(TRACTS)} tracts -> "
          f"{os.path.basename(out)}  (BW = {bw:.1f} N)")
    return os.path.basename(out)


# --- shared-muscle activation drift -------------------------------------------------------------
# The manuscript's "the graft does not perturb the rest of the model" statistic: median
# |davis - raj| of the cycle-mean activation, over the muscles the two models share unchanged.

ALIAS = {"tfl12_ITB_r": "tfl_r", "tfl12_ITB_l": "tfl_l"}   # same muscle, renamed + re-routed


def partition_muscles(dav_names, raj_names):
    """Classify every muscle in either model into comparable groups.

    A bare name intersection is misleading twice over: it counts as "shared" muscles whose
    F0M the graft re-apportioned, and it hides the grafted tracts entirely.

      shared      same muscle in both, identical F0M -> the drift statistic
      regraft     a name in both models carrying DIFFERENT F0M: the glute-max region
                  re-partitioned (only the TOTAL F0M is preserved); reported separately
      renamed     the same muscle under a different name (ALIAS): identical F0M, different
                  distal path -- a path change is exactly what the model is meant to alter
      davis_only  grafted tracts with no raj counterpart; raj_only the converse

    Membership is DERIVED by comparing F0M between the two solve models (no strength
    scaling is applied, so every untouched muscle is identical there) rather than
    hardcoded. ALIAS is the one thing that cannot be derived -- the anatomical claim is
    asserted, then F0M-checked."""
    f0m = {}
    for track, fn in (("davis", "Davis2026_subject01.osim"),
                      ("raj", "Rajagopal2023_subject01.osim")):
        m = osim.Model(os.path.join(MODELS, fn))
        m.initSystem()
        ms = m.getMuscles()  # bind the set; get() on a temporary dangles
        f0m[track] = {ms.get(i).getName(): ms.get(i).getMaxIsometricForce()
                      for i in range(ms.getSize())}
    shared, regraft = [], []
    for n in sorted(set(dav_names) & set(raj_names)):
        a, b = f0m["davis"].get(n), f0m["raj"].get(n)
        (shared if a is not None and b is not None and abs(a - b) < 1e-6
         else regraft).append(n)
    pair = {n: n for n in shared + regraft}
    renamed = []
    for d, r in ALIAS.items():
        if d in dav_names and r in raj_names:
            assert abs(f0m["davis"][d] - f0m["raj"][r]) < 1e-3, (
                f"ALIAS claims {d} == {r} but F0M differs: "
                f"{f0m['davis'][d]:.2f} vs {f0m['raj'][r]:.2f} N")
            pair[d] = r
            renamed.append(d)
    davis_only = sorted(set(dav_names) - set(raj_names) - set(renamed))
    raj_only = sorted(set(raj_names) - set(dav_names) - set(ALIAS.values()))
    return shared, regraft, sorted(renamed), davis_only, raj_only, pair


def _mean_curves(path):
    """muscle -> cycle-mean activation curve, from a registered CSV written above."""
    from collections import defaultdict
    by = defaultdict(lambda: defaultdict(dict))    # muscle -> cycle -> pct -> value
    with open(path) as f:
        for r in csv.DictReader(f):
            by[r["muscle"]][int(r["cycle"])][float(r["pct"])] = float(r["activation"])
    out = {}
    for m, percyc in by.items():
        cyc = sorted(percyc)
        p = sorted(percyc[cyc[0]])
        out[m] = np.array([[percyc[c][x] for x in p] for c in cyc]).mean(0)
    return out


def write_drift():
    pd_ = os.path.join(AGG, "activations_registered_davis.csv")
    pr_ = os.path.join(AGG, "activations_registered_raj.csv")
    if not (os.path.exists(pd_) and os.path.exists(pr_)):
        print("drift: need both tracks' registered activations -- skipped")
        return None
    dav, raj = _mean_curves(pd_), _mean_curves(pr_)
    shared, regraft, renamed, davis_only, raj_only, pair = partition_muscles(dav, raj)
    drift = [(m, float(np.median(np.abs(dav[m] - raj[m])))) for m in shared]
    out = os.path.join(AGG, "shared_muscle_drift.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["muscle", "median_abs_drift", "class"])
        for m, d in sorted(drift, key=lambda x: -x[1]):
            w.writerow([m, f"{d:.6f}", "shared"])
        # drift is computable for these (a raj counterpart exists) but is reported
        # separately: it measures the graft, not the agreement between the models
        for m, cls in ([(x, "regraft") for x in regraft]
                       + [(x, "renamed") for x in renamed]):
            w.writerow([m, f"{float(np.median(np.abs(dav[m] - raj[pair[m]]))):.6f}", cls])
        for m, cls in ([(x, "davis_only") for x in davis_only]
                       + [(x, "raj_only") for x in raj_only]):
            w.writerow([m, "", cls])
    med = float(np.median([d for _, d in drift]))
    print(f"drift: median |davis-raj| = {med:.4f} over {len(shared)} directly comparable "
          f"muscles -> {os.path.basename(out)}")
    return os.path.basename(out)


def main():
    assert_tracks_same_mass() #ensure BW the same
    os.makedirs(AGG, exist_ok=True)
    written = [aggregate_activations(t) for t in TRACKS] + [aggregate_itb(), write_drift()]
    done = [w for w in written if w]
    print(f"DONE aggregate -- wrote {len(done)}/4 artifacts:")
    for w in done:
        print(f"    {w}")
    if len(done) < 4:
        print(f"  WARNING: {4 - len(done)} artifact(s) not written (no solutions present) -- "
              f"results/agg/ is incomplete")


if __name__ == "__main__":
    main()
