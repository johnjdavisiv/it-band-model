"""
Check operating lengths to ensure we aare not in a pathological situation 

"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np
import opensim as osim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.paper1io import RESULTS, MODELS

osim.Logger.setLevelString("Error")
AGG = os.path.join(RESULTS, "agg")
SRC = os.path.join(AGG, "tracts_registered_davis.csv")
ITB = ["glmax12_ITB_r", "glmax34_ITB_r", "tfl12_ITB_r"]
FEMORAL = ["glmax1_r", "glmax2_r", "glmax3_r", "glmax4_r"]
GOOD_MULT = 0.90 #"healthy "


def afl_curve():
    m = osim.Model(os.path.join(MODELS, "Davis2026_subject01_DGF.osim"))
    m.initSystem()
    ms = m.getMuscles() # bind the set bc get() returns into a temporary
    muscles = [osim.DeGrooteFregly2016Muscle.safeDownCast(ms.get(n)) for n in ITB + FEMORAL]
    widths = {round(mu.get_active_force_width_scale(), 9) for mu in muscles}
    assert len(widths) == 1, f"tracts do not share one active-FL width: {widths}"
    ref = muscles[0]
    probe = np.linspace(0.4, 1.8, 15)
    for mu in muscles[1:]:
        assert np.allclose([mu.calcActiveForceLengthMultiplier(x) for x in probe],
                           [ref.calcActiveForceLengthMultiplier(x) for x in probe],
                           atol=1e-12), "tracts do not share one active-FL curve"
    # Risky GC stuff!
    # `ref` is a reference INTO `m`; if `m` is collected when this function returns, the
    # next call reads freed memory (it surfaces as a bogus "0 properties in table" from
    # the property table). Pin model, set and muscle in the closure via a default arg.
    def multiplier(x, _keep=(m, ms, muscles, ref)):
        return np.array([ref.calcActiveForceLengthMultiplier(float(v))
                         for v in np.atleast_1d(x)])

    return multiplier, float(ref.get_active_force_width_scale())


def fv_curve():
    m = osim.Model(os.path.join(MODELS, "Davis2026_subject01_DGF.osim"))
    m.initSystem()
    ms = m.getMuscles()
    muscles = [osim.DeGrooteFregly2016Muscle.safeDownCast(ms.get(n)) for n in ITB + FEMORAL]
    vmaxes = {round(mu.getMaxContractionVelocity(), 9) for mu in muscles}
    assert len(vmaxes) == 1, f"tracts do not share one Vmax: {vmaxes}"
    ref = muscles[0]
    probe = np.linspace(-1.0, 1.0, 21)
    for mu in muscles[1:]:
        assert np.allclose([mu.calcForceVelocityMultiplier(x) for x in probe],
                           [ref.calcForceVelocityMultiplier(x) for x in probe],
                           atol=1e-12), "tracts do not share one force-velocity curve"

    def multiplier(x, _keep=(m, ms, muscles, ref)):     # pin: see afl_curve()
        return np.array([ref.calcForceVelocityMultiplier(float(v))
                         for v in np.atleast_1d(x)])

    return multiplier, vmaxes.pop()


def load(column="norm_fiber_length"):
    by = defaultdict(lambda: defaultdict(dict))    # tract -> cycle -> pct -> value
    with open(SRC) as f:
        for r in csv.DictReader(f):
            by[r["tract"]][int(r["cycle"])][float(r["pct"])] = float(r[column])
    out = {}
    for t, percyc in by.items():
        cyc = sorted(percyc)
        p = np.array(sorted(percyc[cyc[0]]))
        out[t] = (p, np.array([[percyc[c][x] for x in p] for c in cyc]))
    return out


def main():
    if not os.path.exists(SRC):
        print(f"missing {SRC} -- run 05_aggregate.py first")
        return
    afl, width = afl_curve()
    fvm, vmax = fv_curve()
    data = load()
    missing = [t for t in ITB + FEMORAL if t not in data]
    assert not missing, (f"{missing} absent from {os.path.basename(SRC)} -- re-run "
                         "05_aggregate.py (it realizes all 7 tracts, not just the 3 ITB)")

    rows = []
    for t in ITB + FEMORAL:
        p, st = data[t]
        mult = afl(st.ravel()).reshape(st.shape)
        rows.append(dict(
            tract=t, insertion=("ITB" if t in ITB else "femoral"),
            nfl_min=round(float(st.min()), 4), nfl_mean=round(float(st.mean()), 4),
            nfl_max=round(float(st.max()), 4),
            afl_mult_at_nfl_min=round(float(afl(st.min())[0]), 4),
            afl_mult_at_nfl_max=round(float(afl(st.max())[0]), 4),
            afl_mult_min=round(float(mult.min()), 4),
            afl_mult_mean=round(float(mult.mean()), 4),
            frac_cycle_above_0p90_mult=round(float(np.mean(mult >= GOOD_MULT)), 4),
            active_force_width_scale=width))

    out_csv = os.path.join(RESULTS, "fiber_operating_length.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    byname = {r["tract"]: r for r in rows}

    # active-FL multiplier (% of peak) at each tract's own peak-tendon-force instant
    mult_series = {t: (data[t][0], afl(data[t][1].ravel()).reshape(data[t][1].shape) * 100)
                   for t in ITB + FEMORAL}
    force = load("tendon_force_N")
    peak_mark = {}
    for t in ITB + FEMORAL:
        p, F = force[t]
        pm, M = mult_series[t]
        i = int(np.argmax(F.mean(0)))            # phase of peak MEAN tendon force
        peak_mark[t] = (pm[i], float(M.mean(0)[i]))

    # ---- force-velocity, at peak force and worst
    nfv = load("norm_fiber_velocity")
    for t in ITB + FEMORAL:
        p, V = nfv[t]
        _, F = force[t]
        _, L = data[t]
        vbar, fbar = V.mean(0), F.mean(0)
        fv_series = fvm(vbar)
        afl_series = afl(L.mean(0))
        i_pk = int(np.argmax(fbar))
        i_fv = int(np.argmin(fv_series))
        i_afl = int(np.argmin(afl_series))
        loaded = fbar >= 0.25 * fbar.max()
        j = int(np.arange(len(fv_series))[loaded][np.argmin(fv_series[loaded])])
        r = byname[t]
        r["afl_mult_at_peak_force_pct"] = round(peak_mark[t][1], 2)
        r["peak_force_phase_pct"] = round(peak_mark[t][0], 1)
        r["fv_mult_at_peak_force_pct"] = round(float(fv_series[i_pk]) * 100, 2)
        r["norm_fiber_velocity_at_peak_force"] = round(float(vbar[i_pk]), 4)
        r["afl_mult_min_pct"] = round(float(afl_series[i_afl]) * 100, 2)
        r["afl_mult_min_phase_pct"] = round(float(p[i_afl]), 1)
        r["fv_mult_min_pct"] = round(float(fv_series[i_fv]) * 100, 2)
        r["fv_mult_min_phase_pct"] = round(float(p[i_fv]), 1)
        r["force_at_fv_min_pct_of_peak"] = round(float(fbar[i_fv] / fbar.max()) * 100, 1)
        r["fv_mult_min_loaded_pct"] = round(float(fv_series[j]) * 100, 2)
        r["fv_mult_min_loaded_phase_pct"] = round(float(p[j]), 1)
        r["max_contraction_velocity"] = vmax
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Fibre operating length, davis, 5 cycles "
          f"(active-FL width {width:g}, shared by all tracts)\n")
    hdr = (f"{'tract':<16}{'insert':<9}{'nfl min':>9}{'mean':>8}{'max':>8}"
           f"{'min f_L':>9}{'mean f_L':>10}{'% >= 0.90':>11}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['tract']:<16}{r['insertion']:<9}{r['nfl_min']:>9.3f}"
              f"{r['nfl_mean']:>8.3f}{r['nfl_max']:>8.3f}{r['afl_mult_min']:>9.3f}"
              f"{r['afl_mult_mean']:>10.3f}{r['frac_cycle_above_0p90_mult'] * 100:>10.0f}%")
    print("\n  nfl = normalized fibre length (1.0 = optimal).  f_L = active force-length "
          "multiplier\n  at that length (1.0 = full force capacity). The multiplier is the "
          "interpretable one:\n  a long fibre only matters if f_L has actually fallen.")
    print("\n  At each tract's own peak-tendon-force instant the fibre can still make:")
    for t in ITB + FEMORAL:
        print(f"      {t:<16}{peak_mark[t][1]:5.1f}% of peak  (at {peak_mark[t][0]:.0f}% of the cycle)")
    print(f"\nsaved: {os.path.relpath(out_csv, RESULTS)}")

if __name__ == "__main__":
    main()
