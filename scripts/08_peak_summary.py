""" 

ITB peak forces (questionable? )

"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np
import opensim as osim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.paper1io import (RESULTS, MODELS, body_mass_kg, body_weight_N,
                          assert_tracks_same_mass) 

osim.Logger.setLevelString("Error")
AGG = os.path.join(RESULTS, "agg")
SRC = os.path.join(AGG, "tracts_registered_davis.csv")
ITB = ["glmax12_ITB_r", "glmax34_ITB_r", "tfl12_ITB_r"]
FEMORAL = ["glmax1_r", "glmax2_r", "glmax3_r", "glmax4_r"]
TOTAL = "ITB_total (sum of 3)"


def model_params():
    """F0M and tendon strain-at-one-norm-force per tract, from the solved DGF model."""
    m = osim.Model(os.path.join(MODELS, "Davis2026_subject01_DGF.osim"))
    m.initSystem()
    ms = m.getMuscles() 
    out = {}
    for n in ITB + FEMORAL:
        mu = osim.DeGrooteFregly2016Muscle.safeDownCast(ms.get(n))
        out[n] = (mu.getMaxIsometricForce(), mu.get_tendon_strain_at_one_norm_force())
    return out


def load():
    """tract -> (pct grid, force (ncyc,npct), strain (ncyc,npct))."""
    f = defaultdict(lambda: defaultdict(dict))
    s = defaultdict(lambda: defaultdict(dict))
    with open(SRC) as fh:
        for r in csv.DictReader(fh):
            t, c, p = r["tract"], int(r["cycle"]), float(r["pct"])
            f[t][c][p] = float(r["tendon_force_N"])
            s[t][c][p] = float(r["tendon_strain"])
    out = {}
    for t in f:
        cyc = sorted(f[t])
        p = np.array(sorted(f[t][cyc[0]]))
        out[t] = (p,
                  np.array([[f[t][c][x] for x in p] for c in cyc]),
                  np.array([[s[t][c][x] for x in p] for c in cyc]))
    return out


def peaks(p, stacked):
    """Per-cycle peak value and the phase at which it occurs -> (values, phases)."""
    idx = np.argmax(stacked, axis=1)
    return stacked[np.arange(stacked.shape[0]), idx], p[idx]


def ms(v):
    return float(np.mean(v)), float(np.std(v, ddof=1))


PHASE_SPREAD_LIMIT_PCT = 20.0


def phase_stat(phases):
    """(mean, sd, spread, multimodal) for a set of per-cycle peak phases."""
    m, s = ms(phases)
    spread = float(np.max(phases) - np.min(phases))
    return m, s, spread, spread > PHASE_SPREAD_LIMIT_PCT


def main():
    if not os.path.exists(SRC):
        print(f"missing {SRC} -- run 05_aggregate.py first")
        return
    assert_tracks_same_mass()
    bw, mass = body_weight_N("davis"), body_mass_kg("davis")
    par = model_params()
    data = load()
    missing = [t for t in ITB + FEMORAL if t not in data]
    assert not missing, f"{missing} absent from {os.path.basename(SRC)} -- re-run 05_aggregate.py"

    rows = []
    for t in ITB + FEMORAL:
        p, F, S = data[t]
        f0m, e0 = par[t]
        fv, fp = peaks(p, F)
        sv, sp = peaks(p, S)
        fm, fsd = ms(fv)
        ph_m, ph_sd, ph_spread, ph_bad = phase_stat(fp)
        rows.append(dict(
            tract=t, group=("ITB" if t in ITB else "femoral"), n_cycles=F.shape[0],
            F0M_N=round(f0m, 2), e0_pct=round(e0 * 100, 3),
            # BW first: it is the reported unit. SD is converted, not recomputed -- BW is a
            # constant scale factor, so sd(F/BW) == sd(F)/BW exactly.
            peak_force_BW_mean=round(fm / bw, 4), peak_force_BW_sd=round(fsd / bw, 4),
            peak_force_N_mean=round(fm, 1), peak_force_N_sd=round(fsd, 1),
            peak_force_pct_F0M=round(fm / f0m * 100, 1),
            peak_force_phase_pct_mean=round(ph_m, 1),
            peak_force_phase_pct_sd=round(ph_sd, 1),
            peak_force_phase_spread_pct=round(ph_spread, 1),
            peak_force_phase_multimodal=ph_bad,
            peak_strain_pct_mean=round(ms(sv)[0] * 100, 3),
            peak_strain_pct_sd=round(ms(sv)[1] * 100, 3),
            peak_strain_frac_of_e0=round(ms(sv)[0] / e0, 3),
            peak_strain_phase_pct_mean=round(ms(sp)[0], 1)))

    # ---- the summed ITB force
    p = data[ITB[0]][0]
    tot = sum(data[t][1] for t in ITB)
    f0m_tot = sum(par[t][0] for t in ITB)
    tv, tp = peaks(p, tot)
    tm, tsd = ms(tv)
    tph_m, tph_sd, tph_spread, tph_bad = phase_stat(tp)
    naive = sum(r["peak_force_N_mean"] for r in rows if r["group"] == "ITB")
    rows.append(dict(
        tract=TOTAL, group="ITB", n_cycles=tot.shape[0],
        F0M_N=round(f0m_tot, 2), e0_pct="",
        peak_force_BW_mean=round(tm / bw, 4), peak_force_BW_sd=round(tsd / bw, 4),
        peak_force_N_mean=round(tm, 1), peak_force_N_sd=round(tsd, 1),
        peak_force_pct_F0M=round(tm / f0m_tot * 100, 1),
        peak_force_phase_pct_mean=round(tph_m, 1),
        peak_force_phase_pct_sd=round(tph_sd, 1),
        peak_force_phase_spread_pct=round(tph_spread, 1),
        peak_force_phase_multimodal=tph_bad,
        peak_strain_pct_mean="", peak_strain_pct_sd="",
        peak_strain_frac_of_e0="", peak_strain_phase_pct_mean=""))

    out = os.path.join(RESULTS, "itb_peak_summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    itb_rows = [r for r in rows if r["group"] == "ITB"]
    print(f"ITB tract peaks over 5 reported strides (per-cycle peaks, then mean ± SD)")
    print(f"body weight = {bw:.1f} N ({mass:.2f} kg, RRA mass-adjusted solve model)\n")
    hdr = (f"{'tract':<22}{'peak force (BW)':>17}{'(N)':>7}{'%F0M':>7}{'@ % cycle':>12}"
           f"{'peak strain (%)':>19}{'/e0':>7}{'@ %':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in itb_rows:
        strain = ("       —          —      —" if r["tract"] == TOTAL else
                  f"{r['peak_strain_pct_mean']:>13.2f} ±{r['peak_strain_pct_sd']:<4.2f}"
                  f"{r['peak_strain_frac_of_e0']:>7.2f}"
                  f"{r['peak_strain_phase_pct_mean']:>7.0f}")
        phase = ("  multimodal" if r["peak_force_phase_multimodal"] else
                 f"{r['peak_force_phase_pct_mean']:>8.0f} ±{r['peak_force_phase_pct_sd']:<3.0f}")
        print(f"{r['tract']:<22}{r['peak_force_BW_mean']:>11.3f} ±{r['peak_force_BW_sd']:<5.3f}"
              f"{r['peak_force_N_mean']:>7.0f}{r['peak_force_pct_F0M']:>7.0f}"
              f"{phase}{strain}")
    print(f"\n  The three tracts peak {abs(itb_rows[1]['peak_force_phase_pct_mean'] - itb_rows[0]['peak_force_phase_pct_mean']):.0f} "
          f"and {abs(itb_rows[2]['peak_force_phase_pct_mean'] - itb_rows[0]['peak_force_phase_pct_mean']):.0f} "
          "percentage points apart, so the SUM is a plateau:")
    print(f"  total peak {rows[-1]['peak_force_BW_mean']:.3f} BW ({rows[-1]['peak_force_N_mean']:.0f} N) "
          f"vs {naive / bw:.3f} BW ({naive:.0f} N) if the three peaks coincided "
          f"({rows[-1]['peak_force_N_mean'] / naive * 100:.0f}%). Lead with the per-tract peaks.")
    print("\n  Strain is not summable, so the total row reports force only.")
    print(f"  '/e0' = peak strain as a fraction of that tract's own strain-at-one-norm-force.")
    print(f"\nsaved: {os.path.relpath(out, RESULTS)}")


if __name__ == "__main__":
    main()
