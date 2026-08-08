"""

Tendon force-length curve fidelity for the three IT-band tracts.

"""
import argparse
import csv
import json
import os

import numpy as np
import opensim as osim

osim.Logger.setLevelString("Error")

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, "..", ".."))

MILLARD_MODEL = os.path.join(PKG, "models", "Davis2026.osim")
DGF_MODEL = os.path.join(PKG, "models", "Davis2026_DGF.osim")
# Eng's cadaveric control points + the step-2 fitted Millard params, vendored into
# eng-reference/ so validation/ is self-contained.
ENG_REF = os.path.join(HERE, "eng-reference")
ENG_POINTS = os.path.join(ENG_REF, "eng_itb_tendon_curve_spline_points.csv")
ENG_PARAMS = os.path.join(ENG_REF, "eng_tendon_force_length_millard_params.json")

TRACTS = ["glmax12_ITB", "glmax34_ITB", "tfl12_ITB"]
SIDE = "_r" # the graft is mirrored; right limb is the reported one
NGRID = 501

def read_eng_points():
    pts = {t: [] for t in TRACTS}
    with open(ENG_POINTS, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            m = row["muscle"].strip()
            if m in pts:
                pts[m].append((float(row["tendon_strain"]),
                               float(row["normalized_force"])))
    for t in TRACTS:
        pts[t].sort()
    return pts

def build_simm(points):
    spl = osim.SimmSpline()
    for x, y in points:
        spl.addPoint(x, y)
    return spl

def eval_simm(spl, strains):
    v = osim.Vector(1, 0.0)
    out = np.empty(len(strains))
    for i, s in enumerate(strains):
        v.set(0, float(s))
        out[i] = spl.calcValue(v)
    return out


def eval_millard(curve, strains):
    v = osim.Vector(1, 0.0)
    out = np.empty(len(strains))
    for i, s in enumerate(strains):
        v.set(0, 1.0 + float(s))
        out[i] = curve.calcValue(v)
    return out

def eval_dgf(muscle, strains):
    return np.array([muscle.calcTendonForceMultiplier(1.0 + float(s)) for s in strains])

def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

def maxerr(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))

# --- the observed operating range
OBS_SRC = os.path.join(PKG, "results", "agg", "tracts_registered_davis.csv")
SIDE_SUFFIX = "_r"

def observed_ranges():
    if not os.path.exists(OBS_SRC):
        return {}
    cols = {}
    with open(OBS_SRC, newline="") as f:
        rdr = csv.DictReader(f)
        has_bw = "tendon_force_BW" in (rdr.fieldnames or [])
        for row in rdr:
            t = row["tract"]
            if not t.endswith(SIDE_SUFFIX):
                continue
            base = t[: -len(SIDE_SUFFIX)]
            if base not in TRACTS:
                continue
            d = cols.setdefault(base, dict(strains=[], forces=[], forces_bw=[], cycles=set()))
            d["strains"].append(float(row["tendon_strain"]))
            d["forces"].append(float(row["tendon_force_N"]))
            if has_bw:
                d["forces_bw"].append(float(row["tendon_force_BW"]))
            d["cycles"].add(row["cycle"])
    out = {}
    for t, d in cols.items():
        out[t] = dict(strain_min=min(d["strains"]), strain_max=max(d["strains"]),
                      force_N_max=max(d["forces"]),
                      force_BW_max=(max(d["forces_bw"]) if d["forces_bw"] else float("nan")),
                      n_cycles=len(d["cycles"]))
    return out

def invert(strains, forces, query_force):
    """Strain at which a monotonic force-strain curve reaches `query_force`.

    Both curves are monotonically increasing over the sampled window (the Eng spline is
    asserted monotonic in main()), so a plain interpolation on the sampled curve is exact to
    the sampling resolution -- no root-find needed."""
    return float(np.interp(query_force, forces, strains))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-strain", type=float, default=0.10,
                    help="upper edge of the reported strain window (default 0.10)")
    args = ap.parse_args()

    strains = np.linspace(0.0, args.max_strain, NGRID)
    eng_pts = read_eng_points()
    global OBS
    OBS = observed_ranges()
    if not OBS:
        print(f"⚠ {os.path.basename(OBS_SRC)} not found -- the 'obs' (observed operating "
              f"range) window is SKIPPED. Run scripts/05_aggregate.py to enable it.\n")

    mm_model = osim.Model(MILLARD_MODEL)
    mm_model.initSystem()
    dg_model = osim.Model(DGF_MODEL)
    dg_model.initSystem()
    mm_set = mm_model.getMuscles()      # bind the set: get() returns into a temporary
    dg_set = dg_model.getMuscles()

    with open(ENG_PARAMS) as f:
        params = json.load(f)["tendon_force_length_curves"]

    curves, summary = [], []
    for t in TRACTS:
        mm = osim.Millard2012EquilibriumMuscle.safeDownCast(mm_set.get(t + SIDE))
        dg = osim.DeGrooteFregly2016Muscle.safeDownCast(dg_set.get(t + SIDE))
        assert mm is not None and dg is not None, f"missing {t}{SIDE}"

        tfl = mm.getTendonForceLengthCurve()
        spl = build_simm(eng_pts[t])

        y_eng = eval_simm(spl, strains)
        y_mm = eval_millard(tfl, strains)
        y_dg = eval_dgf(dg, strains)

        # --- provenance guards -------------------------------------------------
        # the shipped model must still carry the step-2 fitted curve ...
        e0_json = params[t]["strain_at_one_norm_force"]
        e0_mm = tfl.get_strain_at_one_norm_force()
        assert abs(e0_mm - e0_json) < 1e-9, f"{t}: model e0 {e0_mm} != fitted {e0_json}"
        # ... and the conversion must have carried e0 across (the one transferable param)
        e0_dg = dg.get_tendon_strain_at_one_norm_force()
        assert abs(e0_dg - e0_mm) < 1e-3, f"{t}: DGF e0 {e0_dg} != Millard {e0_mm}"
        # both curves must pass through one norm force AT e0, by definition -- so any
        # divergence below is curve SHAPE, not a mis-set parameter
        assert abs(eval_dgf(dg, [e0_mm])[0] - 1.0) < 1e-6, f"{t}: DGF not 1.0 at e0"
        assert abs(eval_millard(tfl, [e0_mm])[0] - 1.0) < 1e-6, f"{t}: Millard not 1.0 at e0"
        # the SIMM spline must not overshoot between control points in this window
        assert np.all(np.diff(y_eng) > -1e-9), f"{t}: Eng spline non-monotonic in window"

        for i, s in enumerate(strains):
            curves.append(dict(tract=t, strain=f"{s:.6f}",
                               strain_rel_e0=f"{s / e0_mm:.6f}", eng=f"{y_eng[i]:.6f}",
                               millard=f"{y_mm[i]:.6f}", dgf=f"{y_dg[i]:.6f}"))

        # strain at which the DGF curve first departs from Eng by more than TOL
        TOL = 0.10
        over = np.flatnonzero(np.abs(y_dg - y_eng) > TOL)
        onset = strains[over[0]] if over.size else np.nan

        # the same three comparisons over each tract's own toe region, 0 .. e0
        s_e0 = np.linspace(0.0, e0_mm, NGRID)
        e_e0, m_e0, d_e0 = (eval_simm(spl, s_e0), eval_millard(tfl, s_e0),
                            eval_dgf(dg, s_e0))

        # --- the OBSERVED window: 0 .. this tract's peak strain in the production solves ---
        obs = OBS.get(t)
        if obs:
            s_ob = np.linspace(0.0, obs["strain_max"], NGRID)
            e_ob, m_ob, d_ob = (eval_simm(spl, s_ob), eval_millard(tfl, s_ob),
                                eval_dgf(dg, s_ob))
            f0m = mm.getMaxIsometricForce()
            # FORCE error at a given strain (the "don't lead with this" number)
            dgf_minus_eng = d_ob - e_ob
            k = int(np.argmax(np.abs(dgf_minus_eng)))
            # STRAIN read-out error at a given force (the number the paper reports). Invert
            # both curves at the same force and difference the strains. Evaluated over the
            # observed FORCE range, using a shared strain grid wide enough to cover both.
            s_inv = np.linspace(0.0, max(obs["strain_max"] * 2.0, e0_mm * 1.5), 4001)
            d_inv, e_inv = eval_dgf(dg, s_inv), eval_simm(spl, s_inv)
            f_obs_peak = d_ob[-1]           # DGF force at the observed peak strain, in F0M
            fq = np.linspace(0.0, f_obs_peak, NGRID)
            err_pp = (np.array([invert(s_inv, d_inv, q) for q in fq])
                      - np.array([invert(s_inv, e_inv, q) for q in fq])) * 100.0
            obs_cols = dict(
                obs_n_cycles=obs["n_cycles"],
                obs_strain_max=f"{obs['strain_max']:.6f}",
                obs_strain_max_rel_e0=f"{obs['strain_max'] / e0_mm:.3f}",
                obs_peak_force_N=f"{obs['force_N_max']:.1f}",
                obs_peak_force_BW=f"{obs['force_BW_max']:.4f}",
                obs_rmse_millard_vs_eng=f"{rmse(m_ob, e_ob):.5f}",
                obs_rmse_dgf_vs_eng=f"{rmse(d_ob, e_ob):.5f}",
                obs_rmse_dgf_vs_millard=f"{rmse(d_ob, m_ob):.5f}",
                obs_maxerr_millard_vs_eng=f"{maxerr(m_ob, e_ob):.5f}",
                obs_maxerr_dgf_vs_eng=f"{maxerr(d_ob, e_ob):.5f}",
                obs_maxerr_dgf_vs_millard=f"{maxerr(d_ob, m_ob):.5f}",
                # signed worst force error + where it happens + what it is worth in newtons
                obs_worst_force_err_F0M=f"{dgf_minus_eng[k]:+.5f}",
                obs_worst_force_err_N=f"{dgf_minus_eng[k] * f0m:+.1f}",
                obs_worst_force_err_at_strain=f"{s_ob[k]:.6f}",
                obs_worst_force_err_at_rel_e0=f"{s_ob[k] / e0_mm:.3f}",
                # the reported quantity: strain read-out error, percentage points
                obs_strain_readout_err_pp_at_peak=f"{err_pp[-1]:+.3f}",
                obs_strain_readout_err_pp_max=f"{err_pp[int(np.argmax(np.abs(err_pp)))]:+.3f}",
            )
        else:
            obs_cols = {k: "" for k in (
                "obs_n_cycles", "obs_strain_max", "obs_strain_max_rel_e0", "obs_peak_force_N",
                "obs_peak_force_BW", "obs_rmse_millard_vs_eng", "obs_rmse_dgf_vs_eng",
                "obs_rmse_dgf_vs_millard", "obs_maxerr_millard_vs_eng", "obs_maxerr_dgf_vs_eng",
                "obs_maxerr_dgf_vs_millard", "obs_worst_force_err_F0M", "obs_worst_force_err_N",
                "obs_worst_force_err_at_strain", "obs_worst_force_err_at_rel_e0",
                "obs_strain_readout_err_pp_at_peak", "obs_strain_readout_err_pp_max")}

        summary.append(dict(
            tract=t,
            strain_at_one_norm_force=f"{e0_mm:.6f}",
            window_max_strain=f"{args.max_strain:.4f}",
            window_max_rel_e0=f"{args.max_strain / e0_mm:.3f}",
            eng_force_at_window_max=f"{y_eng[-1]:.4f}",
            dgf_divergence_onset_strain=("" if np.isnan(onset) else f"{onset:.5f}"),
            dgf_divergence_onset_rel_e0=("" if np.isnan(onset)
                                         else f"{onset / e0_mm:.3f}"),
            abs_rmse_millard_vs_eng=f"{rmse(y_mm, y_eng):.5f}",
            abs_rmse_dgf_vs_eng=f"{rmse(y_dg, y_eng):.5f}",
            abs_rmse_dgf_vs_millard=f"{rmse(y_dg, y_mm):.5f}",
            abs_maxerr_millard_vs_eng=f"{maxerr(y_mm, y_eng):.5f}",
            abs_maxerr_dgf_vs_eng=f"{maxerr(y_dg, y_eng):.5f}",
            abs_maxerr_dgf_vs_millard=f"{maxerr(y_dg, y_mm):.5f}",
            e0_rmse_millard_vs_eng=f"{rmse(m_e0, e_e0):.5f}",
            e0_rmse_dgf_vs_eng=f"{rmse(d_e0, e_e0):.5f}",
            e0_rmse_dgf_vs_millard=f"{rmse(d_e0, m_e0):.5f}",
            e0_maxerr_millard_vs_eng=f"{maxerr(m_e0, e_e0):.5f}",
            e0_maxerr_dgf_vs_eng=f"{maxerr(d_e0, e_e0):.5f}",
            e0_maxerr_dgf_vs_millard=f"{maxerr(d_e0, m_e0):.5f}",
            **obs_cols,
        ))

    out_curves = os.path.join(HERE, "tendon_curve_samples.csv")
    out_summary = os.path.join(HERE, "tendon_curve_rmse.csv")
    with open(out_curves, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curves[0].keys()))
        w.writeheader()
        w.writerows(curves)
    with open(out_summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    pct = args.max_strain * 100

    def table(prefix, title):
        print(f"\n{title}")
        hdr = (f"{'tract':<14}{'e0':>8}{'RMSE M-E':>10}{'RMSE D-E':>10}"
               f"{'RMSE D-M':>10}{'max M-E':>10}{'max D-E':>10}{'max D-M':>10}")
        print(hdr)
        print("-" * len(hdr))
        for r in summary:
            print(f"{r['tract']:<14}{float(r['strain_at_one_norm_force']):>8.4f}"
                  f"{float(r[prefix + 'rmse_millard_vs_eng']):>10.4f}"
                  f"{float(r[prefix + 'rmse_dgf_vs_eng']):>10.4f}"
                  f"{float(r[prefix + 'rmse_dgf_vs_millard']):>10.4f}"
                  f"{float(r[prefix + 'maxerr_millard_vs_eng']):>10.4f}"
                  f"{float(r[prefix + 'maxerr_dgf_vs_eng']):>10.4f}"
                  f"{float(r[prefix + 'maxerr_dgf_vs_millard']):>10.4f}")

    print(f"Tendon force-length fidelity ({NGRID} pts/window), "
          "normalized force units (1.0 = F0M)")
    print("  M-E = fitted Millard vs Eng spline   (the step-2 curve fit)")
    print("  D-E = DeGroote-Fregly vs Eng spline  (what the solver integrated)")
    print("  D-M = DeGroote-Fregly vs Millard     (the conversion penalty, H1)")

    if OBS:
        table("obs_", "[window 'obs'] 0 .. each tract's OBSERVED peak strain "
                      "(the 5 production cycles) — THE REPORTABLE WINDOW")
    table("e0_", f"[window 'e0']  0 .. each tract's own e0 (toe region, to one norm force)")
    table("abs_", f"[window 'abs'] 0 .. {pct:g}% strain (fixed, NOT comparable across tracts)")

    if OBS:
        print("\n[window 'obs'] the limitation stated in BOTH directions, over the range the "
              "model actually visited")
        hdr = (f"{'tract':<14}{'obs peak eps':>13}{'/e0':>7}{'peak F (N)':>12}{'(BW)':>7}"
               f"{'worst dF (F0M)':>16}{'(N)':>8}{'strain err @peak':>18}{'max':>8}")
        print(hdr)
        print("-" * len(hdr))
        for r in summary:
            if not r["obs_strain_max"]:
                continue
            print(f"{r['tract']:<14}{float(r['obs_strain_max']) * 100:>12.2f}%"
                  f"{float(r['obs_strain_max_rel_e0']):>7.2f}"
                  f"{float(r['obs_peak_force_N']):>12.1f}{float(r['obs_peak_force_BW']):>7.3f}"
                  f"{float(r['obs_worst_force_err_F0M']):>+16.4f}"
                  f"{float(r['obs_worst_force_err_N']):>+8.0f}"
                  f"{float(r['obs_strain_readout_err_pp_at_peak']):>+15.2f} pp"
                  f"{float(r['obs_strain_readout_err_pp_max']):>+8.2f}")
        print("\n  'worst dF' = max |DGF - Eng| at the SAME strain. Large, but do NOT lead with\n"
              "  it: in the solve, tendon force is set by the moment demand and the redundancy\n"
              "  solution, and the curve then sets the strain -- so this overstates how wrong\n"
              "  the reported FORCES are. 'strain err' is the inverse (same force, difference the\n"
              "  strains) and IS what transfers to the reported numbers, one-for-one, because\n"
              "  05_aggregate reads strain by inverting the DGF curve. Positive = DGF over-reads\n"
              "  strain relative to Eng's tissue. The error is zero at e0 by construction, so a\n"
              "  tract riding near its own e0 is the SAFE one -- which inverts the ranking you\n"
              "  would guess from the curve plots.")

    print(f"\n{'tract':<14}{'e0':>8}{'10% is':>9}   DGF departs from Eng by >0.10 F0M at")
    print("-" * 72)
    for r in summary:
        on = r["dgf_divergence_onset_strain"]
        rel = r["dgf_divergence_onset_rel_e0"]
        where = (f"strain {float(on) * 100:5.2f}%  = {float(rel):.2f} x e0"
                 if on else "never within window")
        print(f"{r['tract']:<14}{float(r['strain_at_one_norm_force']):>8.4f}"
              f"{float(r['window_max_rel_e0']):>8.2f}x   {where}")

    print(f"\nsaved: {out_summary}\nsaved: {out_curves}")
    print("(plot: make_figures.py check_F03 draws these curves)")
    return summary


if __name__ == "__main__":
    main()
