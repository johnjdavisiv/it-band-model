"""

QC check against Hicks 2015 guidelines for reserves, residuals, etc. 

"""
import csv
import os
import sys

import numpy as np
import opensim as osim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.paper1io import RESULTS, TRACKS, CYCLES, solve_base, GRF_MOT
from lib.solve_model import RESID_ROT_OPTF, RESID_TRANS_OPTF, RESERVE_OPTF

osim.Logger.setLevelString("Error")

RESID_OPTF = {"tilt": RESID_ROT_OPTF, "list": RESID_ROT_OPTF, "rotation": RESID_ROT_OPTF,
              "tx": RESID_TRANS_OPTF, "ty": RESID_TRANS_OPTF, "tz": RESID_TRANS_OPTF}
RESID_UNITS = {"tilt": "N.m", "list": "N.m", "rotation": "N.m",
               "tx": "N", "ty": "N", "tz": "N"}
SAT_THRESHOLD = 0.99 # railed muscles

def peak_grf():
    t = osim.TimeSeriesTable(GRF_MOT)
    vy = np.zeros(t.getNumRows())
    for L in t.getColumnLabels():
        if L.endswith("_ground_force_vy"):
            vy = vy + t.getDependentColumn(L).to_numpy()
    return float(np.max(np.abs(vy)))

def com_height():
    m = osim.Model(solve_base("davis"))
    s = m.initSystem()
    m.realizePosition(s)
    return float(m.calcMassCenterPosition(s).get(1))

def cropped(track, cycle):
    return os.path.join(RESULTS, track, f"cycle{cycle}_compliant_CROPPED_solution.sto")

def main():
    grf = peak_grf()
    comh = com_height()
    force_lim = 0.05 * grf # Hicks: residual force < 5% peak external force
    moment_lim = 0.01 * comh * grf # Hicks: < 1% of COM height x peak GRF
    print(f"Derived thresholds from this trial:\n"
          f"  peak vertical GRF        {grf:.0f} N\n"
          f"  model COM height         {comh:.3f} m\n"
          f"  residual FORCE limit     {force_lim:.1f} N     (5% of peak GRF)\n"
          f"  residual MOMENT limit    {moment_lim:.1f} N.m  (1% of COMh x peak GRF)\n")

    rows, sat_rows = [], []
    for track in TRACKS:
        peak_act = {}
        sat_frac = {}
        for cycle in CYCLES:
            path = cropped(track, cycle)
            if not os.path.exists(path):
                print(f"  [{track}] cycle{cycle}: missing -- skipped")
                continue
            tbl = osim.TimeSeriesTable(path)
            labels = list(tbl.getColumnLabels())

            resid = {}
            for k, optf in RESID_OPTF.items():
                L = f"/forceset/residual_jointset_ground_pelvis_pelvis_{k}"
                resid[k] = float(np.max(np.abs(tbl.getDependentColumn(L).to_numpy()))) * optf

            reserves = {L.split("reserve_jointset_")[1]:
                        float(np.max(np.abs(tbl.getDependentColumn(L).to_numpy()))) * RESERVE_OPTF
                        for L in labels if "reserve_jointset_" in L}
            worst_res = max(reserves, key=reserves.get)

            forces_ok = all(resid[k] < force_lim for k in ("tx", "ty", "tz"))
            moments_ok = all(resid[k] < moment_lim for k in ("tilt", "list", "rotation"))
            rows.append(dict(
                track=track, cycle=cycle,
                **{f"resid_{k}_{RESID_UNITS[k]}": round(resid[k], 2) for k in RESID_OPTF},
                peak_reserve_Nm=round(reserves[worst_res], 3),
                peak_reserve_actuator=worst_res,
                residual_forces_pass=forces_ok, residual_moments_pass=moments_ok,
                force_limit_N=round(force_lim, 1), moment_limit_Nm=round(moment_lim, 1)))

            for L in labels:
                if not L.endswith("/activation"):
                    continue
                m = L.split("/")[-2]
                a = tbl.getDependentColumn(L).to_numpy()
                peak_act[m] = max(peak_act.get(m, 0.0), float(np.max(a)))
                sat_frac[m] = max(sat_frac.get(m, 0.0),
                                  float(np.mean(a >= SAT_THRESHOLD)))
        for m in sorted(peak_act, key=lambda x: -peak_act[x]):
            sat_rows.append(dict(track=track, muscle=m,
                                 peak_activation=round(peak_act[m], 4),
                                 frac_cycle_at_ceiling=round(sat_frac[m], 4),
                                 saturated=peak_act[m] >= SAT_THRESHOLD))

    out_a = os.path.join(RESULTS, "hicks_actuator_report.csv")
    out_s = os.path.join(RESULTS, "hicks_saturation.csv")
    for path, data in ((out_a, rows), (out_s, sat_rows)):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)

    #   console summary 
    print("RESIDUALS + RESERVES over the reported stride (peak |value|)\n")
    hdr = (f"{'track':<7}{'cyc':>4}{'tilt':>8}{'list':>8}{'rot':>8}"
           f"{'tx':>7}{'ty':>7}{'tz':>7}{'reserve':>9}   flags")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        flags = ("F" if r["residual_forces_pass"] else "f") + \
                ("M" if r["residual_moments_pass"] else "m")
        print(f"{r['track']:<7}{r['cycle']:>4}"
              f"{r['resid_tilt_N.m']:>8.1f}{r['resid_list_N.m']:>8.1f}"
              f"{r['resid_rotation_N.m']:>8.1f}{r['resid_tx_N']:>7.1f}"
              f"{r['resid_ty_N']:>7.1f}{r['resid_tz_N']:>7.1f}"
              f"{r['peak_reserve_Nm']:>9.2f}   {flags}")
    print("\n  upper-case = inside the Hicks limit, lower-case = over "
          "(F/f forces, M/m moments).")
    print(f"  Residual moments are scored against the STRICT {moment_lim:.1f} N.m band; "
          "running\n  routinely exceeds it and RRA does too on this data -- see the "
          "module docstring.")

    print("\nSATURATION (peak activation >= "
          f"{SAT_THRESHOLD}, any cycle)\n")
    for track in TRACKS:
        s = [r for r in sat_rows if r["track"] == track and r["saturated"]]
        near = [r for r in sat_rows if r["track"] == track
                and not r["saturated"] and r["peak_activation"] >= 0.90]
        print(f"  {track}: {len(s)} saturated, {len(near)} more above 0.90")
        for r in s + near:
            flag = "SAT" if r["saturated"] else "   "
            print(f"      {flag} {r['muscle']:<16}{r['peak_activation']:.3f}"
                  f"   at ceiling for {r['frac_cycle_at_ceiling'] * 100:.0f}% of the cycle")

    print(f"\nsaved: {os.path.relpath(out_a, RESULTS)}\nsaved: {os.path.relpath(out_s, RESULTS)}")


if __name__ == "__main__":
    main()
