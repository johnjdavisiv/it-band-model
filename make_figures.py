"""make_figures.py 

one simple check plot per publication figure  
note these are NOT 1:1 verbatim copies of the publication figures; axis limits and styling differ

Not made here:
Fig 1 is a model schematic
Fig 8's counterpart is validation/emg/compare_emg_moco.py -- it needs the optional raw-EMG archive

"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import opensim as osim

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from lib.paper1io import (RESULTS, PLOTS, MODELS, ICS, TOE, CYCLES, TRACKS,
                          crop_window, body_weight_N, solve_base, GRF_MOT)
from lib.registration import register
from lib.solve_model import RESID_ROT_OPTF, RESID_TRANS_OPTF, RESERVE_OPTF

osim.Logger.setLevelString("Error")

AGG = os.path.join(RESULTS, "agg")
VAL = os.path.join(ROOT, "validation")
ENG_REF = os.path.join(VAL, "tendon-curve", "eng-reference")
MA_DIR = os.path.join(VAL, "moment-arm")
DIGI = os.path.join(MA_DIR, "eng-curve-digitization")

DAVIS_C, RAJ_C = "C3", "0.4" # our model red, Rajagopal grey -- every overlay figure
NPT = 101
TOE_PCT = float(np.mean([(TOE[c] - ICS[c]) / (ICS[c + 1] - ICS[c])
                         for c in range(len(TOE))]) * 100)

ITB = ["tfl12_ITB_r", "glmax12_ITB_r", "glmax34_ITB_r"]
FEMORAL = ["glmax1_r", "glmax2_r", "glmax3_r", "glmax4_r"]
LABEL = {"tfl12_ITB_r": "TFL-ITB", "glmax12_ITB_r": "GMax1-2-ITB",
         "glmax34_ITB_r": "GMax3-4-ITB", "glmax1_r": "GMax1", "glmax2_r": "GMax2",
         "glmax3_r": "GMax3", "glmax4_r": "GMax4"}

DGF_MODEL = {"davis": "Davis2026_subject01_DGF.osim",
             "raj": "Rajagopal2023_subject01_DGF.osim"}

_KEEP = []  # pin SWIG-wrapped OpenSim objects; a collected Model dangles its muscles


def save(fig, name):
    out = os.path.join(PLOTS, name)
    fig.tight_layout(rect=(0, 0, 1, 0.955))   # keep the suptitle off the top panels
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  wrote plots/{name}")

def missing(*paths):
    return [p for p in paths if not os.path.exists(p)]


def skip(name, gone, hint):
    print(f"  SKIP {name}: missing {', '.join(os.path.relpath(p, ROOT) for p in gone)}"
          f"  ({hint})")

def read_sto(path):
    with open(path) as fh:
        for i, line in enumerate(fh):
            if line.strip().lower() == "endheader":
                return pd.read_csv(path, sep="\t", skiprows=i + 1)
    raise SystemExit(f"no endheader in {path}")


def solution(track, cycle, cropped=False):
    tag = "_CROPPED" if cropped else ""
    return os.path.join(RESULTS, track, f"cycle{cycle}_compliant{tag}_solution.sto")


def load_model(name):
    m = osim.Model(os.path.join(MODELS, name))
    m.initSystem()
    _KEEP.append(m)
    return m


def get_muscle(model, name):
    ms = model.getMuscles()      # bind the set; get() on a temporary dangles
    _KEEP.append(ms)
    return ms.get(name)


def fn(curve, xs):
    """Sample an OpenSim Function-based curve."""
    return np.array([curve.calcValue(osim.Vector(1, float(x))) for x in xs])


def eng_spline(csv_path, xcol, ycol, subset=None):
    """SimmSpline through Eng's SIMM control points (natural cubic, as SIMM defined it).

    Returns (points_x, points_y, spline). Built from ALL control points -- including flat
    pads and extrapolation tails -- because those set the spline's shape; sample only the
    window you plot."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if subset is not None:
        df = df[df.muscle == subset]
    spl = osim.SimmSpline()
    for x, y in zip(df[xcol], df[ycol]):
        spl.addPoint(float(x), float(y))
    _KEEP.append(spl)
    return df[xcol].to_numpy(float), df[ycol].to_numpy(float), spl


def stock_millard():
    """A bare Millard2012EquilibriumMuscle: OpenSim's default curves (no shipped model
    still carries the stock ACTIVE force-length curve -- the Rajagopal lineage widens it)."""
    m = osim.Model()
    body = osim.Body("b", 1.0, osim.Vec3(0), osim.Inertia(0.1, 0.1, 0.1, 0, 0, 0))
    m.addBody(body)
    m.addJoint(osim.SliderJoint("j", m.getGround(), osim.Vec3(0), osim.Vec3(0),
                                body, osim.Vec3(0), osim.Vec3(0)))
    mus = osim.Millard2012EquilibriumMuscle("probe", 1000.0, 0.10, 0.20, 0.0)
    mus.addNewPathPoint("p1", m.getGround(), osim.Vec3(0, 0, 0))
    mus.addNewPathPoint("p2", body, osim.Vec3(0, 0, 0))
    m.addForce(mus)
    m.finalizeConnections()
    m.initSystem()
    _KEEP.extend([m, mus])
    return mus


def band(ax, x_lo, y_lo, x_up, y_up, **kw):
    """Shaded envelope between two digitized bound curves (interpolated to a common x)."""
    x = np.linspace(max(x_lo.min(), x_up.min()), min(x_lo.max(), x_up.max()), 100)
    ax.fill_between(x, np.interp(x, x_lo, y_lo), np.interp(x, x_up, y_up), **kw)


def agg_curves(df, key, value):
    """{key: (pct_grid, array(ncycles, npct))} from a registered results/agg table."""
    out = {}
    for kv, sub in df.groupby(key):
        piv = sub.pivot_table(index="cycle", columns="pct", values=value)
        out[kv] = (piv.columns.to_numpy(float), piv.to_numpy(float))
    return out


def spaghetti(ax, pct, stacked, color, mean_color=None, lw=2.2):
    for k in range(stacked.shape[0]):
        ax.plot(pct, stacked[k], color=color, lw=0.6, alpha=0.35)
    ax.plot(pct, stacked.mean(0), color=mean_color or color, lw=lw)


# ==============================================================================
# F02 -- generic muscle/tendon characteristic curves (2 x 2)
# ==============================================================================
def check_F02():
    gone = missing(os.path.join(ENG_REF, "eng_active_force_length_curve_spline_points.csv"))
    if gone:
        return skip("F02", gone, "Eng curve points should ship with the repo")
    davis = load_model("Davis2026.osim")
    dgfm = load_model("Davis2026_DGF.osim")
    mil = osim.Millard2012EquilibriumMuscle.safeDownCast(get_muscle(davis, "tfl12_ITB_r"))
    dg = osim.DeGrooteFregly2016Muscle.safeDownCast(get_muscle(dgfm, "tfl12_ITB_r"))
    # the generic TENDON curve must come from a non-ITB muscle (the ITB tracts override it)
    mil_ten = osim.Millard2012EquilibriumMuscle.safeDownCast(get_muscle(davis, "glmax1_r"))
    dg_ten = osim.DeGrooteFregly2016Muscle.safeDownCast(get_muscle(dgfm, "glmax1_r"))
    stock = stock_millard()

    x_afl = np.linspace(0.3, 1.8, 301)
    x_fv = np.linspace(-1.0, 1.0, 301)
    x_pfl = np.linspace(0.9, 1.9, 301)
    s_ten = np.linspace(0.0, 0.08, 301)          # tendon strain (fraction); plotted as %

    panels = [
        ("active force-length", "norm. fiber length", x_afl, 1.0,
         fn(stock.getActiveForceLengthCurve(), x_afl),
         fn(mil.getActiveForceLengthCurve(), x_afl),
         np.array([dg.calcActiveForceLengthMultiplier(float(v)) for v in x_afl]),
         "eng_active_force_length_curve_spline_points.csv", "normalized_length"),
        ("force-velocity", "norm. fiber velocity", x_fv, 1.0,
         fn(stock.getForceVelocityCurve(), x_fv),
         fn(mil.getForceVelocityCurve(), x_fv),
         np.array([dg.calcForceVelocityMultiplier(float(v)) for v in x_fv]),
         "eng_force_velocity_curve_spline_points.csv", "normalized_velocity"),
        ("passive fiber force-length", "norm. fiber length", x_pfl, 1.0,
         fn(stock.getFiberForceLengthCurve(), x_pfl),
         fn(mil.getFiberForceLengthCurve(), x_pfl),
         np.array([dg.calcPassiveForceMultiplier(float(v)) for v in x_pfl]),
         "eng_passive_fiber_curve_spline_points.csv", "normalized_length"),
        ("tendon force-length (generic)", "tendon strain (%)", s_ten * 100.0, 100.0,
         fn(stock.getTendonForceLengthCurve(), 1.0 + s_ten),
         fn(mil_ten.getTendonForceLengthCurve(), 1.0 + s_ten),
         np.array([dg_ten.calcTendonForceMultiplier(1.0 + float(v)) for v in s_ten]),
         "eng_generic_tendon_curve_spline_points.csv", "tendon_strain"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (title, xlab, xs, xscale, y_stock, y_ours, y_dgf, pts_csv, xcol) in \
            zip(axes.ravel(), panels):
        px, py, spl = eng_spline(os.path.join(ENG_REF, pts_csv), xcol, "normalized_force")
        xs_native = xs / xscale
        ax.plot(xs, fn(spl, xs_native), color="k", lw=1.6, label="Eng 2015 (SIMM)")
        m = (px * xscale >= xs.min()) & (px * xscale <= xs.max())
        ax.plot(px[m] * xscale, py[m], "o", ms=4, mfc="w", mec="k", label="Eng control points")
        ax.plot(xs, y_stock, color="0.55", lw=1.6, label="Millard default")
        ax.plot(xs, y_ours, color=DAVIS_C, lw=2.0, label="ours (Millard)")
        ax.plot(xs, y_dgf, color=DAVIS_C, lw=1.4, ls=":", label="ours (DGF, solved)")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlab, fontsize=9)
        ax.set_ylabel("norm. force", fontsize=9)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("check F02 -- generic muscle/tendon characteristic curves")
    save(fig, "check_F02.png")


# ==============================================================================
# F03 -- ITB tract tendon force-length curves (1 x 3)
# ==============================================================================
def check_F03():
    samples = os.path.join(VAL, "tendon-curve", "tendon_curve_samples.csv")
    pts_csv = os.path.join(ENG_REF, "eng_itb_tendon_curve_spline_points.csv")
    gone = missing(samples, pts_csv)
    if gone:
        return skip("F03", gone, "run validation/tendon-curve/validate_tendon_curves.py")
    df = pd.read_csv(samples)
    pts = pd.read_csv(pts_csv, encoding="utf-8-sig")
    stock = stock_millard()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, tract in zip(axes, ["tfl12_ITB", "glmax12_ITB", "glmax34_ITB"]):
        d = df[df.tract == tract].sort_values("strain")
        s_pct = d.strain.to_numpy() * 100.0
        ax.plot(s_pct, fn(stock.getTendonForceLengthCurve(), 1.0 + d.strain.to_numpy()),
                color="0.55", lw=1.4, label="generic Millard tendon")
        ax.plot(s_pct, d.eng, color="k", lw=1.6, label="Eng 2015 (SIMM)")
        ax.plot(s_pct, d.millard, color=DAVIS_C, lw=2.0, label="ours (Millard fit)")
        ax.plot(s_pct, d.dgf, color=DAVIS_C, lw=1.4, ls=":", label="ours (DGF, solved)")
        p = pts[(pts.muscle == tract) & (pts.tendon_strain <= d.strain.max())]
        ax.plot(p.tendon_strain * 100.0, p.normalized_force, "o", ms=5, mfc="w", mec="k",
                label="Eng control points")
        ax.set_title(LABEL[tract + "_r"], fontsize=10)
        ax.set_xlabel("tendon strain (%)", fontsize=9)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("norm. tendon force", fontsize=9)
    axes[0].legend(fontsize=8)
    fig.suptitle("check F03 -- ITB tract tendon force-length curves")
    save(fig, "check_F03.png")


# ==============================================================================
# F04 -- ITB moment arms vs Eng 2015 (3 x 3: rows = DOF, columns = tract)
# ==============================================================================
def check_F04():
    cad_csv = os.path.join(DIGI, "eng-itb-moment-arms-long.csv")
    simm_csv = os.path.join(DIGI, "eng-simm-model-moment-arms-long.csv")
    swp_csv = os.path.join(MA_DIR, "model_moment_arm_sweeps_long.csv")
    gone = missing(cad_csv, simm_csv, swp_csv)
    if gone:
        return skip("F04", gone, "run validation/moment-arm/validate_moment_arms.py")
    cad = pd.read_csv(cad_csv)
    simm = pd.read_csv(simm_csv)
    swp = pd.read_csv(swp_csv)
    swp = swp[(swp.dataset == "Eng2015") & (swp.model == "step14")]

    tracts = ["tfl12_ITB", "glmax12_ITB", "glmax34_ITB"]
    dofs = ["hip_flexion", "hip_adduction", "knee_flexion"]
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    for r, dof in enumerate(dofs):
        for c, tract in enumerate(tracts):
            ax = axes[r, c]
            cp = cad[(cad.model_tract == tract) & (cad.angle_type == dof)]
            for sub in sorted(cp.muscle.unique()):    # one band per cadaveric sub-tract
                s = cp[cp.muscle == sub]
                up = s[s.bound_type == "upper"].sort_values("angle_deg")
                lo = s[s.bound_type == "lower"].sort_values("angle_deg")
                band(ax, lo.angle_deg.to_numpy(), lo.moment_arm_cm.to_numpy(),
                     up.angle_deg.to_numpy(), up.moment_arm_cm.to_numpy(),
                     facecolor="0.5", alpha=0.2, lw=0)
            sp = simm[(simm.model_tract == tract)
                      & (simm.angle_type == dof)].sort_values("grid_idx")
            ax.plot(sp.angle_deg, sp.moment_arm_cm, color="k", ls=":", lw=2.0,
                    label="Eng SIMM model")
            mp = swp[(swp.model_tract == tract) & (swp.dof == dof)].sort_values("grid_idx")
            ax.plot(mp.angle_deg, mp.moment_arm_cm, color=DAVIS_C, lw=2.2, label="ours")
            ax.axhline(0, color="0.8", lw=0.8)
            ax.grid(alpha=0.2)
            if r == 0:
                ax.set_title(LABEL[tract + "_r"], fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{dof.replace('_', ' ')}\nmoment arm (cm)", fontsize=9)
            if r == 2:
                ax.set_xlabel("angle (deg)", fontsize=9)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("check F04 -- ITB moment arms vs Eng 2015 (band = cadaver mean +/- SD)")
    save(fig, "check_F04.png")


# ==============================================================================
# F05 -- glute-max moment arms vs Blemker 2005 (1 x 3, all six glmax lines)
# ==============================================================================
def check_F05():
    blem_csv = os.path.join(DIGI, "blemker-gmax-moment-arms-long.csv")
    swp_csv = os.path.join(MA_DIR, "model_moment_arm_sweeps_long.csv")
    gone = missing(blem_csv, swp_csv)
    if gone:
        return skip("F05", gone, "run validation/moment-arm/validate_moment_arms.py")
    blem = pd.read_csv(blem_csv)
    swp = pd.read_csv(swp_csv)
    swp = swp[(swp.dataset == "Blemker2005") & (swp.model == "step14")]

    muscles = ["glmax1", "glmax2", "glmax3", "glmax4", "glmax12_ITB", "glmax34_ITB"]
    colors = {"glmax1": "#BDD7E7", "glmax2": "#6BAED6", "glmax3": "#3182BD",
              "glmax4": "#08519C", "glmax12_ITB": "#1B9E77", "glmax34_ITB": "#7570B3"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, dof in zip(axes, ["hip_flexion", "hip_adduction", "hip_rotation"]):
        bb = blem[blem.angle_type == dof]
        up = bb[bb.bound_type == "upper"].sort_values("angle_deg")
        lo = bb[bb.bound_type == "lower"].sort_values("angle_deg")
        band(ax, lo.angle_deg.to_numpy(), lo.moment_arm_cm.to_numpy(),
             up.angle_deg.to_numpy(), up.moment_arm_cm.to_numpy(),
             facecolor="0.5", alpha=0.25, lw=0, label="Blemker fiber range")
        for m in muscles:
            d = swp[(swp.model_tract == m) & (swp.dof == dof)].sort_values("grid_idx")
            ax.plot(d.angle_deg, d.moment_arm_cm, color=colors[m], lw=1.8,
                    label=LABEL[m + "_r"])
        ax.axhline(0, color="0.8", lw=0.8)
        ax.set_title(dof.replace("_", " "), fontsize=10)
        ax.set_xlabel("angle (deg)", fontsize=9)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("moment arm (cm)", fontsize=9)
    axes[2].legend(fontsize=7, loc="best")
    fig.suptitle("check F05 -- glute-max moment arms vs Blemker & Delp 2005")
    save(fig, "check_F05.png")


# ==============================================================================
# F06 -- total passive joint moment vs Silder 2007 (5 condition panels)
# ==============================================================================
def check_F06():
    tot_csv = os.path.join(VAL, "passive-moment", "passive_joint_moments_total.csv")
    gone = missing(tot_csv)
    if gone:
        return skip("F06", gone,
                    "run validation/passive-moment/passive_joint_moment_validation.py")
    df = pd.read_csv(tot_csv)
    series = [("Silder 2007 (digitized)", "k", ":"),
              ("Rajagopal2016", "#92C5DE", "-"),
              ("RajagopalLaiUhlrich2023", "#2166AC", "-"),
              ("Davis2026 (ITB model)", DAVIS_C, "-")]
    panels = [(j, c) for j in ("hip", "knee")
              for c in df[df.joint == j].condition.unique()]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey="row")
    for ax, (joint, cond) in zip(axes.ravel(), panels):
        sub = df[(df.joint == joint) & (df.condition == cond)]
        for model, color, ls in series:
            d = sub[sub.model == model].sort_values("angle_deg")
            if len(d):
                ax.plot(d.angle_deg, d.total_moment_Nm, color=color, ls=ls, lw=1.8,
                        label=model)
        ax.axhline(0, color="0.8", lw=0.8)
        ax.set_title(f"{joint}:  {cond}", fontsize=9)
        ax.set_xlabel(f"{joint} flexion angle (deg)", fontsize=9)
        ax.grid(alpha=0.25)
    for ax in axes[:, 0]:
        ax.set_ylabel("passive moment (N*m)", fontsize=9)
    axes[1, 2].axis("off")
    h, l = axes[0, 0].get_legend_handles_labels()
    axes[1, 2].legend(h, l, loc="center", fontsize=9, frameon=False)
    fig.suptitle("check F06 -- total passive joint moment vs Silder 2007")
    save(fig, "check_F06.png")


# ==============================================================================
# F07 -- ITB tract + glute-max compartment forces during running (2 x 4)
# ==============================================================================
def check_F07():
    src = os.path.join(AGG, "tracts_registered_davis.csv")
    gone = missing(src)
    if gone:
        return skip("F07", gone, "run scripts/05_aggregate.py")
    df = pd.read_csv(src)
    force = agg_curves(df, "tract", "tendon_force_BW")
    pct = force[ITB[0]][0]
    total = np.sum([force[t][1] for t in ITB], axis=0)   # peak-of-sum lives on this curve
    bw = body_weight_N("davis")

    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), sharex=True, sharey=True)
    panels = [(axes[0, i], LABEL[t], force[t][1]) for i, t in enumerate(ITB)]
    panels.append((axes[0, 3], "Total ITB (sum of 3)", total))
    panels += [(axes[1, i], LABEL[t], force[t][1]) for i, t in enumerate(FEMORAL)]
    for ax, title, stacked in panels:
        ax.axvline(TOE_PCT, color="0.6", ls=":", lw=1)
        spaghetti(ax, pct, stacked, DAVIS_C)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
        ax.set_xlim(0, 100)
    for ax in axes[1]:
        ax.set_xlabel("% gait cycle", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("tendon force (BW)", fontsize=9)
    fig.suptitle(f"check F07 -- ITB tract (top) + femoral glute-max compartment (bottom) "
                 f"force;  thin = each of {len(CYCLES)} cycles, thick = mean;  "
                 f"dotted = toe-off;  BW = {bw:.0f} N")
    save(fig, "check_F07.png")


# ==============================================================================
# S01 / S05 -- all-86-muscle grids, ours vs Rajagopal (geometry pairing)
# ==============================================================================
# davis <-> raj pairing is by GEOMETRY, not name: davis glmax4's path points are
# byte-identical to Rajagopal's glmax3, while davis glmax3 is an extra intermediate
# tract with no counterpart. A name-based join would silently compare two different
# muscles. Base-name map; None = no raj twin. Unlisted names pair with themselves.
PAIR = {"glmax3": None, "glmax4": "glmax3",
        "glmax12_ITB": None, "glmax34_ITB": None, "tfl12_ITB": "tfl"}
# panel order: A->Z per side, except the two glute-max ITB tracts move after glmax4
SORT_OVERRIDE = {"glmax12_ITB": "glmax4_zz1", "glmax34_ITB": "glmax4_zz2"}
N_EXPECT = {"davis": 86, "raj": 80}

_SOLUTION_CACHE = {}


def _solution_df(track, cycle):
    if (track, cycle) not in _SOLUTION_CACHE:
        _SOLUTION_CACHE[(track, cycle)] = read_sto(solution(track, cycle))
    return _SOLUTION_CACHE[(track, cycle)]


def muscle_names(df):
    """Positive identification: a muscle is a bare /forceset/<name> control column that
    also has an /activation state (the bare block also holds coordinate actuators)."""
    return sorted(c.split("/")[-1] for c in df.columns
                  if c.startswith("/forceset/") and c.count("/") == 2
                  and f"{c}/activation" in df.columns)


def track_series(track, kind):
    """{muscle: (pct, array(ncycles, NPT))} of excitation or tendon force (N)."""
    f0m = None
    if kind == "force":
        ms = load_model(DGF_MODEL[track]).getMuscles()
        _KEEP.append(ms)
        f0m = {ms.get(i).getName(): ms.get(i).getMaxIsometricForce()
               for i in range(ms.getSize())}
    per = {}
    names = None
    for c in CYCLES:
        df = _solution_df(track, c)
        t = df["time"].to_numpy()
        t0, t1 = crop_window(c)
        names = muscle_names(df)
        assert len(names) == N_EXPECT[track], \
            f"{track} cycle{c}: {len(names)} muscles, expected {N_EXPECT[track]}"
        for n in names:
            if kind == "excitation":
                y = df[f"/forceset/{n}"].to_numpy()
            else:   # tendon force of a compliant DGF muscle is exactly ntf * F0M
                y = df[f"/forceset/{n}/normalized_tendon_force"].to_numpy() * f0m[n]
            pct, yr = register(t, y, t0, t1, n=NPT)
            per.setdefault(n, []).append(yr)
    return {n: (pct, np.array(v)) for n, v in per.items()}


def pairing(davis_names, raj_names):
    """Ordered davis panel list + davis->raj map, with the injectivity asserts."""
    def base(n):
        assert n[-2:] in ("_r", "_l"), n
        return n[:-2], n[-1]
    davis_bases = sorted({base(n)[0] for n in davis_names},
                         key=lambda b: SORT_OVERRIDE.get(b, b))
    order, pmap = [], {}
    for side in ("r", "l"):
        for b in davis_bases:
            d = f"{b}_{side}"
            rb = PAIR.get(b, b)
            order.append(d)
            pmap[d] = f"{rb}_{side}" if rb else None
    used = [v for v in pmap.values() if v]
    assert len(used) == len(set(used)), "pairing is not injective over raj"
    orphan = set(raj_names) - set(used)
    assert not orphan, f"raj muscles with no davis panel: {sorted(orphan)}"
    return order, pmap


def grid86(davis_series, raj_series, ylabel, title, fname, sharey, ylim=None):
    order, pmap = pairing(list(davis_series), list(raj_series))
    ncol, nrow = 6, int(np.ceil(len(order) / 6))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.1 * ncol, 1.55 * nrow),
                             sharex=True, sharey=sharey)
    axes = axes.ravel()
    for k, d in enumerate(order):
        ax = axes[k]
        r = pmap[d]
        if r is not None:
            pct, st = raj_series[r]
            spaghetti(ax, pct, st, RAJ_C, mean_color="0.2", lw=1.2)
        pct, st = davis_series[d]
        spaghetti(ax, pct, st, DAVIS_C, lw=1.2)
        name = d if (r is None or r == d) else f"{d} / {r}"
        ax.set_title(name, fontsize=6.5)
        ax.tick_params(labelsize=6)
        ax.set_xlim(0, 100)
        if ylim:
            ax.set_ylim(*ylim)
    for k in range(len(order), len(axes)):
        axes[k].axis("off")
    fig.suptitle(f"{title}\nred = ours, grey = Rajagopal;  thin = each of {len(CYCLES)} "
                 f"cycles, thick = mean;  '<ours> / <raj>' marks geometry-paired names  "
                 f"({ylabel})", fontsize=10)
    save(fig, fname)


def check_S01():
    gone = missing(*[solution(t, c) for t in TRACKS for c in CYCLES])
    if gone:
        return skip("S01", gone[:1], f"+{len(gone) - 1} more; run scripts/04_moco_inverse.py")
    grid86(track_series("davis", "excitation"), track_series("raj", "excitation"),
           "excitation, shared 0-1 axis", "check S01 -- muscle excitations",
           "check_S01.png", sharey=True, ylim=(0, 1))


def check_S05():
    gone = missing(*[solution(t, c) for t in TRACKS for c in CYCLES])
    if gone:
        return skip("S05", gone[:1], f"+{len(gone) - 1} more; run scripts/04_moco_inverse.py")
    # y is FREE per panel (peaks span ~20-3000 N); panel heights are NOT comparable
    grid86(track_series("davis", "force"), track_series("raj", "force"),
           "tendon force in N, free y per panel", "check S05 -- muscle tendon forces",
           "check_S05.png", sharey=False)


# ==============================================================================
# S02 -- reserve torques + pelvis residuals vs the Hicks limits (4 x 5)
# ==============================================================================
RESERVE_COORDS = [
    ("hip_r", "hip_flexion_r"), ("hip_r", "hip_adduction_r"), ("hip_r", "hip_rotation_r"),
    ("walker_knee_r", "knee_angle_r"), ("ankle_r", "ankle_angle_r"),
    ("hip_l", "hip_flexion_l"), ("hip_l", "hip_adduction_l"), ("hip_l", "hip_rotation_l"),
    ("walker_knee_l", "knee_angle_l"), ("ankle_l", "ankle_angle_l"),
]
RESIDUALS = [("pelvis_tx", "FX", RESID_TRANS_OPTF, "N"),
             ("pelvis_ty", "FY", RESID_TRANS_OPTF, "N"),
             ("pelvis_tz", "FZ", RESID_TRANS_OPTF, "N"),
             ("pelvis_tilt", "MZ (tilt)", RESID_ROT_OPTF, "N*m"),
             ("pelvis_list", "MX (list)", RESID_ROT_OPTF, "N*m"),
             ("pelvis_rotation", "MY (rotation)", RESID_ROT_OPTF, "N*m")]


def check_S02():
    gone = missing(*[solution(t, c, cropped=True) for t in TRACKS for c in CYCLES])
    if gone:
        return skip("S02", gone[:1], f"+{len(gone) - 1} more; run scripts/04_moco_inverse.py")
    # residual limits derived exactly as scripts/06_hicks_report.py derives them
    g = read_sto(GRF_MOT)
    vy = sum(g[c].to_numpy() for c in g.columns if c.endswith("_ground_force_vy"))
    peak_grf = float(np.max(np.abs(vy)))
    m = osim.Model(solve_base("davis"))
    s = m.initSystem()
    m.realizePosition(s)
    force_lim = 0.05 * peak_grf
    moment_lim = 0.01 * float(m.calcMassCenterPosition(s).get(1)) * peak_grf

    fig, axes = plt.subplots(4, 5, figsize=(14, 9.5), sharex=True)
    # paper layout: reserves R / reserves L / residual forces / residual moments
    slot = list(range(10)) + [10, 11, 12, 15, 16, 17]
    panels = ([(axes.ravel()[i], f"reserve {c}",
                f"/forceset/reserve_jointset_{j}_{c}", RESERVE_OPTF, None)
               for i, (j, c) in enumerate(RESERVE_COORDS)]
              + [(axes.ravel()[slot[10 + i]], f"residual {lab}",
                  f"/forceset/residual_jointset_ground_pelvis_{sfx}", optf,
                  force_lim if unit == "N" else moment_lim)
                 for i, (sfx, lab, optf, unit) in enumerate(RESIDUALS)])
    cropped_df = {(tr, c): read_sto(solution(tr, c, cropped=True))
                  for tr in TRACKS for c in CYCLES}
    for ax, title, col, optf, lim in panels:
        for track, color in (("raj", RAJ_C), ("davis", DAVIS_C)):
            for c in CYCLES:
                df = cropped_df[(track, c)]
                t = df["time"].to_numpy()
                t0, t1 = crop_window(c)
                # the solver's own mesh, mapped to % of stride (no resampling: an
                # extremum between uniform grid points would be clipped off the apex)
                ax.plot(100.0 * (t - t0) / (t1 - t0), df[col].to_numpy() * optf,
                        color=color, lw=0.7, alpha=0.5)
        if lim is not None:
            ax.axhline(lim, color="k", ls="--", lw=1.0)
            ax.axhline(-lim, color="k", ls="--", lw=1.0)
        ax.set_title(title, fontsize=8.5)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
        ax.set_xlim(0, 100)
    axes.ravel()[13].axis("off")
    axes.ravel()[13].legend(
        [plt.Line2D([0], [0], color=DAVIS_C), plt.Line2D([0], [0], color=RAJ_C),
         plt.Line2D([0], [0], color="k", ls="--")],
        ["ours", "Rajagopal", "Hicks residual limit"], loc="center", fontsize=8,
        frameon=False)
    for k in (14, 18, 19):
        axes.ravel()[k].axis("off")
    for ax in axes[3, :3]:
        ax.set_xlabel("% gait cycle", fontsize=8)
    fig.suptitle("check S02 -- reserve torques (N*m) + pelvis residuals (N / N*m), "
                 "all cycles, both tracks\n(reserve panels have no line: the 5%-of-joint-"
                 "moment limit needs an inverse-dynamics denominator -- see "
                 "results/hicks_actuator_report.csv for the scored peaks)", fontsize=10)
    save(fig, "check_S02.png")


# ==============================================================================
# S03 / S04 -- active force-length + force-velocity multipliers over the cycle
# ==============================================================================
def _multiplier_grid(value, curve_fn, title, band_lohi, fname):
    src = os.path.join(AGG, "tracts_registered_davis.csv")
    gone = missing(src)
    if gone:
        return skip(title.split()[1], gone, "run scripts/05_aggregate.py")
    df = pd.read_csv(src)
    series = agg_curves(df, "tract", value)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), sharex=True, sharey=True)
    panels = [(axes[0, i], t) for i, t in enumerate(ITB)] + \
             [(axes[1, i], t) for i, t in enumerate(FEMORAL)]
    for ax, t in panels:
        pct, st = series[t]
        mult = curve_fn(st.ravel()).reshape(st.shape)
        ax.axhspan(*band_lohi, color="#7FBF7F", alpha=0.18)
        ax.axhline(1.0, color="0.5", ls=":", lw=1)
        ax.axvline(TOE_PCT, color="0.6", ls=":", lw=1)
        spaghetti(ax, pct, mult, DAVIS_C)
        ax.set_title(LABEL[t], fontsize=10)
        ax.grid(alpha=0.25)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1.55)
    axes[0, 3].axis("off")
    for ax in axes[1]:
        ax.set_xlabel("% gait cycle", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("multiplier", fontsize=9)
    fig.suptitle(title)
    save(fig, fname)


def _dgf_curves():
    dg = osim.DeGrooteFregly2016Muscle.safeDownCast(
        get_muscle(load_model("Davis2026_subject01_DGF.osim"), "tfl12_ITB_r"))
    _KEEP.append(dg)   # all 86 muscles share one active-FL and one FV curve
    afl = lambda x: np.array([dg.calcActiveForceLengthMultiplier(float(v))
                              for v in np.atleast_1d(x)])
    fvm = lambda x: np.array([dg.calcForceVelocityMultiplier(float(v))
                              for v in np.atleast_1d(x)])
    return afl, fvm


def check_S03_S04():
    afl, fvm = _dgf_curves()
    _multiplier_grid(
        "norm_fiber_length", afl,
        "check S03 -- ACTIVE FORCE-LENGTH multiplier of the grafted tracts "
        "(green band = >= 0.90 of peak isometric force)",
        (0.90, 1.0), "check_S03.png")
    _multiplier_grid(
        "norm_fiber_velocity", fvm,
        "check S04 -- FORCE-VELOCITY multiplier of the grafted tracts "
        "(> 1 = boosted while lengthening)",
        (0.90, 1.0), "check_S04.png")


# ==============================================================================
MANIFEST = """
Publication figure -> check plot (this script unless noted):
  Fig 1   model schematic               : OpenSim GUI (not scripted)
  Fig 2   generic characteristic curves : check_F02.png
  Fig 3   ITB tendon curves             : check_F03.png
  Fig 4   ITB moment arms vs Eng        : check_F04.png
  Fig 5   glute-max arms vs Blemker     : check_F05.png
  Fig 6   passive moments vs Silder     : check_F06.png
  Fig 7   ITB + compartment forces      : check_F07.png
  Fig 8   EMG vs excitation             : validation/emg/compare_emg_moco.py
  Fig S1  all 86 excitations            : check_S01.png
  Fig S2  reserves + residuals          : check_S02.png
  Fig S3  active force-length mult.     : check_S03.png
  Fig S4  force-velocity mult.          : check_S04.png
  Fig S5  all 86 tendon forces          : check_S05.png

Result tables (results/): hicks_*.csv (06), fiber_operating_length.csv (07),
itb_peak_summary.csv (08), agg/shared_muscle_drift.csv (05).
"""


def main():
    os.makedirs(PLOTS, exist_ok=True)
    check_F02()
    check_F03()
    check_F04()
    check_F05()
    check_F06()
    check_F07()
    check_S01()
    check_S02()
    check_S03_S04()
    check_S05()
    print(MANIFEST)
    print("DONE figures")


if __name__ == "__main__":
    main()
