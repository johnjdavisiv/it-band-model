"""

RRA iterations

"""
from __future__ import annotations

import os
import re
from collections import defaultdict

import numpy as np

from .paper1io import ICS, TEMPLATES, WORK, crop_window

# Avoiding edge effects!!
# The padded window RRA runs over is [IC_k - PAD_S, IC_{k+1} + PAD_S]; everything reported is
# cropped back to [IC_k, IC_{k+1}].
PAD_S = 0.20
DT = 0.004                 # 250 Hz prescribed-kinematics grid

# RRA solver settings
LOWPASS_HZ = 15.0
INTEGRATOR_TOL = 1e-6
CONVERGENCE_TOL = 1e-5

# Residual costs
RESID_FORCE_OPTF = 2.0
RESID_MOMENT_OPTF = 0.5

# Hicks et al. (2015) residual guidelines, derived per trial
HICKS_FORCE_FRAC = 0.05 # pct of peak vertical GRF
HICKS_MOMENT_FRAC = 0.01 # pct of (COM height x peak vertical GRF)

# Weld coordinates we dont' care about 
WRIST_ACTUATORS = ["wrist_flex_r", "wrist_flex_l", "wrist_dev_r", "wrist_dev_l"]

RRA_WORK = os.path.join(WORK, "rra")
RRA_TEMPLATES = os.path.join(RRA_WORK, "templates")
RRA_LOGS = os.path.join(RRA_WORK, "logs")

TASKS_TEMPLATE = os.path.join(TEMPLATES, "rra_tasks_template.xml")
ACTUATORS_TEMPLATE = os.path.join(TEMPLATES, "rra_actuators_template.xml")


def window(cycle):
    """The padded RRA window [IC_k - PAD_S, IC_{k+1} + PAD_S] for cycle 1..5."""
    return round(ICS[cycle - 1] - PAD_S, 4), round(ICS[cycle] + PAD_S, 4)


def ensure_dirs():
    for d in (RRA_WORK, RRA_TEMPLATES, RRA_LOGS):
        os.makedirs(d, exist_ok=True)

TASK_WEIGHTS = {
    "pelvis_tx": 100, "pelvis_ty": 200, "pelvis_tz": 100,
    "pelvis_tilt": 100, "pelvis_list": 100, "pelvis_rotation": 100,
    "hip_flexion_r": 50, "hip_adduction_r": 25, "hip_rotation_r": 25,
    "hip_flexion_l": 50, "hip_adduction_l": 25, "hip_rotation_l": 25,
    "knee_angle_r": 50, "ankle_angle_r": 50,
    "knee_angle_l": 50, "ankle_angle_l": 50,
    "lumbar_extension": 50, "lumbar_bending": 50, "lumbar_rotation": 25,
    "arm_flex_r": 1, "arm_add_r": 1, "arm_rot_r": 1, "elbow_flex_r": 1, "pro_sup_r": 1,
    "arm_flex_l": 1, "arm_add_l": 1, "arm_rot_l": 1, "elbow_flex_l": 1, "pro_sup_l": 1,
}

TASKS_OUT = os.path.join(RRA_TEMPLATES, "rra_tasks_symmetric.xml")


def ensure_tasks():
    #Check tracking weights match! 
    ensure_dirs()
    xml = open(TASKS_TEMPLATE).read()
    for coord, w in TASK_WEIGHTS.items():
        pat = re.compile(rf'(<CMC_Joint name="{coord}">.*?<weight>)\s*[\d.\s]+?(\s*</weight>)',
                         re.S)
        xml, k = pat.subn(rf"\g<1> {w} {w} {w}\g<2>", xml)
        if k != 1:
            raise SystemExit(f"expected exactly 1 CMC_Joint for {coord}, matched {k}")
    with open(TASKS_OUT, "w") as f:
        f.write(xml)
    txt = open(TASKS_OUT).read()
    for coord, w in TASK_WEIGHTS.items():
        m = re.search(rf'<CMC_Joint name="{coord}">.*?<weight>\s*([-\d.eE+]+)', txt, re.S)
        assert m and abs(float(m.group(1)) - w) < 1e-9, f"task weight mismatch for {coord}"
    return TASKS_OUT


# COM actuators (must move after adjustment!)
def pelvis_com(model_path):
    import opensim as osim
    m = osim.Model(model_path)
    m.initSystem()
    c = m.getBodySet().get("pelvis").get_mass_center()
    return (c.get(0), c.get(1), c.get(2))


def residual_actuators(model_path, optf_force=RESID_FORCE_OPTF, optf_moment=RESID_MOMENT_OPTF):
    ensure_dirs()
    com = pelvis_com(model_path)
    xml = open(ACTUATORS_TEMPLATE).read()

    def rewrite(optf):
        def f(m):
            s = re.sub(r"<point>[^<]*</point>",
                       f"<point>{com[0]:.17g} {com[1]:.17g} {com[2]:.17g}</point>", m.group(0))
            s = re.sub(r"<optimal_force>\s*[\d.eE+-]+\s*</optimal_force>",
                       f"<optimal_force>{optf:g}</optimal_force>", s)
            return s
        return f

    xml = re.sub(r"<PointActuator name=\"[FXYZ]+\">.*?</PointActuator>",
                 rewrite(optf_force), xml, flags=re.S)
    xml = re.sub(r"<TorqueActuator name=\"[MXYZ]+\">.*?</TorqueActuator>",
                 rewrite(optf_moment), xml, flags=re.S)

    stem = os.path.splitext(os.path.basename(model_path))[0]
    # No dots in the filename stem: OpenSim treats text after the last '.' as an extension.
    tag = f"{stem}_resid_F{optf_force:g}_M{optf_moment:g}".replace(".", "p")
    out_path = os.path.join(RRA_TEMPLATES, f"_{tag}.xml")
    open(out_path, "w").write(xml)
    return out_path, com


def verify_residual_actuators(model_path, acts_path, tol_mm=1e-6):
    # Assert every PointActuator sits on the pelvis COM - easy to miss!
    com = pelvis_com(model_path)
    pts = re.findall(r"<PointActuator name=\"(F[XYZ])\">.*?<point>([^<]+)</point>",
                     open(acts_path).read(), flags=re.S)
    if len(pts) != 3:
        raise SystemExit(f"{acts_path}: expected 3 PointActuators, found {len(pts)}")
    for name, p in pts:
        v = [float(x) for x in p.split()]
        d = max(abs(a - b) for a, b in zip(v, com)) * 1000
        if d > tol_mm:
            raise SystemExit(f"{acts_path}: {name} is {d:.4f} mm off the pelvis COM")
    return com


# automatic log parsing
MASS_ROW = re.compile(r"\*\s*([A-Za-z0-9_]+):\s*orig mass\s*=\s*([-\d.eE+]+),"
                      r"\s*new mass\s*=\s*([-\d.eE+]+)")
TOTAL_CHANGE = re.compile(r"Total mass change:\s*([-\d.eE+]+)")

# lol
SETUP = """<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="30000">
\t<RRATool name="{name}">
\t\t<model_file>{model}</model_file>
\t\t<replace_force_set>true</replace_force_set>
\t\t<force_set_files>{actuators}</force_set_files>
\t\t<results_directory>{results}</results_directory>
\t\t<output_precision>20</output_precision>
\t\t<initial_time>{t0}</initial_time>
\t\t<final_time>{t1}</final_time>
\t\t<solve_for_equilibrium_for_auxiliary_states>false</solve_for_equilibrium_for_auxiliary_states>
\t\t<maximum_number_of_integrator_steps>20000</maximum_number_of_integrator_steps>
\t\t<maximum_integrator_step_size>1</maximum_integrator_step_size>
\t\t<minimum_integrator_step_size>1e-008</minimum_integrator_step_size>
\t\t<integrator_error_tolerance>{itol}</integrator_error_tolerance>
\t\t<AnalysisSet name="Analyses"><objects /><groups /></AnalysisSet>
\t\t<ControllerSet name="Controllers"><objects /><groups /></ControllerSet>
\t\t<external_loads_file>{extloads}</external_loads_file>
\t\t<desired_points_file />
\t\t<desired_kinematics_file>{ik}</desired_kinematics_file>
\t\t<task_set_file>{tasks}</task_set_file>
\t\t<constraints_file />
\t\t<lowpass_cutoff_frequency>{lowpass}</lowpass_cutoff_frequency>
\t\t<optimizer_algorithm>ipopt</optimizer_algorithm>
\t\t<numerical_derivative_step_size>0.0001</numerical_derivative_step_size>
\t\t<optimization_convergence_tolerance>{ctol}</optimization_convergence_tolerance>
\t\t<adjust_com_to_reduce_residuals>{adjust_com}</adjust_com_to_reduce_residuals>
\t\t<initial_time_for_com_adjustment>-1</initial_time_for_com_adjustment>
\t\t<final_time_for_com_adjustment>-1</final_time_for_com_adjustment>
\t\t<adjusted_com_body>torso</adjusted_com_body>
\t\t<output_model_file>{out_model}</output_model_file>
\t\t<use_verbose_printing>false</use_verbose_printing>
\t</RRATool>
</OpenSimDocument>
"""


def run_pass(tag, model_path, t0, t1, extloads, ik, adjust_com, tasks, actuators, root):
    # do one RRA pass and rreturn (results_dir, log_path, adjusted_model_path)
    import opensim as osim
    ensure_dirs()
    results_dir = os.path.join(root, tag)
    os.makedirs(results_dir, exist_ok=True)
    setup = os.path.join(root, f"{tag}_setup.xml")
    out_model = os.path.join(root, f"{tag}_adjusted.osim")
    log = os.path.join(RRA_LOGS, f"{tag}.log")
    with open(setup, "w") as fh:
        fh.write(SETUP.format(name=tag, model=model_path, actuators=actuators,
                              results=results_dir, t0=t0, t1=t1, extloads=extloads, ik=ik,
                              tasks=tasks, adjust_com=str(adjust_com).lower(),
                              out_model=out_model, lowpass=LOWPASS_HZ,
                              itol=INTEGRATOR_TOL, ctol=CONVERGENCE_TOL))
    if os.path.exists(log):
        os.remove(log)
    osim.Logger.removeFileSink()
    osim.Logger.addFileSink(log)
    # RRA prints its mass table through the OpenSim logger at info-level, fetch it (need for adjustment)
    osim.Logger.setLevelString("Info")
    try:
        ok = osim.RRATool(setup).run()
    finally:
        osim.Logger.setLevelString("Error")
        osim.Logger.removeFileSink()
    if not ok:
        raise SystemExit(f"{tag}: RRA failed (setup {setup})")
    return results_dir, log, out_model


def mass_table(log_path):
    txt = open(log_path, errors="replace").read()
    return {b: (float(o), float(n)) for b, o, n in MASS_ROW.findall(txt)}


def mass_recommendation(log_path):
    #What does RRA recommend for body mass?
    tot = TOTAL_CHANGE.findall(open(log_path, errors="replace").read())
    return float(tot[-1]) if tot else float("nan")


def torso_com(model_path):
    import opensim as osim
    m = osim.Model(model_path)
    m.initSystem()
    c = m.getBodySet().get("torso").get_mass_center()
    return np.array([c.get(0), c.get(1), c.get(2)])


def kinematics_q(results_dir):
    hits = [f for f in os.listdir(results_dir) if f.endswith("_Kinematics_q.sto")]
    if not hits:
        raise SystemExit(f"{results_dir}: no *_Kinematics_q.sto produced")
    return os.path.join(results_dir, hits[0])


# Actually apply RRA recommendations 
def strip_wrist_actuators(model):
    fs = model.updForceSet()
    for nm in WRIST_ACTUATORS:
        i = fs.getIndex(nm)
        if i >= 0:
            fs.remove(i)
    return model


def apply_mean_masses_and_com(base_model, tables, dcom, out_path):
    # NOn-default! Apply the MEAN per-body mass table and the MEAN torso-COM shift to `base_model`.
    #Inertia scales by the same ratio as mass so the radius of gyration is unchanged. Order mattters!
    import opensim as osim
    m = osim.Model(base_model)
    s = m.initSystem()
    before = m.getTotalMass(s)

    t = m.updBodySet().get("torso")
    c0 = t.get_mass_center()
    t.set_mass_center(osim.Vec3(c0.get(0) + float(dcom[0]),
                                c0.get(1) + float(dcom[1]),
                                c0.get(2) + float(dcom[2])))

    new_by_body = defaultdict(list)
    for tab in tables:
        for b, (_orig, new) in tab.items():
            new_by_body[b].append(new)
    for b in m.updBodySet():
        if b.getName() not in new_by_body:
            continue
        new = float(np.mean(new_by_body[b.getName()]))
        ratio = new / b.getMass()
        b.set_mass(new)
        inertia = b.get_inertia()
        b.set_inertia(osim.Vec6(*[inertia.get(i) * ratio for i in range(6)]))

    strip_wrist_actuators(m)
    m.finalizeConnections()

    # getTotalMass(state) must be called with a state from THIS model instance.
    m2 = osim.Model(m)
    s2 = m2.initSystem()
    after = m2.getTotalMass(s2)
    m2.printToXML(out_path)
    return before, after


# Prescribed kinematics
def build_cycle_kinematics(cycle, pass2_dir, conv_model, out_dir):
    # We resamnple over RRA's fine integratorpass, to 250 Hz, and convert deg to rad (once!!)
    # NOTE: 
    """
    Motion files are written in RADIANS with `inDegrees=no` so nothing downstream converts again:
    OpenSim's deg->rad conversion is applied per coordinate BY MOTION TYPE and converts only
    `Rotational` ones -- `knee_angle_*_beta` drives both a rotation and a translation of its
    patellofemoral joint, is classified `Coupled`, and would pass through RAW, throwing the
    patella metres from the knee while the rest of the pose looks fine."""

    import opensim as osim
    src = kinematics_q(pass2_dir)
    out = os.path.join(out_dir, f"cycle{cycle}_q_uniform250.sto")

    sto = osim.Storage(src)
    sto.resampleLinear(DT)
    conv_model.getSimbodyEngine().convertDegreesToRadians(sto)
    sto.setInDegrees(False)
    sto.setName(f"Run_30002_cycle{cycle}_RRA_q_uniform250_rad")
    sto.printToFile(out, "w")

    tab = osim.TimeSeriesTable(out)
    t = np.array(tab.getIndependentColumn())
    dt = np.diff(t)
    assert tab.getTableMetaDataAsString("inDegrees") == "no", f"cycle{cycle}: inDegrees != no"
    assert np.allclose(dt, dt[0], atol=1e-6), f"cycle{cycle}: non-uniform grid"
    assert abs(1.0 / dt[0] - 250.0) < 1.0, f"cycle{cycle}: {1.0 / dt[0]:.1f} Hz, expected 250"
    # A running hip flexion in radians is ~ 1; in degrees it would be ~ 30. This catches a
    # double conversion or a missed one, which the file format itself cannot express.
    hip = tab.getDependentColumn("hip_flexion_r").to_numpy()
    assert np.max(np.abs(hip)) < 3.5, f"cycle{cycle}: hip_flexion_r looks like DEGREES"

    print(f"  cycle{cycle}: {tab.getNumRows()} rows  t=[{t[0]:.4f},{t[-1]:.4f}]  "
          f"dt={dt.mean():.5f}  hip_flexion_r[0]={hip[0]:.4f} rad")
    return out


def verify_against_crop(cycle, path):
    # Checjk to meke sure pad containts the stride we want!
    import opensim as osim
    tab = osim.TimeSeriesTable(path)
    t = np.array(tab.getIndependentColumn())
    c0, c1 = crop_window(cycle)
    if not (t[0] <= c0 and c1 <= t[-1]):
        raise SystemExit(f"cycle{cycle}: crop [{c0}, {c1}] is not inside window "
                         f"[{t[0]:.4f}, {t[-1]:.4f}]")
    return c0, c1
