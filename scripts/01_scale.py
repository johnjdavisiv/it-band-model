"""

scale both generic models to Hamner subject01 of 72.84 kg

Two tracks:

raj - RajagopalLaiUhlrich2023.osim 
davis - the ITB-grafted model (Davis2026.osim)

We rescale by applying raj ScaleSet directly (`Model.scale()`), not by re-measuring markers

Outputs (in pipeline-work/scale/):
Rajagopal2023_subject01_scaled.osim, Davis2026_subject01_scaled.osim,
scaleSet_applied.xml, static_output.mot

"""
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.paper1io import (MODELS, TEMPLATES, WORK, STATIC_TRC, SUBJECT_MASS_KG,
                          resolve_template)  # noqa: E402

OUT = os.path.join(WORK, "scale")
RAJ_GENERIC = os.path.join(MODELS, "RajagopalLaiUhlrich2023.osim")
DAVIS_GENERIC = os.path.join(MODELS, "Davis2026.osim")
RAJ_SCALED = os.path.join(OUT, "Rajagopal2023_subject01_scaled.osim")
DAVIS_SCALED = os.path.join(OUT, "Davis2026_subject01_scaled.osim")
SCALESET = os.path.join(OUT, "scaleSet_applied.xml")

ITB_TRACTS = ["glmax12_ITB_r", "glmax34_ITB_r", "tfl12_ITB_r",
              "glmax12_ITB_l", "glmax34_ITB_l", "tfl12_ITB_l"]

def run_scale_tool():
    setup = resolve_template(
        os.path.join(TEMPLATES, "scale_setup.xml"),
        {"GENERIC_MODEL": RAJ_GENERIC,
         "STATIC_TRC": STATIC_TRC,
         "OUT_SCALESET": SCALESET,
         "OUT_STATIC_MOT": os.path.join(OUT, "static_output.mot"),
         "OUT_MODEL": RAJ_SCALED},
        os.path.join(OUT, "scale_setup_resolved.xml"))
    print(f"[scale] running ScaleTool ({os.path.basename(setup)}) ...")
    subprocess.run(["opensim-cmd", "run-tool", setup], check=True, cwd=OUT)
    assert os.path.exists(RAJ_SCALED), f"ScaleTool did not write {RAJ_SCALED}"
    assert os.path.exists(SCALESET), f"ScaleTool did not write {SCALESET}"


def body_inertials(model):
    s = model.initSystem()
    out = {}
    for b in model.getBodySet():
        com = b.get_mass_center()
        inertia = b.get_inertia()
        out[b.getName()] = (b.get_mass(),
                            tuple(round(com.get(i), 10) for i in range(3)),
                            tuple(round(inertia.get(i), 10) for i in range(6)))
    return out, model.getTotalMass(s)


def scale_davis(osim):
    """Apply the recorded ScaleSet to the ITB-grafted model. The ScaleTool<->Model.scale
    preserveMassDist flag semantics are subtle, so both values are tried and the one that
    reproduces the raj-scaled inertials is kept."""
    for preserve in (True, False):
        m = osim.Model(DAVIS_GENERIC)
        s = m.initSystem()
        ok = m.scale(s, osim.ScaleSet(SCALESET), preserve, SUBJECT_MASS_KG)
        if not ok:
            raise SystemExit(f"Model.scale returned False (preserveMassDist={preserve})")
        if gate_a1a(osim, m, verbose=False):
            print(f"[davis] preserveMassDist={preserve} reproduces the raj-scaled inertials")
            return m
    raise SystemExit("Neither preserveMassDist value reproduced the raj-scaled inertials.")


def gate_a1a(osim, davis_model, verbose=True):
    #Assert matching!!
    """Every body's mass/COM/inertia must match the raj-scaled model (the graft adds muscles,
    never bodies). Mass to 1e-6 kg; COM/inertia to 1e-4 (the ScaleSet stores factors to ~6
    significant figures while the ScaleTool applied them at full float precision)."""
    TOL_MASS, TOL_COM, TOL_I = 1e-6, 1e-4, 1e-4
    ref, ref_total = body_inertials(osim.Model(RAJ_SCALED))
    got, got_total = body_inertials(davis_model)
    ok = abs(got_total - ref_total) < TOL_MASS
    bad = []
    for name, (mass, com, inertia) in got.items():
        if name not in ref:
            continue
        rmass, rcom, rinertia = ref[name]
        if (abs(mass - rmass) > TOL_MASS
                or max(abs(a - b) for a, b in zip(com, rcom)) > TOL_COM
                or max(abs(a - b) for a, b in zip(inertia, rinertia)) > TOL_I):
            bad.append(name)
    if verbose:
        print(f"[A1a] total mass {got_total:.5f} kg (raj {ref_total:.5f}); "
              f"bodies over tolerance: {len(bad)}{' ' + str(bad[:5]) if bad else ''}")
    return ok and not bad


def muscle_blocks(path):
    txt = open(path).read()
    blocks = {}
    for mtype in ("Millard2012EquilibriumMuscle", "DeGrooteFregly2016Muscle"):
        for m in re.finditer(rf'<{mtype} name="([^"]+)">(.*?)</{mtype}>', txt, flags=re.S):
            blocks[m.group(1)] = m.group(2)
    return blocks


def strip_scaling_variables(block):
    """Remove the parts that legitimately change on scaling: geometry + Lopt/Lts values."""
    b = re.sub(r'<GeometryPath[^>]*>.*?</GeometryPath>', '<GeometryPath/>', block, flags=re.S)
    b = re.sub(r'<optimal_fiber_length>[^<]*</optimal_fiber_length>', '<optimal_fiber_length/>', b)
    b = re.sub(r'<tendon_slack_length>[^<]*</tendon_slack_length>', '<tendon_slack_length/>', b)
    return b


def gate_a1b():
    pre = muscle_blocks(DAVIS_GENERIC)
    post = muscle_blocks(DAVIS_SCALED)
    assert len(pre) == len(post), f"muscle count changed on scaling: {len(pre)} -> {len(post)}"
    changed = [name for name in pre
               if name not in post
               or strip_scaling_variables(pre[name]) != strip_scaling_variables(post[name])]
    if changed:
        raise SystemExit(f"[A1b] FAIL: {len(changed)} muscles changed beyond geometry/Lopt/Lts: "
                         f"{changed[:10]}")
    print(f"[A1b] OK: all {len(post)} muscles' non-geometry properties survived scaling "
          f"(F0M, pennation, all curves incl. the compliant ITB tendons)")


def gate_a1c(osim):
    m = osim.Model(DAVIS_SCALED)
    s = m.initSystem()
    coord = m.getCoordinateSet().get("hip_flexion_r")
    muscles = m.getMuscles()      # bind the set: .get() on a temporary returns a dangling ref
    for musc in ["glmax12_ITB_r", "tfl12_ITB_r", "glmax1_r"]:
        mm = muscles.get(musc)
        mas = []
        for deg in (0, 30, 60):
            coord.setValue(s, deg * 3.14159265 / 180.0)
            m.realizePosition(s)
            mas.append(round(mm.computeMomentArm(s, coord) * 100, 3))
        print(f"[A1c] {musc:16s} hip-flex moment arm @0/30/60 deg = {mas} cm")


def main():
    import opensim as osim
    osim.Logger.setLevelString("Error")
    os.makedirs(OUT, exist_ok=True)

    run_scale_tool()
    raj = osim.Model(RAJ_SCALED)
    total = raj.getTotalMass(raj.initSystem())
    assert abs(total - SUBJECT_MASS_KG) < 1e-4, f"raj scaled mass {total} != {SUBJECT_MASS_KG}"
    print(f"[scale] raj-scaled model OK: total mass {total:.3f} kg -> {RAJ_SCALED}")

    davis = scale_davis(osim)
    davis.printToXML(DAVIS_SCALED)
    gate_a1a(osim, osim.Model(DAVIS_SCALED))
    gate_a1b()
    gate_a1c(osim)
    print(f"[scale] davis-scaled model OK -> {DAVIS_SCALED}")


if __name__ == "__main__":
    main()
