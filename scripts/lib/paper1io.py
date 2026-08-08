"""

Shared paths, cycle geometry, solve constants, etc - helps get around absolute path requirements

    from lib.paper1io import MODELS, INPUTS, RESULTS, ICS, solve_base, resolve_external_loads
"""
import os
import re

# repo root = two levels up from this file (paper1-public-code/scripts/lib/paper1io.py)
LIB = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(LIB)
ROOT = os.path.dirname(SCRIPTS)

MODELS = os.path.join(ROOT, "models")
DATA = os.path.join(ROOT, "data")  # raw Hamner subject01 inputs (see data/README.md)
TEMPLATES = os.path.join(DATA, "templates") # Scale/IK/RRA setup templates
INPUTS = os.path.join(ROOT, "inputs") # frozen RRA kinematics + ExternalLoads
RESULTS = os.path.join(ROOT, "results")
PLOTS = os.path.join(ROOT, "plots")
WORK = os.path.join(ROOT, "pipeline-work") # intermediates from 01-03 (gitignored)

# raw data files
STATIC_TRC = os.path.join(DATA, "Static_FJC.trc")
RUN_TRC = os.path.join(DATA, "Run_30002.trc")
GRF_MOT = os.path.join(DATA, "Run_30002_GRF.mot")

SUBJECT_MASS_KG = 72.840 # subject01 mass, used by the ScaleTool

# --- cycle geometry (subject01 Run_30002, 3 m/s) -----------------------------------------------
# Hardcoded here but came from GRF detection at vGRF > 10 N
ICS = [0.515, 1.232, 1.955, 2.675, 3.382, 4.098]
TOE = [0.805, 1.521, 2.240, 2.955, 3.672]
CYCLES = (1, 2, 3, 4, 5)
TRACKS = ("davis", "raj")

# Millard solve-input models (subject01-scaled + RRA-mass-adjusted + wrist-actuators removed)
_SOLVE_BASE = {
    "davis": "Davis2026_subject01.osim",
    "raj": "Rajagopal2023_subject01.osim",
}

def solve_base(track):
    """Absolute path to the Millard solve-input model for a track ('davis'|'raj')."""
    return os.path.join(MODELS, _SOLVE_BASE[track])

# --- BW normalization ----
GRAVITY = 9.80665          # m/s^2, standard gravity (OpenSim's default)
_BW_CACHE = {}

def body_mass_kg(track="davis"):
    if track not in _BW_CACHE:
        import opensim as osim          # local: keeps this module importable without OpenSim
        m = osim.Model(solve_base(track))
        s = m.initSystem()
        _BW_CACHE[track] = float(m.getTotalMass(s))   # `s` must come from THIS model instance
    return _BW_CACHE[track]


def body_weight_N(track="davis"):
    return body_mass_kg(track) * GRAVITY


def assert_tracks_same_mass(tol=1e-6):
    a, b = body_mass_kg("davis"), body_mass_kg("raj")
    assert abs(a - b) < tol, (f"davis and raj solve models differ in total mass "
                              f"({a:.6f} vs {b:.6f} kg) -- BW normalization is ambiguous")
    return a

def kinematics(cycle):
    return os.path.join(INPUTS, f"cycle{cycle}_q_uniform250.sto")

def crop_window(cycle):
    return ICS[cycle - 1], ICS[cycle]

def resolve_template(template_path, substitutions, out_path):
    # Materialize a setup XML from a template by replacing @TOKEN@ placeholders with absolute paths (rel paths are a disaster in osim)
    with open(template_path) as fh:
        xml = fh.read()
    for token, value in substitutions.items():
        needle = f"@{token}@"
        assert needle in xml, f"{os.path.basename(template_path)} has no {needle} placeholder"
        xml = xml.replace(needle, value)
    leftover = re.findall(r"@[A-Z_]+@", xml)
    assert not leftover, f"unsubstituted placeholders remain in {template_path}: {leftover}"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(xml)
    return out_path


def resolve_external_loads(out_dir):
    #Sets absolute external load xmls
    template = os.path.join(INPUTS, "Run_30002_ExternalLoads.xml")
    if not os.path.exists(GRF_MOT):
        raise FileNotFoundError(f"GRF missing: {GRF_MOT}")
    with open(template) as fh:
        xml = fh.read()
    xml = xml.replace("<datafile>Run_30002_GRF.mot</datafile>",
                      f"<datafile>{GRF_MOT}</datafile>")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "Run_30002_ExternalLoads_resolved.xml")
    with open(out, "w") as fh:
        fh.write(xml)
    return out
