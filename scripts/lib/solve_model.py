"""solve_model.py 

Configs for MocoInverse and model-building recipe.

"""
import os

from .paper1io import solve_base

# --- joints welded
WELD = ["mtp_r", "mtp_l", "subtalar_r", "subtalar_l", "radius_hand_r", "radius_hand_l"]

# arm + lumbar coordinate actuators
ARMLUMBAR = ["lumbar_ext", "lumbar_bend", "lumbar_rot",
             "shoulder_flex_r", "shoulder_add_r", "shoulder_rot_r", "elbow_flex_r", "pro_sup_r",
             "shoulder_flex_l", "shoulder_add_l", "shoulder_rot_l", "elbow_flex_l", "pro_sup_l"]

# Residual actuators at the pelvis
RESID_ROT_OPTF, RESID_TRANS_OPTF, RESID_CBOUND = 250.0, 50.0, 3.0
RESERVE_OPTF, RESERVE_BOUND = 1.0, 50.0
WIDTH_K = 1.34 # active force-length width restore after DGF conversion
ARM_OPTF, ARM_BOUND = 250.0, 10.0 #Arms cheap and strong
MESH = 0.025 #sets number of collocation points

# Important!! Moco has some spline artifacts at the edges of the data, so you want trim off a bit to avoid high residuals that are impossible track
TRIM_FRAMES = 5


def build_model(track, extloads):    
    # "Builds" a moco-ready model from the Millard default. 
    import opensim as osim
    mp = osim.ModelProcessor(solve_base(track))
    mp.append(osim.ModOpAddExternalLoads(extloads))
    welds = osim.StdVectorString()
    for j in WELD:
        welds.append(j)
    mp.append(osim.ModOpReplaceJointsWithWelds(welds))
    mp.append(osim.ModOpAddResiduals(RESID_ROT_OPTF, RESID_TRANS_OPTF, RESID_CBOUND))
    mp.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
    mp.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(WIDTH_K))  # DGF resets width to 1.0
    mp.append(osim.ModOpTendonComplianceDynamicsModeDGF("implicit"))
    mp.append(osim.ModOpAddReserves(RESERVE_OPTF, RESERVE_BOUND, True))
    model = mp.process()
    model.initSystem()
    fs = model.getForceSet()
    for nm in ARMLUMBAR:
        if fs.getIndex(nm) >= 0:
            ca = osim.CoordinateActuator.safeDownCast(fs.get(nm))
            if ca:
                ca.set_optimal_force(ARM_OPTF)
                ca.setMinControl(-ARM_BOUND)
                ca.setMaxControl(ARM_BOUND)
    model.initSystem()
    return model

def generic_dgf_model(src_path):
    #Converts Davis 2026 to Davis 2026 with DGF muscles (generic)
    import opensim as osim
    mp = osim.ModelProcessor(src_path)
    mp.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
    mp.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(WIDTH_K))
    mp.append(osim.ModOpTendonComplianceDynamicsModeDGF("implicit"))
    model = mp.process()
    model.initSystem()
    return model


def strip_external_loads(path):
    # Drops external load files that are "baked" into an osim file
    with open(path) as fh:
        xml = fh.read()
    i = xml.find("<ExternalLoads ")
    if i < 0:
        return False
    j = xml.find("</ExternalLoads>", i)
    assert j > i, f"{path}: unbalanced ExternalLoads block"
    start = xml.rfind("\n", 0, i) + 1          # swallow the element + indentation + newline
    end = xml.find("\n", j) + 1
    out = xml[:start] + xml[end:]
    assert "<datafile>" not in out, f"{path}: a <datafile> survived the strip"
    with open(path, "w") as fh:
        fh.write(out)
    return True
