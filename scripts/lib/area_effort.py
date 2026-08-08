"""

Helper to modify MocoInverse to use muscle area proportional costs.

ESSENTIAL for fair comparisons between Rajagopal/Lai/Uhlrich and our model because we divide the glutemax into more compartments. 

Standard squared-activations is not invariant to muscle subdivion:  split one compartment into N weaker tracts (total strength preserved) and its group cost is multiplied ~N-fold, so the solver offloads it to un-fragmented antagonists. Bad! 

Also note we actually normalize by  weight_i = F0M_i / ref which keeps the objective scale around 1.0
(and notice that the standard case collapses to 1.0 for all muscles)

For cross-model comparability (davis vs raj) ref is a FIXED constant = the Rajagopal-model mean F0M

"""
import opensim as osim

# fixed reference constant = mean F0M over shared_raj_dgf.osim (80 muscles); see scratchpad/ref_f0m.py
RAJAGOPAL_MEAN_F0M = 1244.5670


def area_weight_muscle_effort(problem, model, ref=RAJAGOPAL_MEAN_F0M):
    """Weight each muscle's excitation^2 (MocoControlGoal 'excitation_effort') and activation^2
    (MocoSumSquaredStateGoal 'activation_effort') by F0M_i/ref. Coordinate actuators untouched.
    Returns a dict of counts + the weight range, for logging."""
    exc = osim.MocoControlGoal.safeDownCast(problem.updGoal("excitation_effort"))
    act = osim.MocoSumSquaredStateGoal.safeDownCast(problem.updGoal("activation_effort"))
    muscles = model.getMuscles()
    ws = []
    for i in range(muscles.getSize()):
        m = muscles.get(i)
        w = m.getMaxIsometricForce() / ref
        path = m.getAbsolutePathString()
        exc.setWeightForControl(path, w) #                muscle excitation^2  -> area-weighted
        act.setWeightForState(path + "/activation", w) # muscle activation^2  -> area-weighted
        ws.append(w)
    return dict(n_muscle=len(ws), ref=ref, w_min=min(ws), w_mean=sum(ws) / len(ws), w_max=max(ws))
