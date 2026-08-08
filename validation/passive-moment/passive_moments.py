"""
Passive lower-extremity joint moments from Silder 
"""

import numpy as np

_D2R = np.pi / 180.0

DEFAULT_PARAMS = {
    'beta_HF': 5.1, 'alpha_HF': 19.5,     # hip flexors  (uni)
    'beta_HE': 2.0, 'alpha_HE': 27.3,     # hip extensors (uni)
    'beta_KF': 5.8, 'alpha_KF': 12.8,     # knee flexors (uni)
    'beta_KE': 3.5, 'alpha_KE': 101.9,    # knee extensors (uni)
    'beta_PF': 4.9, 'alpha_PF': -3.4,     # ankle plantarflexors (uni)
    'beta_RF_prox': 3.1, 'beta_RF_dist': 1.9, 'alpha_RF': 24.4,   # rectus femoris  (hip/knee)
    'beta_HAM_prox': 5.1, 'beta_HAM_dist': 3.9, 'alpha_HAM': 30.8, # hamstrings      (hip/knee)
    'beta_GAS_prox': 4.5, 'beta_GAS_dist': 4.7, 'alpha_GAS': 26.0, # gastrocnemius   (knee/ankle)
}

PARAM_SD = {
    'beta_HF': 0.9, 'alpha_HF': 10.0,
    'beta_HE': 0.9, 'alpha_HE': 18.0,
    'beta_KF': 4.2, 'alpha_KF': 13.1,
    'beta_KE': 1.7, 'alpha_KE': 11.9,
    'beta_PF': 1.5, 'alpha_PF': 5.0,
    'beta_RF_prox': 1.4, 'beta_RF_dist': 0.7, 'alpha_RF': 9.0,
    'beta_HAM_prox': 2.0, 'beta_HAM_dist': 2.5, 'alpha_HAM': 14.9,
    'beta_GAS_prox': 1.6, 'beta_GAS_dist': 3.4, 'alpha_GAS': 28.9,
}

_JOINTS = ('hip', 'knee', 'ankle')

MEASURED_RANGE_DEG = {
    'hip':   (-15.0, 100.0),
    'knee':  (0.0, 135.0),
    'ankle': (-20.0, 20.0),
}

_TRIAL_ENDPOINTS_DEG = [
    ((-15, 0, 20), (0, 60, 20)),     # 1  RF, HF
    ((-15, 0, 20), (0, 10, 20)),     # 2  RF, HF
    ((-15, 0, 20), (0, 135, 20)),    # 3  RF, HF
    ((0, 0, 20),   (0, 135, 20)),    # 4  RF, HF
    ((0, 0, 20),   (30, 35, 20)),    # 5  HAM, HE
    ((0, 0, 20),   (30, 135, 20)),   # 6  RF, HF, HE
    ((0, 0, 20),   (60, 35, 20)),    # 7  HAM, HE
    ((0, 0, 20),   (60, 135, 20)),   # 8  RF, HF, HE
    ((0, 0, 20),   (85, 35, 20)),    # 9  HAM, HE
    ((0, 0, 20),   (85, 135, 20)),   # 10 HAM, RF, HE
    ((15, 0, 20),  (15, 0, -20)),    # 11 GAS, PF
    ((15, 0, 20),  (15, 60, -20)),   # 12 GAS, PF
    ((15, 0, 20),  (15, 135, -20)),  # 13 GAS, PF
    ((-15, 0, 20), (0, 25, 20)),     # 14 HF
    ((0, 0, 20),   (100, 25, 20)),   # 15 HE
]
_ENDPTS = np.array([p for seg in _TRIAL_ENDPOINTS_DEG for p in seg], float)


def _convex_hull(points):
    """CCW convex hull via monotone chain (pure numpy, so the core needs no SciPy)."""
    pts = np.unique(np.asarray(points, float), axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and cross(h[-2], h[-1], p) <= 0:
                h.pop()
            h.append(p)
        return h[:-1]

    return np.array(half(pts) + half(pts[::-1]))


def _in_hull_2d(P, hull, tol=1e-9):
    """Boolean mask: which rows of P (N, 2) lie inside/on the CCW convex `hull`."""
    P = np.asarray(P, float)
    inside = np.ones(len(P), bool)
    m = len(hull)
    for i in range(m):
        a, b = hull[i], hull[(i + 1) % m]
        cross = (b[0] - a[0]) * (P[:, 1] - a[1]) - (b[1] - a[1]) * (P[:, 0] - a[0])
        inside &= cross >= -tol
    return inside


# Measured envelopes, one per muscle-pair, built once at import. RF/HAM and the
# uni-articular hip/knee terms are constrained by the (hip, knee) cloud; GAS and
# the ankle terms by the (knee, ankle) cloud.
_HULL_HK = _convex_hull(_ENDPTS[:, [0, 1]])   # (hip, knee)
_HULL_KA = _convex_hull(_ENDPTS[:, [1, 2]])   # (knee, ankle)


def in_measured_envelope(hip=0.0, knee=0.0, ankle=0.0, joint='all', degrees=True):
    """Boolean mask of whether angle combinations fall within the measured envelope.

    A moment is 'supported' only where every joint-pair its muscles span lies
    inside the corresponding convex hull of the Table-2 configurations:
      * hip moment   -> (hip, knee) inside the hip-knee hull
      * knee moment  -> (hip, knee) AND (knee, ankle) hulls
      * ankle moment -> (knee, ankle) hull
    Inputs broadcast together. Returns a bool array (broadcast shape) for a
    single joint, or a dict of them for joint='all'. This is the geometric test
    that `passive_moment(..., extrapolate=False)` uses to decide what to mask.
    """
    h, k, a = np.broadcast_arrays(np.asarray(hip, float),
                                  np.asarray(knee, float),
                                  np.asarray(ankle, float))
    if not degrees:
        h, k, a = np.rad2deg(h), np.rad2deg(k), np.rad2deg(a)
    shape = h.shape
    in_hk = _in_hull_2d(np.column_stack([h.ravel(), k.ravel()]), _HULL_HK).reshape(shape)
    in_ka = _in_hull_2d(np.column_stack([k.ravel(), a.ravel()]), _HULL_KA).reshape(shape)
    masks = {'hip': in_hk, 'knee': in_hk & in_ka, 'ankle': in_ka}
    if joint == 'all':
        return masks
    if joint not in masks:
        raise ValueError(f"joint must be one of {_JOINTS} or 'all', got {joint!r}")
    return masks[joint]


def _assemble(hip, knee, ankle, p, gastroc_sign):
    """Return (M_hip, M_knee, M_ankle) for angles already in RADIANS.

    `p` is a params dict; every value may be a scalar or an array that broadcasts against the
    angles (this is what lets the Monte-Carlo routine evaluate thousands of virtual subjects at
    once). `gastroc_sign` is -1 for the corrected model, +1 to reproduce the printed equations.
    Each muscle's exponential is computed exactly once, so the bi-articular energy-transfer
    relationship M_dist = (beta_dist/beta_prox) * M_prox is preserved by construction.
    """
    d = _D2R

    # --- uni-articular base exponentials 
    e_HF = np.exp(-p['beta_HF'] * (hip   - p['alpha_HF'] * d))
    e_HE = np.exp( p['beta_HE'] * (hip   - p['alpha_HE'] * d))
    e_KF = np.exp(-p['beta_KF'] * (knee  - p['alpha_KF'] * d))
    e_KE = np.exp( p['beta_KE'] * (knee  - p['alpha_KE'] * d))
    e_PF = np.exp( p['beta_PF'] * (ankle - p['alpha_PF'] * d))

    # --- bi-articular base exponentials --------------------------------------------------
    r_RF = p['beta_RF_dist'] / p['beta_RF_prox']
    e_RF = np.exp(-p['beta_RF_prox'] * (hip - r_RF * knee - p['alpha_RF'] * d))

    r_HAM = p['beta_HAM_dist'] / p['beta_HAM_prox']
    e_HAM = np.exp( p['beta_HAM_prox'] * (hip - r_HAM * knee - p['alpha_HAM'] * d))

    r_GAS = p['beta_GAS_dist'] / p['beta_GAS_prox']
    e_GAS = np.exp(gastroc_sign * p['beta_GAS_prox'] * (knee - r_GAS * ankle - p['alpha_GAS'] * d))

    # --- assemble the three joint moments (eqs A.5, A.6, A.7) ----------------------------
    M_hip = e_RF - e_HAM + e_HF - e_HE
    M_knee = -r_RF * e_RF + r_HAM * e_HAM + e_GAS + e_KF - e_KE
    M_ankle = -r_GAS * e_GAS - e_PF
    return M_hip, M_knee, M_ankle


def passive_moment(joint, hip=0.0, knee=0.0, ankle=0.0, degrees=True,
                   params=None, gastroc='corrected', extrapolate=True):
    """Predicted passive elastic moment (Nm) at one joint.

    Parameters
    ----------
    joint : {'hip', 'knee', 'ankle', 'all'}
        Which moment to return. 'all' returns a dict with all three.
        (Note per the model: the hip moment depends on hip & knee only; the knee moment on all
        three angles; the ankle moment on knee & ankle only. Unused angles are simply ignored.)
    hip, knee, ankle : float | list | np.ndarray
        Joint angles. Any mix of scalars and arrays is accepted and broadcast together, so e.g.
        passive_moment('hip', hip=np.linspace(-20, 20, 101), knee=0) sweeps the hip with the knee
        held at 0 for the whole sweep.
    degrees : bool, default True
        If True, inputs are interpreted as degrees; if False, as radians.
    params : dict, optional
        Override the Table-4 defaults (same keys as DEFAULT_PARAMS).
    gastroc : {'corrected', 'literal'}, default 'corrected'
        'corrected' flips the sign of the printed gastrocnemius gain so that its passive moment
        grows with knee EXTENSION + ankle DORSIFLEXION (physically correct, and the only version
        that reproduces Fig. 3). 'literal' reproduces eqs A.6/A.7 exactly as typeset. See the docs.
    extrapolate : bool, default True
        If True (default, backward-compatible), the model is evaluated everywhere, including the
        physically implausible blow-ups outside the region the trials measured. If False, angle
        combinations outside the measured envelope (see `in_measured_envelope` and
        MEASURED_RANGE_DEG) return NaN, so a caller can detect/clamp them instead of consuming a
        300+ Nm extrapolation. The envelope is joint-specific: the hip moment is guarded on
        (hip, knee), the ankle moment on (knee, ankle), the knee moment on both.

    Returns
    -------
    float if all inputs are scalar, otherwise a broadcast np.ndarray. For joint='all', a dict.
    """
    if joint not in _JOINTS and joint != 'all':
        raise ValueError(f"joint must be one of {_JOINTS} or 'all', got {joint!r}")
    if gastroc not in ('corrected', 'literal'):
        raise ValueError("gastroc must be 'corrected' or 'literal'")

    p = DEFAULT_PARAMS if params is None else {**DEFAULT_PARAMS, **params}
    gsign = -1.0 if gastroc == 'corrected' else 1.0

    hb, kb, ab = np.broadcast_arrays(np.asarray(hip, float),
                                     np.asarray(knee, float),
                                     np.asarray(ankle, float))
    h, k, a = (hb * _D2R, kb * _D2R, ab * _D2R) if degrees else (hb, kb, ab)

    M_hip, M_knee, M_ankle = _assemble(h, k, a, p, gsign)
    result = {'hip': M_hip, 'knee': M_knee, 'ankle': M_ankle}

    if not extrapolate:
        masks = in_measured_envelope(hb, kb, ab, joint='all', degrees=degrees)
        result = {j: np.where(masks[j], result[j], np.nan) for j in _JOINTS}

    def _out(x):
        x = np.asarray(x, float)
        return x.item() if x.ndim == 0 else x

    if joint == 'all':
        return {j: _out(result[j]) for j in _JOINTS}
    return _out(result[joint])


# Thin convenience wrappers that expose only the angles each moment actually depends on.
def hip_moment(hip=0.0, knee=0.0, **kw):
    """Passive hip moment (Nm). Depends on hip & knee angles (eq A.5)."""
    return passive_moment('hip', hip=hip, knee=knee, **kw)


def knee_moment(hip=0.0, knee=0.0, ankle=0.0, **kw):
    """Passive knee moment (Nm). Depends on hip, knee & ankle angles (eq A.6)."""
    return passive_moment('knee', hip=hip, knee=knee, ankle=ankle, **kw)


def ankle_moment(knee=0.0, ankle=0.0, **kw):
    """Passive ankle moment (Nm). Depends on knee & ankle angles (eq A.7)."""
    return passive_moment('ankle', knee=knee, ankle=ankle, **kw)

if __name__ == '__main__':
    print("scalar hip @ (hip=-10, knee=15):", round(hip_moment(-10, 15), 2), "Nm")
    sweep = passive_moment('hip', hip=np.linspace(-20, 20, 5), knee=0.0)
    print("vector hip sweep (knee=0):", np.round(sweep, 2))
    print("ankle @ (knee=15, ankle=20):", round(ankle_moment(15, 20), 2), "Nm")

    # Extrapolation guard
    corner = dict(hip=-15, knee=110) # never co-measured
    print("\nhip @ (hip=-15, knee=110), extrapolate=True :",
          round(hip_moment(**corner), 1), "Nm  (runaway)")
    print("hip @ (hip=-15, knee=110), extrapolate=False:",
          hip_moment(**corner, extrapolate=False), " (masked, outside envelope)")
    print("in_measured_envelope there:", bool(in_measured_envelope(**corner, joint='hip')))
    print("in_measured_envelope @ gait-like (hip=15, knee=25):",
          bool(in_measured_envelope(hip=15, knee=25, joint='hip')))
