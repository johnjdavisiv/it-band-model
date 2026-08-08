"""Time-normalize a time series to 0-100% of a gait cycle.

Works for EMG envelope, activations, joint angles, etc... 
"""
from __future__ import annotations
import numpy as np

def register(t, y, t_start, t_end, n=101):
    """
    t : (T,) time samples (seconds)
    y : (T,) or (T, M) signal(s).
    t_start, t_end : cycle bounds (seconds); phase 0% and 100%.
    n : number of phase points.

    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    squeeze = y.ndim == 1
    if squeeze:
        y = y[:, None]

    phase = np.linspace(0.0, 100.0, n)
    query = np.linspace(t_start, t_end, n)
    y_reg = np.column_stack([np.interp(query, t, y[:, m]) for m in range(y.shape[1])])

    return phase, (y_reg[:, 0] if squeeze else y_reg)