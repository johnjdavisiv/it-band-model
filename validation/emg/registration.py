"""
Time-normalize a time series to 0-100% of a gait cycle.

"""
from __future__ import annotations

import numpy as np


def register(t, y, t_start, t_end, n=101, stance_frac=None,
             stance_target=None, n_stance=None, n_swing=None):
    """Register y(t) onto a phase axis for one gait cycle.

    Parameters
    ----------
    t : (T,) time samples (seconds), monotonically increasing.
    y : (T,) or (T, M) signal(s).
    t_start, t_end : cycle bounds (seconds); phase 0% and 100%.
    n : number of phase points for single-phase mode.
    stance_frac : if given, enables stance/swing split registration. This is the
        stance fraction of THIS cycle (t_toe_off relative to the cycle), in (0,1).
    stance_target : phase (%) at which stance should end after registration
        (default = stance_frac*100, i.e. no warping for a lone cycle; pass a fixed
        value such as the cohort-mean stance % to align many cycles).
    n_stance, n_swing : points in each phase (default 51 / 50 -> 101 total).

    Returns
    -------
    phase : (n,) phase in percent [0..100].
    y_reg : (n,) or (n, M) resampled signal.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    squeeze = y.ndim == 1
    if squeeze:
        y = y[:, None]

    def _interp_time(query_times):
        cols = [np.interp(query_times, t, y[:, m]) for m in range(y.shape[1])]
        return np.column_stack(cols)

    if stance_frac is None:
        # single-phase
        phase = np.linspace(0.0, 100.0, n)
        query = np.linspace(t_start, t_end, n)
        y_reg = _interp_time(query)
    else:
        if not (0.0 < stance_frac < 1.0):
            raise ValueError(f"stance_frac must be in (0,1), got {stance_frac}")
        n_stance = n_stance if n_stance is not None else 51
        n_swing = n_swing if n_swing is not None else 50
        stance_target = stance_target if stance_target is not None else stance_frac * 100.0

        t_to = t_start + stance_frac * (t_end - t_start)
        # stance: real time [t_start, t_to] -> phase [0, stance_target]
        ph_stance = np.linspace(0.0, stance_target, n_stance)
        q_stance = np.linspace(t_start, t_to, n_stance)
        # swing: real time [t_to, t_end] -> phase [stance_target, 100]
        ph_swing = np.linspace(stance_target, 100.0, n_swing + 1)[1:]
        q_swing = np.linspace(t_to, t_end, n_swing + 1)[1:]

        phase = np.concatenate([ph_stance, ph_swing])
        y_reg = _interp_time(np.concatenate([q_stance, q_swing]))

    return phase, (y_reg[:, 0] if squeeze else y_reg)


def register_cycle(t, y, cycle, n=101, split=False, stance_target=None,
                   n_stance=None, n_swing=None):
    """Convenience wrapper: register using a gait_events.Cycle record."""
    sf = cycle.stance_frac if (split and np.isfinite(cycle.stance_frac)) else None
    return register(t, y, cycle.t_start, cycle.t_end, n=n, stance_frac=sf,
                    stance_target=stance_target, n_stance=n_stance, n_swing=n_swing)
