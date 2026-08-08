"""
Gait-event detection from GRF 
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

@dataclass
class Cycle:
    index: int # 0-based cycle number within the trial
    t_start: float # this IC (start of cycle)
    t_end: float # next IC (end of cycle) = start of next cycle
    t_toe_off: float # toe-off within this cycle (stance -> swing transition)
    stance_frac: float # (t_toe_off - t_start) / (t_end - t_start), in [0, 1]

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


def _rising_falling(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rise = np.where((~mask[:-1]) & (mask[1:]))[0] + 1
    fall = np.where((mask[:-1]) & (~mask[1:]))[0] + 1
    return rise, fall


def detect_gait_cycles(grf: pd.DataFrame, side: str = "R",
                       threshold: float = 10.0,   # below Hamner's ~10 N zeroing floor; see module docstring
                       drop_partial_first: bool = True,
                       min_flight_frac: float = 0.0) -> list[Cycle]:
    """Detect right (or left) gait cycles from a GRF DataFrame.

    Parameters
    ----------
    grf : DataFrame indexed by time with a `{side}_ground_force_vy` column.
    side : 'R' or 'L'.
    threshold : vertical force (N) above which the foot is in contact. Must sit BELOW the
        trial's zeroing floor (~10 N for Hamner subject01 3/4/5 m/s) or IC lands late.
    drop_partial_first : if the trial starts already in contact, that leading
        partial stance is not a valid IC -> drop it (matches the old MATLAB).
    min_flight_frac : ignore contact gaps shorter than this fraction of the median
        stride (debounce tiny threshold flickers); 0 disables.

    Returns a list of full IC->IC cycles (one fewer than the number of ICs).
    """
    col = f"{side}_ground_force_vy"
    if col not in grf.columns:
        raise KeyError(f"{col} not in GRF columns: {list(grf.columns)[:8]}...")

    t = grf.index.values
    contact = grf[col].values > threshold

    rise, fall = _rising_falling(contact)

    # If we start mid-contact, the first "IC" isn't a true IC then drop it.
    if drop_partial_first and contact[0] and len(rise) and (len(fall) == 0 or fall[0] < rise[0]):
        # falling edge before any rising edge then leading partial stance present
        fall = fall[fall > (rise[0] if len(rise) else 0)] if len(rise) else fall[1:]
    # ICs are rising edges of contact.
    ic_idx = rise.copy()
    if drop_partial_first and contact[0]:
        # first sample already in contact -> its stance has no detected IC; the first
        # rising edge is the first *true* IC, so nothing to drop from `rise` itself.
        pass

    # Optional debounce of spurious short flights between ICs.
    if min_flight_frac > 0 and len(ic_idx) >= 2:
        med_stride = np.median(np.diff(t[ic_idx]))
        keep = [ic_idx[0]]
        for k in ic_idx[1:]:
            if (t[k] - t[keep[-1]]) >= min_flight_frac * med_stride:
                keep.append(k)
        ic_idx = np.array(keep)

    cycles: list[Cycle] = []
    for c in range(len(ic_idx) - 1):
        i0, i1 = ic_idx[c], ic_idx[c + 1]
        t0, t1 = t[i0], t[i1]
        # toe-off = first falling edge strictly after this IC and before next IC
        to_candidates = fall[(fall > i0) & (fall < i1)]
        if len(to_candidates) == 0:
            # no clean toe-off found in-window; mark unknown
            t_to = np.nan
            stance_frac = np.nan
        else:
            t_to = t[to_candidates[0]]
            stance_frac = (t_to - t0) / (t1 - t0)
        cycles.append(Cycle(index=c, t_start=float(t0), t_end=float(t1),
                            t_toe_off=float(t_to), stance_frac=float(stance_frac)))
    return cycles
