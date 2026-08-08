"""
Hamner-style EMG processing (following their conventions)


SUMMARY: --- 

1. offset-correct  : subtract each channel's whole-trial mean
2. rectify         : absolute value
3. low-pass 10 Hz  : zero-phase 2nd-order Butterworth (butter order-1 + filtfilt)
4. segment         : extract each right IC->IC gait cycle
5. normalize       : divide every channel by that channel's MAX across all
                    (non-bad) cycles of all speeds for the subject -> [0,1]

Three things:
  * Bad electrode data are excluded from the per-subject max BEFORE normalizing,
    not removed afterward. (Including bad trials inflates the max, deflating
    every normalized curve for that muscle. subject01 has no bad data, so its
    numbers are unchanged -- but the fix matters for other subjects.)
  * `.sto` files are read through the OpenSim API (osim_sto.read_sto), not a
    hand-rolled text parser.

Duplicated channels: Hamner wired ONE electrode to several tracts and copied the
signal. In the raw file glmax1_r == glmax2_r == glmax3_r, glmed1_r == glmed2_r ==
glmed3_r, and semimem_r == semiten_r (verified byte-identical). gaslat/gasmed and
vaslat/vasmed are genuinely separate electrodes. See EMG_TO_MODEL for how a single
sensor maps onto our multi-tract model.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from osim_sto import read_sto, sampling_rate
from gait_events import detect_gait_cycles, Cycle
from registration import register_cycle

# ---------------------------------------------------------------------------
# EMG channel  ->  model muscle(s) mapping (right side).
#   'direct'  : one electrode == one model muscle.
#   'grouped' : one electrode picked up several tracts; the raw file duplicates
#               the signal across the listed channels. `model_davis`/`model_raj`
#               list the model tracts this single measurement should be compared
#               against (aggregate the MODEL side to match the one EMG signal).
# TFL has no electrode -> tfl12_ITB_r is unvalidated by EMG.
# ---------------------------------------------------------------------------
EMG_DIRECT = {
    "bflh_r":   ["bflh_r"],
    "gaslat_r": ["gaslat_r"],
    "gasmed_r": ["gasmed_r"],
    "recfem_r": ["recfem_r"],
    "soleus_r": ["soleus_r"],
    "tibant_r": ["tibant_r"],
    "vaslat_r": ["vaslat_r"],
    "vasmed_r": ["vasmed_r"],
}

# groupname -> dict(representative EMG channel, duplicate channels, model tracts)
EMG_GROUPS = {
    "glmax": dict(
        emg_rep="glmax1_r",
        emg_dupes=["glmax1_r", "glmax2_r", "glmax3_r"],
        model_davis=["glmax1_r", "glmax2_r", "glmax3_r", "glmax4_r",
                     "glmax12_ITB_r", "glmax34_ITB_r"],
        model_raj=["glmax1_r", "glmax2_r", "glmax3_r"],
    ),
    "glmed": dict(
        emg_rep="glmed1_r",
        emg_dupes=["glmed1_r", "glmed2_r", "glmed3_r"],
        model_davis=["glmed1_r", "glmed2_r", "glmed3_r"],
        model_raj=["glmed1_r", "glmed2_r", "glmed3_r"],
    ),
    "hamstring_med": dict(
        emg_rep="semimem_r",
        emg_dupes=["semimem_r", "semiten_r"],
        model_davis=["semimem_r", "semiten_r"],
        model_raj=["semimem_r", "semiten_r"],
    ),
}

# The unique, physically-independent EMG signals (drop duplicate copies).
UNIQUE_EMG_CHANNELS = list(EMG_DIRECT) + [g["emg_rep"] for g in EMG_GROUPS.values()]


def parse_speed_from_name(fname: str) -> float:
    """'Run_30002_EMG_RAW.sto' / 'Run_30002_GRF.mot' -> 3.0 (m/s). First digit after Run_."""
    m = re.search(r"Run_(\d)", os.path.basename(fname))
    if not m:
        raise ValueError(f"cannot parse speed from {fname}")
    return float(m.group(1))


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------
def emg_envelope(emg: pd.DataFrame, lowpass_hz: float = 10.0,
                 fs: float | None = None) -> pd.DataFrame:
    """Offset-correct -> rectify -> zero-phase 2nd-order Butterworth low-pass.

    Filtering is done on the whole continuous trial (before segmentation) so there
    are no per-cycle edge artifacts. Returns a DataFrame with the same time index.
    """
    if fs is None:
        fs = sampling_rate(emg)
    x = emg.to_numpy()
    x = np.abs(x - x.mean(axis=0, keepdims=True))          # offset + rectify
    b, a = butter(1, lowpass_hz / (fs / 2.0))              # order-1 -> 2nd order via filtfilt
    env = filtfilt(b, a, x, axis=0)
    env = np.clip(env, 0.0, None)                          # rectified envelope is >= 0
    return pd.DataFrame(env, index=emg.index, columns=emg.columns)


# ---------------------------------------------------------------------------
# Trial / subject containers
# ---------------------------------------------------------------------------
@dataclass
class TrialEMG:
    speed: float
    emg_path: str
    grf_path: str
    envelope: pd.DataFrame                 # processed (unnormalized) envelope, whole trial
    cycles: list[Cycle]
    fs: float
    bad_channels: set[str] = field(default_factory=set)  # channels excluded from max/output


def _match_grf(emg_path: str, grf_paths: list[str]) -> str:
    """Match an EMG file to its GRF by the 'Run_N' speed token."""
    tag = os.path.basename(emg_path)[:5]        # 'Run_3'
    hits = [g for g in grf_paths if os.path.basename(g).startswith(tag)]
    if not hits:
        raise FileNotFoundError(f"no GRF for {emg_path} (tag {tag})")
    return hits[0]


def load_trial(emg_path: str, grf_path: str, lowpass_hz: float = 10.0,
               threshold: float = 10.0, bad_channels: set[str] | None = None) -> TrialEMG:
    """Process one trial's EMG and detect its right IC->IC gait cycles from the GRF.

    threshold : vertical-GRF contact threshold (N) for IC detection. Default 10 N is the SAME value
        the Moco cycle definitions use (lib/paper1io) and sits below Hamner subject01's ~10 N
        hard-zeroing floor, so IC = the first nonzero sample. Kept a parameter, not a constant: the
        zeroing floor is trial-specific (e.g. ~28 N at 2 m/s), so for other subjects/speeds in
        Paper 2 pass the per-trial value rather than relying on this default.
    """
    emg_raw = read_sto(emg_path)
    grf = read_sto(grf_path)
    fs = sampling_rate(emg_raw)
    env = emg_envelope(emg_raw, lowpass_hz=lowpass_hz, fs=fs)
    cycles = detect_gait_cycles(grf, side="R", threshold=threshold)
    return TrialEMG(speed=parse_speed_from_name(emg_path), emg_path=emg_path,
                    grf_path=grf_path, envelope=env, cycles=cycles, fs=fs,
                    bad_channels=set(bad_channels or set()))


def cycle_slice(trial: TrialEMG, cycle: Cycle) -> pd.DataFrame:
    """The processed envelope for exactly one gait cycle (by time bounds)."""
    idx = trial.envelope.index.values
    i0 = int(np.argmin(np.abs(idx - cycle.t_start)))
    i1 = int(np.argmin(np.abs(idx - cycle.t_end)))
    return trial.envelope.iloc[i0:i1 + 1]


def subject_max(trials: list[TrialEMG]) -> pd.Series:
    """Per-channel max envelope across all NON-BAD cycles of all trials.

    This is the normalization denominator. Bad (speed,channel) pairs are skipped so
    they cannot inflate the max -- the bug fix. Max is taken over within-cycle data
    only (excludes lead-in/out), matching the original method.
    """
    channels = trials[0].envelope.columns
    acc = {c: -np.inf for c in channels}
    for tr in trials:
        for cyc in tr.cycles:
            seg = cycle_slice(tr, cyc)
            cmax = seg.max(axis=0)
            for c in channels:
                if c in tr.bad_channels:
                    continue
                acc[c] = max(acc[c], float(cmax[c]))
    return pd.Series({c: (np.nan if not np.isfinite(v) else v) for c, v in acc.items()})


@dataclass
class SubjectEMG:
    subject: str
    trials: list[TrialEMG]
    norm: pd.Series # per-channel subject-max normalization value

    @classmethod
    def from_folder(cls, folder: str, subject: str | None = None,
                    lowpass_hz: float = 10.0, threshold: float = 10.0,
                    bad_channels_by_speed: dict[float, list[str]] | None = None):
        """Load + process every speed for one subject folder, then normalize.

        threshold : GRF contact threshold (N) for IC detection, forwarded to every trial. Default
            10 N is consistent with the Moco cycle definitions; override per subject/speed for
            cohort work where the hard-zeroing floor differs (see load_trial).
        bad_channels_by_speed : {speed(m/s): ['glmax1_r', ...] or ['ALL']} to exclude
            known-bad electrode data from the per-subject max (and mark for dropping).
        """
        subject = subject or os.path.basename(folder.rstrip("/"))
        emg_files = sorted(glob.glob(os.path.join(folder, "*EMG_RAW.sto")))
        grf_files = sorted(glob.glob(os.path.join(folder, "*_GRF.mot")))
        if not emg_files:
            raise FileNotFoundError(f"no *EMG_RAW.sto in {folder}")

        bad_map = bad_channels_by_speed or {}
        trials = []
        for ep in emg_files:
            gp = _match_grf(ep, grf_files)
            speed = parse_speed_from_name(ep)
            bad = bad_map.get(speed, [])
            tr = load_trial(ep, gp, lowpass_hz=lowpass_hz, threshold=threshold)
            if bad == ["ALL"]:
                tr.bad_channels = set(tr.envelope.columns)
            else:
                tr.bad_channels = set(bad)
            trials.append(tr)

        norm = subject_max(trials)
        return cls(subject=subject, trials=trials, norm=norm)

    # -- accessors -----------------------------------------------------------
    def trial(self, speed: float) -> TrialEMG:
        for tr in self.trials:
            if tr.speed == speed:
                return tr
        raise KeyError(f"no trial at speed {speed}")

    def normalized_cycle(self, speed: float, cycle_index: int) -> pd.DataFrame:
        """Normalized envelope (divided by subject max), one cycle, real-time index."""
        tr = self.trial(speed)
        seg = cycle_slice(tr, tr.cycles[cycle_index]).copy()
        return seg / self.norm

    def registered_cycles(self, speed: float, n: int = 101, split: bool = False,
                          stance_target: float | None = None,
                          normalized: bool = True) -> tuple[np.ndarray, dict[int, pd.DataFrame]]:
        """Register every cycle of a speed to 0-100% gait cycle.

        Returns (phase, {cycle_index: DataFrame[n x channels]}).
        """
        tr = self.trial(speed)
        out = {}
        phase_ref = None
        for cyc in tr.cycles:
            seg = cycle_slice(tr, cyc)
            y = (seg / self.norm) if normalized else seg
            phase, yr = register_cycle(seg.index.values, y.to_numpy(), cyc,
                                       n=n, split=split, stance_target=stance_target)
            out[cyc.index] = pd.DataFrame(yr, columns=seg.columns)
            phase_ref = phase
        return phase_ref, out

    def tidy(self, n: int = 101, split: bool = False, stance_target: float | None = None,
             unique_only: bool = True, drop_bad: bool = True) -> pd.DataFrame:
        """Long tidy table: one row per (speed, cycle, muscle, phase point).

        Columns: subject, speed_m_s, cycle, muscle, phase_pct, emg_norm.
        """
        records = []
        for tr in self.trials:
            phase, regs = self.registered_cycles(tr.speed, n=n, split=split,
                                                  stance_target=stance_target, normalized=True)
            channels = UNIQUE_EMG_CHANNELS if unique_only else list(tr.envelope.columns)
            for cidx, df in regs.items():
                for ch in channels:
                    if drop_bad and ch in tr.bad_channels:
                        continue
                    for p, v in zip(phase, df[ch].values):
                        records.append((self.subject, tr.speed, cidx, ch, p, v))
        return pd.DataFrame.from_records(
            records, columns=["subject", "speed_m_s", "cycle", "muscle", "phase_pct", "emg_norm"])
