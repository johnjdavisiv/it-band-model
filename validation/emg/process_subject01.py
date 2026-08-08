"""

Analyze subject01 data (EMG)

"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hamner_emg import SubjectEMG, UNIQUE_EMG_CHANNELS

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE)) # repo root (validation/emg  to repo)
DATA_EMG = os.path.join(PKG, "data", "emg") # raw Hamner EMG + GRF archive
RESULTS = os.path.join(HERE, "results")
PLOTS = os.path.join(HERE, "plots")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(PLOTS, exist_ok=True)

FOLDER = DATA_EMG

# subject01 has no known bad electrode data (Hamner's bad trials are on other
# subjects). This dict is where per-subject exclusions go, e.g.:
#   bad_channels_by_speed={5.0: ["ALL"]} # subject10 5 m/s misaligned
#   bad_channels_by_speed={2.0: ["glmax1_r","glmax2_r",...]} # subject20 glmax
BAD_CHANNELS_BY_SPEED = {}

def main():
    subj = SubjectEMG.from_folder(FOLDER, subject="subject01",
                                  bad_channels_by_speed=BAD_CHANNELS_BY_SPEED)
    # 1) tidy normalized+registered table (all speeds, all cycles)
    tidy = subj.tidy(n=101)
    tidy.to_csv(os.path.join(RESULTS, "subject01_emg_normalized_registered.csv"), index=False)
    # 2) normalization denominators
    subj.norm.rename("subject_max").to_csv(
        os.path.join(RESULTS, "subject01_emg_subject_max.csv"))
    # 3) per-cycle index
    rows = []
    for tr in subj.trials:
        for c in tr.cycles:
            rows.append((tr.speed, c.index, c.t_start, c.t_end, c.duration,
                         c.t_toe_off, c.stance_frac))
    pd.DataFrame(rows, columns=["speed_m_s", "cycle", "t_start", "t_end",
                                "duration_s", "t_toe_off", "stance_frac"]).to_csv(
        os.path.join(RESULTS, "subject01_cycle_index.csv"), index=False)

    # 4) overview grid: muscles (rows) x speeds (cols); all cycles thin + mean thick
    speeds = sorted(tr.speed for tr in subj.trials)
    muscles = UNIQUE_EMG_CHANNELS
    fig, axes = plt.subplots(len(muscles), len(speeds),
                             figsize=(2.6 * len(speeds), 1.5 * len(muscles)),
                             sharex=True, sharey=True)
    for r, m in enumerate(muscles):
        for c, sp in enumerate(speeds):
            ax = axes[r, c]
            sub = tidy[(tidy.speed_m_s == sp) & (tidy.muscle == m)]
            phase = np.linspace(0, 100, 101)
            stack = []
            for cyc, g in sub.groupby("cycle"):
                y = g.sort_values("phase_pct").emg_norm.values
                ax.plot(phase, y, color="0.55", lw=0.7, alpha=0.7)
                stack.append(y)
            if stack:
                ax.plot(phase, np.mean(stack, axis=0), color="C3", lw=1.8)
            ax.set_ylim(0, 1)
            ax.set_xlim(0, 100)
            if r == 0:
                ax.set_title(f"{sp:.0f} m/s", fontsize=10)
            if c == 0:
                ax.set_ylabel(m.replace("_r", ""), fontsize=8, rotation=0,
                              ha="right", va="center")
            ax.tick_params(labelsize=7)
    for c in range(len(speeds)):
        axes[-1, c].set_xlabel("% gait cycle", fontsize=8)
    fig.suptitle("subject01 Hamner EMG — normalized to per-subject max, all cycles "
                 "(grey) + mean (red)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out = os.path.join(PLOTS, "subject01_emg_all_cycles_grid.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)

    n_cycles = sum(len(tr.cycles) for tr in subj.trials)
    print(f"[ok] subject01: {len(subj.trials)} speeds, {n_cycles} gait cycles")
    print(f"[ok] tidy rows: {len(tidy)}  ({n_cycles} cycles x {len(muscles)} muscles x 101 pts)")
    print(f"[ok] wrote results/ + {os.path.relpath(out, HERE)}")

if __name__ == "__main__":
    main()
