"""
IK on the whole 3 m/s recording (IK is framewise, doesn't matter)
Outputs (in pipeline-work/ik/):  Run_30002_ik.mot, marker error report.

"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.paper1io import TEMPLATES, WORK, RUN_TRC, resolve_template  # noqa: E402

OUT = os.path.join(WORK, "ik")
SCALED_MODEL = os.path.join(WORK, "scale", "Rajagopal2023_subject01_scaled.osim")
IK_MOT = os.path.join(OUT, "Run_30002_ik.mot")


def main():
    assert os.path.exists(SCALED_MODEL), f"missing {SCALED_MODEL} -- run 01_scale.py first"
    os.makedirs(OUT, exist_ok=True)
    setup = resolve_template(
        os.path.join(TEMPLATES, "ik_setup.xml"),
        {"OUT_DIR": OUT,
         "SCALED_MODEL": SCALED_MODEL,
         "RUN_TRC": RUN_TRC,
         "OUT_MOT": IK_MOT},
        os.path.join(OUT, "ik_setup_resolved.xml"))
    print(f"[ik] running InverseKinematicsTool ({os.path.basename(setup)}) ...")
    subprocess.run(["opensim-cmd", "run-tool", setup], check=True, cwd=OUT)
    assert os.path.exists(IK_MOT), f"IK did not write {IK_MOT}"
    report_marker_errors()
    print(f"[ik] OK -> {IK_MOT}")


def report_marker_errors():
    # per-frame marker error report (log file not useful)
    import numpy as np
    err_files = [f for f in os.listdir(OUT) if f.endswith("_ik_marker_errors.sto")]
    if not err_files:
        print("[ik] no marker-error report found (setup did not request it)")
        return
    path = os.path.join(OUT, err_files[0])
    with open(path) as fh:
        lines = fh.readlines()
    header = next(i for i, ln in enumerate(lines) if ln.startswith("endheader")) + 1
    cols = lines[header].split()
    data = np.loadtxt(lines[header + 1:])
    rms = data[:, cols.index("marker_error_RMS")]
    wmax = data[:, cols.index("marker_error_max")]
    print(f"[ik] marker error over {len(rms)} frames: "
          f"RMS mean {rms.mean()*1000:.1f} mm (worst frame {rms.max()*1000:.1f} mm), "
          f"max-marker mean {wmax.mean()*1000:.1f} mm (worst {wmax.max()*1000:.1f} mm)")


if __name__ == "__main__":
    main()
