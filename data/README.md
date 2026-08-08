# data/ -- experimental inputs (Hamner & Delp 2013, subject01)

Dat from one subject ("subject01") of the published Hamner & Delp 2013 dataset

> Hamner SR, Delp SL. Muscle contributions to fore-aft and vertical body mass center accelerations over a range of running speeds. J Biomech. 2013;46(4):780-787.

## Shipped in this repository (used by the pipeline)

| file | data |
|---|---|
| `Static_FJC.trc` | static cal, 64 markers, 100 Hz (input to `scripts/01_scale.py`) |
| `Run_30002.trc` | running trial at 3 m/s, 42 markers (tracking removed), 100 Hz (input to `scripts/02_ik.py`) |
| `Run_30002_GRF.mot` | ground reaction forces + COP for the same trial, 1000 Hz, as filtered and distributed by Hamner (input to RRA and MocoInverse) |
| `templates/` | OpenSim tool setup templates: Scale and IK setups (adapted from the Rajagopal 2016 model distribution), and the RRA task/actuator structure templates. FYI - scripts change the paths dynamically because opensim needs absolute paths, not relative ones

(`Run_30002.trc` and `Run_30002_GRF.mot` are Hamner's files `Run_300 02.trc` and `Run_300 02_newCOP3_v24.mot`, renamed to remove the space; the bytes are unchanged.)

## Downloaded separately: `emg/`

The EMG validation (`validation/emg/`) additionally needs subject01's raw EMG and the GRFs of the other running speeds (~11 MB). Download the **input-data archive** from the paper's figshare deposit and unzip it into `data/emg/` (see `data/emg/README.md` for the expected file list). Nothing in the core pipeline (scale -> IK -> RRA -> MocoInverse) needs it.
