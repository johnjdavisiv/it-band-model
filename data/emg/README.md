# data/emg/ -- the input-data archive unpacks here

EMG data are really large, you need the FigShare data instead (does not fit on GitHub). Downloaded separately and unzipped here -expected contents:

| file | what it is |
|---|---|
| `Run_20002_EMG_RAW.sto` | raw surface EMG, 2 m/s trial |
| `Run_30002_EMG_RAW.sto` | raw surface EMG, 3 m/s trial (the simulated trial) |
| `Run_40002_EMG_RAW.sto` | raw surface EMG, 4 m/s trial |
| `Run_50002_EMG_RAW.sto` | raw surface EMG, 5 m/s trial |
| `Run_20002_GRF.mot` | ground reactions, 2 m/s (gait-cycle events for EMG segmentation) |
| `Run_30002_GRF.mot` | ground reactions, 3 m/s |
| `Run_40002_GRF.mot` | ground reactions, 4 m/s |
| `Run_50002_GRF.mot` | ground reactions, 5 m/s |

Note: we need ALL SPEEDS even though the paper only analyzes 3.0 m/s because the EMG pipeline normalizes each electrode to its **maximum** across ALL of the subject's running trials. The GRF
are Hamner's `Run_X00 02_newCOP3_v24.mot`, renamed without spaces; bytes unchanged. (and we need GRF to segment out steps at all speeds!)

Only `validation/emg/` reads this directory; the core pipeline does not need it.
