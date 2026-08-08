"""
Semi-hacky OpenSim-API readers/writers for .sto / .mot storage files esp for older versions.

"""
from __future__ import annotations

import numpy as np
import pandas as pd
import opensim as osim


def read_sto(path: str) -> pd.DataFrame:
    #Read a .sto/.mot into a DataFrame indexed by time (seconds).
    #Columns are the file's data labels (the 'time' column becomes the index).
    #Works for both old-version storage files (Hamner EMG/GRF) and modern ones.
    st = osim.Storage(path)
    labels = st.getColumnLabels()
    # labels.get(0) == 'time'; the rest are data columns
    col_names = [labels.get(i) for i in range(1, labels.getSize())]

    n = st.getSize()
    if n == 0:
        raise ValueError(f"empty storage file: {path}")

    ncol = st.getStateVector(0).getSize()
    if ncol != len(col_names):
        # Trailing-tab / label-count mismatch: trust the actual data width.
        col_names = col_names[:ncol] + [f"col{k}" for k in range(len(col_names), ncol)]

    t = np.empty(n)
    data = np.empty((n, ncol))
    for i in range(n):
        sv = st.getStateVector(i)
        t[i] = sv.getTime()
        row = sv.getData()
        for j in range(ncol):
            data[i, j] = row.get(j)

    df = pd.DataFrame(data, columns=col_names)
    df.insert(0, "time", t)
    return df.set_index("time")


def sampling_rate(df: pd.DataFrame) -> float:
    dt = np.median(np.diff(df.index.values))
    return float(1.0 / dt)


def write_sto(df: pd.DataFrame, path: str, name: str | None = None,
              in_degrees: bool = False) -> None:
    table = osim.TimeSeriesTable()
    table.setColumnLabels(list(df.columns))
    for t, row in zip(df.index.values, df.to_numpy()):
        table.appendRow(float(t), osim.RowVector(list(map(float, row))))
    table.addTableMetaDataString("inDegrees", "yes" if in_degrees else "no")
    if name:
        table.addTableMetaDataString("name", name)
    osim.STOFileAdapter().write(table, path)
