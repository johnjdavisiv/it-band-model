"""
Read and aggregate moco solution

"""
from __future__ import annotations

import numpy as np
import pandas as pd
import opensim as osim
from osim_sto import read_sto

def read_activations(sto_path: str) -> pd.DataFrame:
    df = read_sto(sto_path)
    cols = {}
    for c in df.columns:
        if c.startswith("/forceset/") and c.endswith("/activation"):
            cols[c] = c[len("/forceset/"):-len("/activation")]
    act = df[list(cols)].rename(columns=cols)
    return act

def read_controls(sto_path: str) -> pd.DataFrame:
    df = read_sto(sto_path)
    cols = {}
    for c in df.columns:
        if (c.startswith("/forceset/") and c.count("/") == 2
                and "reserve" not in c and "residual" not in c):
            cols[c] = c[len("/forceset/"):]
    return df[list(cols)].rename(columns=cols)

def read_f0m(model_path: str) -> dict[str, float]:
    model = osim.Model(model_path)
    ms = model.getMuscles()
    return {ms.get(i).getName(): float(ms.get(i).getMaxIsometricForce())
            for i in range(ms.getSize())}

def aggregate_activation(act: pd.DataFrame, tracts: list[str],
                         f0m: dict[str, float] | None = None,
                         method: str = "f0m") -> pd.Series:
    present = [t for t in tracts if t in act.columns]
    A = act[present].to_numpy()
    if method == "mean":
        agg = A.mean(axis=1)
    elif method == "max":
        agg = A.max(axis=1)
    elif method == "f0m":
        if f0m is None:
            raise ValueError("method='f0m' requires f0m dict")
        w = np.array([f0m[t] for t in present])
        agg = (A * w).sum(axis=1) / w.sum()
    else:
        raise ValueError(f"unknown method {method}")
    return pd.Series(agg, index=act.index, name="+".join(present))
