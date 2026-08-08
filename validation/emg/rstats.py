"""

Aggregation of Pearson r across gait cycles (Fisher z)

"""
import numpy as np

# arctanh(±1) is ±inf; clamp just inside so a degenerate perfect correlation cannot
# poison an average. 1 - 1e-12 -> z = 14.2, far beyond anything real data produces.
_CLAMP = 1.0 - 1e-12

def _z(rs):
    a = np.asarray(list(rs), dtype=float)
    a = a[np.isfinite(a)]
    return np.arctanh(np.clip(a, -_CLAMP, _CLAMP)) if a.size else a

def fisher_mean(rs):
    z = _z(rs)
    return float(np.tanh(z.mean())) if z.size else float("nan")

def fisher_sd_z(rs, ddof=1):
    z = _z(rs)
    return float(z.std(ddof=ddof)) if z.size > ddof else float("nan")

def fisher_interval(rs, k=1.0, ddof=1):
    z = _z(rs)
    if z.size <= ddof:
        return (float("nan"), float("nan"))
    m, s = z.mean(), z.std(ddof=ddof)
    return (float(np.tanh(m - k * s)), float(np.tanh(m + k * s)))

def raw_sd(rs, ddof=1):
    a = np.asarray(list(rs), dtype=float)
    a = a[np.isfinite(a)]
    return float(a.std(ddof=ddof)) if a.size > ddof else float("nan")

def arith_mean(rs):
    a = np.asarray(list(rs), dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")

def summarize(rs, ddof=1):
    lo, hi = fisher_interval(rs, ddof=ddof)
    return dict(r_mean=fisher_mean(rs), r_sd=raw_sd(rs, ddof=ddof),
                r_lo68=lo, r_hi68=hi, r_sd_z=fisher_sd_z(rs, ddof=ddof),
                r_mean_arith_RETIRED=arith_mean(rs))
