"""Probability-integral-transform (PIT) uniformity statistics.

A calibrated posterior has PIT values uniform on [0, 1]. The *mean* PIT ≈ 0.5 is
necessary but not sufficient — a U-shaped (over-confident) or ∩-shaped
(under-confident) PIT also has mean 0.5. We therefore quantify uniformity with a
Kolmogorov–Smirnov test and a binned χ² against the flat distribution, so that
"calibrated" is an actual goodness-of-fit statement, not just a centred mean.
"""

from __future__ import annotations

import numpy as np


def pit_uniformity(pit, n_bins: int = 10) -> dict:
    """Uniformity statistics for PIT values on [0, 1].

    Returns a dict with the mean and std (ideal 0.5, 1/√12 ≈ 0.289), the
    Kolmogorov–Smirnov statistic and p-value vs ``U(0,1)``, and a binned χ²
    (``n_bins`` equal bins) with its dof and p-value. Large p-values (≳ 0.05) are
    consistent with calibration; a small KS/χ² p-value flags miscalibration even
    when the mean is 0.5 (e.g. a U-shaped PIT).
    """
    from scipy import stats

    pit = np.asarray(pit, dtype=np.float64)
    pit = pit[np.isfinite(pit)]
    n = pit.size
    if n < 2:
        return dict(n=n, mean=np.nan, std=np.nan, ks=np.nan, ks_p=np.nan,
                    chi2=np.nan, chi2_dof=n_bins - 1, chi2_p=np.nan)
    ks, ks_p = stats.kstest(np.clip(pit, 0.0, 1.0), "uniform")
    obs, _ = np.histogram(pit, bins=n_bins, range=(0.0, 1.0))
    exp = n / n_bins
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    chi2_p = float(stats.chi2.sf(chi2, n_bins - 1))
    return dict(n=int(n), mean=float(pit.mean()), std=float(pit.std()),
                ks=float(ks), ks_p=float(ks_p),
                chi2=chi2, chi2_dof=n_bins - 1, chi2_p=chi2_p)


def format_pit(stats_dict: dict) -> str:
    """One-line human-readable summary of :func:`pit_uniformity`."""
    s = stats_dict
    return (f"mean={s['mean']:.3f} std={s['std']:.3f}  "
            f"KS={s['ks']:.3f} (p={s['ks_p']:.2f})  "
            f"χ²/dof={s['chi2']/max(s['chi2_dof'],1):.2f} (p={s['chi2_p']:.2f})")
