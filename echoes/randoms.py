"""Generic random-catalog utilities for ECHOES.

The BOSS completion and validation code often needs a random catalog drawn from
an angular HEALPix selection map and the empirical redshift distribution of the
data.  This module intentionally contains only survey-agnostic helpers used by
the ECHOES pipeline.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def make_random_from_selection_function(
    sel_map: np.ndarray,
    n_random: int,
    z_data: np.ndarray,
    nside: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw randoms from a HEALPix angular selection map and empirical n(z).

    Pixels are sampled with probability proportional to ``sel_map``.  Positions
    are made continuous inside each selected pixel by jittering within fine
    subpixels, and redshifts are drawn from the data's empirical redshift CDF.

    Parameters
    ----------
    sel_map
        HEALPix RING-order completeness/selection map. Negative values are
        clipped to zero.
    n_random
        Number of random points to draw.
    z_data
        Data redshifts used as the empirical n(z).
    nside
        HEALPix NSIDE. If omitted, inferred from ``sel_map.size``.
    rng
        NumPy random generator. Defaults to a deterministic generator seeded
        with zero.

    Returns
    -------
    ra, dec, z
        One-dimensional arrays with angles in degrees and dimensionless
        redshift.
    """
    import healpy as hp

    if rng is None:
        rng = np.random.default_rng(0)
    sel_map = np.asarray(sel_map, dtype=np.float64)
    if nside is None:
        npix_in = sel_map.size
        nside = int(np.sqrt(npix_in / 12))
    npix = 12 * nside ** 2
    if sel_map.size != npix:
        raise ValueError(f"selection map has {sel_map.size} pixels, expected {npix}")

    p = np.maximum(sel_map, 0.0)
    p_sum = p.sum()
    if p_sum <= 0:
        raise ValueError("selection map has no positive pixels")
    pix_idx = rng.choice(npix, size=int(n_random), p=p / p_sum)

    # Draw inside each coarse pixel.  Choosing fine subpixel centers alone
    # quantizes the random catalog and biases tiny-angle RR; the rejection jitter
    # restores a continuous angular distribution while staying inside the parent
    # selected pixel.
    nside_jitter = nside * 8
    n_jit_per_pix = (nside_jitter // nside) ** 2
    sub = rng.integers(0, n_jit_per_pix, size=int(n_random))
    parent_nest = hp.ring2nest(nside, pix_idx)
    fine_nest = parent_nest * n_jit_per_pix + sub
    fine_ring = hp.nest2ring(nside_jitter, fine_nest)
    theta, phi = hp.pix2ang(nside_jitter, fine_ring)

    pix_size_rad = np.sqrt(4.0 * np.pi / (12.0 * nside_jitter ** 2))
    half = 0.5 * pix_size_rad
    accepted = np.zeros(int(n_random), dtype=bool)
    theta_out = theta.copy()
    phi_out = phi.copy()
    for _ in range(30):
        idx = np.flatnonzero(~accepted)
        if idx.size == 0:
            break
        d_theta = rng.uniform(-half, half, size=idx.size)
        sin_t = np.sin(np.clip(theta[idx], half, np.pi - half))
        d_phi = rng.uniform(-half, half, size=idx.size) / sin_t
        cand_theta = np.clip(theta[idx] + d_theta, 1e-9, np.pi - 1e-9)
        cand_phi = np.mod(phi[idx] + d_phi, 2.0 * np.pi)
        keep = hp.ang2pix(nside_jitter, cand_theta, cand_phi) == fine_ring[idx]
        sel_idx = idx[keep]
        theta_out[sel_idx] = cand_theta[keep]
        phi_out[sel_idx] = cand_phi[keep]
        accepted[sel_idx] = True

    ra = np.degrees(phi_out)
    dec = 90.0 - np.degrees(theta_out)
    z = _sample_z_from_data(z_data, int(n_random), rng)
    return ra, dec, z


def _sample_z_from_data(
    z_data: np.ndarray,
    n: int,
    rng: np.random.Generator,
    n_bins: int = 200,
) -> np.ndarray:
    """Draw ``n`` redshifts from the empirical CDF of ``z_data``."""
    z_data = np.asarray(z_data, dtype=np.float64)
    z_min = float(np.min(z_data))
    z_max = float(np.max(z_data))
    edges = np.linspace(z_min, z_max, n_bins + 1)
    hist, _ = np.histogram(z_data, bins=edges)
    cdf = np.concatenate(([0.0], np.cumsum(hist.astype(np.float64))))
    cdf = cdf / cdf[-1]
    return np.interp(rng.uniform(size=int(n)), cdf, edges)
