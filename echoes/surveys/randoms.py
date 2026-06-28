"""Uniform-footprint randoms for ECHOES completed catalogs.

An ECHOES completed catalog has had the survey's *angular selection* removed (the
fiber-assignment / coverage incompleteness is restored as points; the resample +
inpaint completes the footprint). So the matching random catalog must be **uniform
over the footprint** — NOT the survey's own random catalog, whose density traces the
completeness (sparse exactly where the completed catalog added objects). Pairing a
completed catalog with completeness-imprinted survey randoms imprints a spurious
large-scale clustering signal.

For BOSS CMASS-South (COMP ~ 0.99) the survey randoms are already uniform to ~1%, so
this is unnecessary. For DESI (NTILE 1–7) and Quaia (smooth ~0.86 completeness) the
difference is large and this generator is required.

The footprint is defined by the **survey randoms** (their angular extent = the true
observed footprint). We flatten their completeness imprint by importance-resampling
them with weight ``1/completeness`` (the completeness at each random's pixel), which
yields uniform angular density over exactly the survey footprint. A small sub-pixel
jitter breaks the exact duplicates that resampling-with-replacement would otherwise
create (duplicates would put random pairs at zero separation and bias the estimator).
"""
from __future__ import annotations

import numpy as np


def uniform_footprint_randoms(cat, n, *, rng, floor=0.05, jitter_frac=0.4):
    """Uniform-density randoms over the footprint of ``cat`` (which must expose
    ``ra_random``, ``dec_random``, ``z_random``, ``sel_map``, ``nside``).

    Importance-resamples the survey randoms with weight ``1/clip(sel_map, floor, 1)``
    so the completeness imprint is flattened to uniform, then adds a sub-pixel jitter.
    Returns ``(ra, dec, z)``.
    """
    import healpy as hp
    ra_r = np.asarray(cat.ra_random, np.float64)
    dec_r = np.asarray(cat.dec_random, np.float64)
    z_r = np.asarray(cat.z_random, np.float64)
    nside = int(cat.nside)
    sel = np.asarray(cat.sel_map, np.float64)

    pix = hp.ang2pix(nside, np.deg2rad(90.0 - dec_r), np.deg2rad(ra_r % 360.0))
    w = 1.0 / np.clip(sel[pix], floor, 1.0)           # flatten completeness -> uniform density
    w = w / w.sum()
    idx = rng.choice(len(ra_r), size=n, p=w)
    ra = ra_r[idx].copy(); dec = dec_r[idx].copy(); z = z_r[idx].copy()

    # sub-pixel jitter to break the exact duplicates from sampling-with-replacement
    pix_deg = np.degrees(np.sqrt(4.0 * np.pi / (12.0 * nside ** 2)))
    s = jitter_frac * pix_deg
    dec = np.clip(dec + rng.normal(0.0, s, n), -89.999, 89.999)
    ra = (ra + rng.normal(0.0, s, n) / np.cos(np.radians(dec))) % 360.0
    return ra, dec, z
