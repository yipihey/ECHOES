"""Imaging systematics-potential (SP) maps for BOSS — the data-driven
decontamination reference (Stage 0b).

The 5 SP columns that drove the BOSS imaging-systematics weights are present
per-object in BOTH the galaxy and the random FITS but were never loaded by the
ECHOES pipeline (only ``WEIGHT_SYSTOT`` and SFD extinction were used). This module
loads them from the **randoms** (which trace the angular footprint uniformly, so a
mean-per-pixel aggregate is the SP template at each sky position) and exposes:

  * :func:`load_sp_maps` → :class:`SPMaps` (HEALPix mean-per-pixel templates),
  * :func:`isd_decontamination` → purely data-driven per-galaxy weights via
    :func:`echoes.systematics.isd_fit` (regress galaxy density flat against every
    template). Dividing out that measured density-vs-SP relation places the field
    at the SP-flat reference state — the data-driven counterfactual that Tier A
    conditions on (see ``snug-sleeping-micali`` plan, §A.2).

SDSS per-band ``5E`` columns are reduced to r-band (index 2), matching the
existing ``extinction_r`` convention in ``validation/sp_null_tests.build_sp_maps``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

RAND_DEFAULT = "data/boss/random0_DR12v5_CMASS_South.fits.gz"
NSIDE_SP = 64                                   # matches validation.sp_null_tests.NSIDE_SP
R_BAND = 2                                       # ugriz index for the 5E per-band columns

# template name -> (FITS column, band index or None for scalar columns)
SP_COLUMNS = {
    "skyflux":     ("SKYFLUX", R_BAND),
    "image_depth": ("IMAGE_DEPTH", R_BAND),
    "psf_fwhm":    ("PSF_FWHM", R_BAND),
    "airmass":     ("AIRMASS", None),
    "eb_minus_v":  ("EB_MINUS_V", None),
}


def _pix(ra, dec, nside):
    import healpy as hp
    return hp.ang2pix(nside, np.radians(90.0 - np.asarray(dec)),
                      np.radians(np.asarray(ra) % 360.0))


def _map_from_values(ra, dec, val, nside):
    """Mean-per-pixel HEALPix map; empty footprint pixels filled with the global
    median so the map can be evaluated at arbitrary query positions. Identical
    convention to ``validation.sp_null_tests._map_from_values``."""
    npix = 12 * nside ** 2
    pix = _pix(ra, dec, nside)
    val = np.asarray(val, float)
    good = np.isfinite(val)
    s = np.bincount(pix[good], weights=val[good], minlength=npix)
    n = np.bincount(pix[good], minlength=npix)
    m = np.full(npix, np.nan); ok = n > 0
    m[ok] = s[ok] / n[ok]
    m[~np.isfinite(m)] = np.nanmedian(m[ok])
    return m


@dataclass
class SPMaps:
    """HEALPix SP template maps + position lookup (the decontamination reference)."""
    nside: int
    names: list                                  # ordered template names
    maps: dict                                   # name -> (npix,) float HEALPix map

    def at(self, ra, dec, name):
        return self.maps[name][_pix(ra, dec, self.nside)]

    def stack_at(self, ra, dec):
        """(N, n_tpl) matrix of every template evaluated at the query positions —
        the input layout :func:`echoes.systematics.isd_fit` expects."""
        return np.column_stack([self.at(ra, dec, nm) for nm in self.names])


def load_sp_maps(rand_path=RAND_DEFAULT, *, nside=NSIDE_SP, columns=None,
                 n_max=None, seed=0, verbose=True) -> SPMaps:
    """Build SP template maps from the survey randoms.

    Reads only the needed columns from the random FITS, reduces per-band ``5E``
    columns to r-band, and mean-aggregates each onto a HEALPix grid.
    """
    from astropy.io import fits
    cols = dict(columns or SP_COLUMNS)
    with fits.open(rand_path) as h:
        d = h[1].data
        ra = np.asarray(d["RA"], float)
        dec = np.asarray(d["DEC"], float)
        raw = {}
        for nm, (col, band) in cols.items():
            arr = np.asarray(d[col])
            raw[nm] = arr[:, band] if band is not None else arr
    if n_max is not None and len(ra) > n_max:
        idx = np.random.default_rng(seed).choice(len(ra), int(n_max), replace=False)
        ra, dec = ra[idx], dec[idx]
        raw = {nm: v[idx] for nm, v in raw.items()}
    names = list(cols.keys())
    maps = {nm: _map_from_values(ra, dec, raw[nm], nside) for nm in names}
    if verbose:
        for nm in names:
            v = maps[nm]
            print(f"  [sp_maps] {nm:12s} median={np.median(v):.4g} "
                  f"[{np.percentile(v,1):.4g}, {np.percentile(v,99):.4g}]  ({len(ra):,} randoms)")
    return SPMaps(nside=nside, names=names, maps=maps)


def isd_decontamination(cat, sp_maps: SPMaps, *, data_weights=None, random_weights=None,
                        thresh=2.0, n_bins=10, order=3):
    """Purely data-driven SP decontamination weights for the galaxies in ``cat``.

    Evaluates every SP template at the data and random positions and runs
    :func:`echoes.systematics.isd_fit` (iterative density-vs-template flattening).
    Returns the :class:`~echoes.systematics.ISDResult` whose ``.weight`` (= 1/F per
    galaxy) places the catalog at the SP-flat reference state. ``data_weights``
    seeds the iteration (e.g. ``cat.w_sys_data`` to remove only the RESIDUAL beyond
    WEIGHT_SYSTOT; ``None`` derives a fresh, fully data-driven weight).
    """
    try:
        from .systematics import isd_fit
    except ImportError:                          # run as a script (smoke test)
        from echoes.systematics import isd_fit
    D = sp_maps.stack_at(np.asarray(cat.ra_data), np.asarray(cat.dec_data))
    R = sp_maps.stack_at(np.asarray(cat.ra_random), np.asarray(cat.dec_random))
    return isd_fit(D, R, names=sp_maps.names, data_weights=data_weights,
                   random_weights=random_weights, thresh=thresh, n_bins=n_bins, order=order)


if __name__ == "__main__":                       # smoke test: load maps + ISD flatten
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from echoes.surveys.boss import load_boss
    sp = load_sp_maps()
    cat = load_boss(["data/boss/galaxy_DR12v5_CMASS_South.fits.gz"],
                    [RAND_DEFAULT], sample="CMASS", nside=256)
    res = isd_decontamination(cat, sp)
    print("\n=== ISD density-vs-SP χ²/dof (≈1 = clean) ===")
    print(f"{'template':12s} {'before':>9s} {'after':>9s}")
    for i, nm in enumerate(res.names):
        print(f"{nm:12s} {res.chi2_before[i]:9.2f} {res.chi2_after[i]:9.2f}")
    print(f"removed (in order): {[res.names[j] for j in res.removal_order]}")
    print(f"weight range: [{res.weight.min():.3f}, {res.weight.max():.3f}]  clean={res.clean}")
