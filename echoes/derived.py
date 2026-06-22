"""Rest-frame absolute magnitudes + stellar masses from ugriz + z (kcorrect SED fit).

These are deterministic functions of a galaxy's photometry and redshift, so they are
DERIVED on demand rather than stored in the posterior package: the redshift varies per
realization, and because the completed catalog already reproduces the true ``P(ugriz | z)``
at each redshift (Phase A; ``validation/property_recovery.py``), *any* such derived
quantity — absolute magnitude, stellar mass — reproduces the truth at each redshift by
construction.

Self-consistent and cosmology-light: the SED fit uses the carried ugriz photometry and the
(measured or completed) redshift; only the absolute-magnitude distance modulus uses a
fiducial cosmology (kcorrect's default). Requires the optional ``kcorrect`` dependency
(``pip install kcorrect``); kept out of the numpy-only release sampler.

    from echoes.derived import add_derived
    cat = draw(pkg, seed=0)            # carries mags + z
    add_derived(cat)                   # adds cat['absmag'] (N,5 rest-frame ugriz), cat['logmass']
"""
from __future__ import annotations

import numpy as np

SDSS_RESPONSES = ["sdss_u0", "sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0"]
_CACHE = {}


def _kcorrect(responses=tuple(SDSS_RESPONSES)):
    key = tuple(responses)
    if key not in _CACHE:
        from kcorrect.kcorrect import Kcorrect          # lazy: optional dependency
        _CACHE[key] = Kcorrect(responses=list(key))
    return _CACHE[key]


def derive_properties(mags, z, *, err_frac=0.02, band_shift=0.0, responses=tuple(SDSS_RESPONSES)):
    """Absolute mags (N,5 rest-frame ugriz) + log10 stellar mass (N,) from ugriz + z.

    ``mags`` (N,5) apparent ugriz model mags (NaN bands are down-weighted, not dropped —
    the frequently-bad u still lets g/r/i/z drive the fit). ``err_frac`` sets a nominal
    per-band photometric error (CMASS has no per-object errors in the LSS file). Returns
    ``{absmag, logmass}`` (float32; NaN where the fit gives non-positive mass)."""
    mags = np.asarray(mags, float)
    z = np.asarray(z, float)
    fin = np.isfinite(mags)
    maggies = np.where(fin, 10.0 ** (-0.4 * np.where(fin, mags, 0.0)), 0.0)
    ivar = np.where(fin, 1.0 / (err_frac * np.maximum(maggies, 1e-30)) ** 2, 0.0)
    kc = _kcorrect(responses)
    coeffs = kc.fit_coeffs(redshift=z, maggies=maggies, ivar=ivar)
    absmag = kc.absmag(redshift=z, maggies=maggies, ivar=ivar, coeffs=coeffs,
                       band_shift=band_shift)
    mrem = np.asarray(kc.derived(redshift=z, coeffs=coeffs)["mremain"], float)
    logmass = np.where(mrem > 0, np.log10(mrem, where=mrem > 0), np.nan)
    return {"absmag": np.asarray(absmag, np.float32), "logmass": logmass.astype(np.float32)}


def add_derived(cat, **kw):
    """Append ``absmag`` (N,5) and ``logmass`` (N,) to a drawn catalog dict (needs ``mags``)."""
    if "mags" not in cat:
        raise ValueError("catalog has no 'mags' — draw from a package built with photometry")
    d = derive_properties(cat["mags"], cat["z"], **kw)
    cat["absmag"] = d["absmag"]
    cat["logmass"] = d["logmass"]
    return cat
