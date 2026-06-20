"""Generative inpainting of the un-observed footprint (veto holes + empty regions).

This is the stage that makes ECHOES catalogs uniform: it GENERATES new galaxies
where imaging is vetoed/empty, tagging them ``PROV['inpaint']=5`` with a per-galaxy
prior-dominance ``uncert`` flag. M1 (this module) fills the data-surrounded interior
holes (Regime D) by **analog transplant** (:mod:`echoes.inpaint`) — the cosmology-free
filler that preserves higher-order clustering and the colour/luminosity structure by
construction. Regime-P constrained-realization fills for large empty regions arrive in
M2 via the field engines; ``sample_inpaint_catalog`` is the single entry point both use.
"""

from __future__ import annotations

import numpy as np

from .geometry import _radec_to_nhat

PROV_INPAINT = 5            # must match echoes.completion.PROV["inpaint"]


def _uncert(ra, dec, obs_ra, obs_dec, scale_deg):
    """Per-galaxy prior-dominance flag in [0,1]: normalised angular distance to the
    nearest observed galaxy (0 = data-surrounded interior hole, 1 = ``>=scale_deg``
    from any data, i.e. prior-dominated)."""
    from scipy.spatial import cKDTree
    ra = np.asarray(ra, float); dec = np.asarray(dec, float)
    if obs_ra is None or len(np.atleast_1d(obs_ra)) == 0 or len(ra) == 0:
        return np.zeros(len(ra), np.float32)
    tree = cKDTree(_radec_to_nhat(np.asarray(obs_ra), np.asarray(obs_dec)))
    d, _ = tree.query(_radec_to_nhat(ra, dec), workers=-1)
    dist_deg = np.degrees(2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0)))
    return np.clip(dist_deg / max(scale_deg, 1e-6), 0.0, 1.0).astype(np.float32)


def _empty():
    e = np.zeros(0, np.float32)
    return {"ra": e, "dec": e, "z": e, "prov": np.zeros(0, np.int8), "uncert": e}


def sample_inpaint_catalog(footprint, *, donor_ra, donor_dec, donor_z,
                           rand_ra, rand_dec, donor_colors=None, donor_mags=None,
                           mode="analog", seed=0, density_boost=1.0,
                           uncert_scale_deg=1.0):
    """Generate inpaint galaxies in the fill region of ``footprint``.

    Returns ``dict(ra, dec, z, prov, uncert)`` of NEW galaxies (``prov``=5). ``mode``:
    - ``'analog'`` (M1): transplant real donor galaxies into the interior holes
      (``footprint.holes``), amplitude set selection-immune by the local collar ratio
      × ``density_boost`` (≈⟨w_c⟩). Preserves higher-order clustering by construction.
    Donors are the observed galaxies; ``rand_*`` the survey randoms (window).
    """
    if mode == "analog":
        from .inpaint import inpaint_holes
        if not footprint.holes:
            return _empty()
        reals = inpaint_holes(footprint.holes, footprint.counts, footprint.nside,
                              donor_ra=donor_ra, donor_dec=donor_dec, donor_z=donor_z,
                              rand_ra=rand_ra, rand_dec=rand_dec,
                              donor_colors=donor_colors, donor_mags=donor_mags,
                              seed=seed, n_real=1, density_boost=density_boost)
        r = reals[0]
        ra = np.asarray(r["ra"], np.float32); dec = np.asarray(r["dec"], np.float32)
        z = np.asarray(r["z"], np.float32)
    else:
        raise ValueError(f"inpaint mode {mode!r} not available yet (M1: 'analog' only; "
                         "field-thin / constrained-realization engines arrive in M2)")
    if len(ra) == 0:
        return _empty()
    unc = _uncert(ra, dec, donor_ra, donor_dec, uncert_scale_deg)
    return {"ra": ra, "dec": dec, "z": z,
            "prov": np.full(len(ra), PROV_INPAINT, np.int8), "uncert": unc}
