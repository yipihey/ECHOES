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
                           mode="thin", seed=0, density_boost=1.0, uncert_scale_deg=1.0,
                           thin_oversample=8, thin_aperture_deg=0.7):
    """Generate inpaint galaxies in the fill region of ``footprint``.

    Returns ``dict(ra, dec, z, prov, uncert)`` of NEW galaxies (``prov``=5). ``mode``:
    - ``'thin'`` (default): Poisson-thin proposal randoms drawn uniformly over the
      WHOLE intended footprint (incl. holes + large empty regions) by
      ``fill_weight · (1+δ)``, where ``1+δ`` is the random-normalised angular
      conditional field (:func:`echoes.selection_coupling.local_overdensity`) — the
      aperture pulls the surrounding density into a hole, and reverts to the prior
      (1+δ→1) in regions deeper than the aperture. Fills holes AND empty regions to
      the surrounding mean density with the large-scale gradient. Redshifts ~ n̄(z).
      (Small-scale stochastic structure inside large voids is mean-field here — the
      full stochastic constrained realization is a documented refinement; those
      galaxies carry a high ``uncert``.)
    - ``'analog'``: transplant real donor galaxies into the interior holes only
      (``footprint.holes``); preserves higher-order clustering, cosmology-free.
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
    elif mode == "thin":
        from .randoms import make_random_from_selection_function
        from .selection_coupling import local_overdensity
        rng = np.random.default_rng(seed)
        n_data = len(np.atleast_1d(donor_ra))
        n_prop = max(int(thin_oversample) * n_data, 1)
        pr, pd, pz = make_random_from_selection_function(
            footprint.target_mask.astype(float), n_prop, np.asarray(donor_z),
            nside=footprint.nside, rng=rng)
        fw = footprint.fill_weight[footprint.pix(pr, pd)]
        sel = fw > 0
        pr, pd, pz, fw = pr[sel], pd[sel], pz[sel], fw[sel]
        if len(pr) == 0:
            return _empty()
        delta = local_overdensity(pr, pd, donor_ra, donor_dec, rand_ra, rand_dec,
                                  aperture_deg=thin_aperture_deg, min_rand=10.0)
        opd = np.clip(np.where(np.isfinite(delta), 1.0 + delta, 1.0), 0.0, None)
        # alpha normalises the accepted density to the (completeness-corrected) parent
        p_acc = np.clip(fw * opd * density_boost / float(thin_oversample), 0.0, 1.0)
        acc = rng.random(len(pr)) < p_acc
        ra = pr[acc].astype(np.float32); dec = pd[acc].astype(np.float32); z = pz[acc].astype(np.float32)
    else:
        raise ValueError(f"inpaint mode {mode!r} not recognised ('thin' or 'analog')")
    if len(ra) == 0:
        return _empty()
    unc = _uncert(ra, dec, donor_ra, donor_dec, uncert_scale_deg)
    return {"ra": ra, "dec": dec, "z": z,
            "prov": np.full(len(ra), PROV_INPAINT, np.int8), "uncert": unc}
