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
                           thin_oversample=8, thin_aperture_deg=0.7,
                           field_ctx=None, cr_nz=40, transform=None, field_nside=None):
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
    elif mode == "cr":
        # Constrained-realization (LGCP Poisson) fill: per fill pixel draw the STOCHASTIC
        # conditional field 1+δ(z) (Matheron, echoes.fieldpost), set the expected count to
        # the parent density (amplitude calibrated), and draw z from the FIELD-MODULATED
        # n(z) — so the fill has 3-D structure and the per-LOS radial profile, not just the
        # global n(z). Deep in a void the draw reverts to a fair prior realization (correct
        # clustering, uninformed phase) -> flagged by uncert.
        if field_ctx is None:
            raise ValueError("inpaint mode 'cr' requires field_ctx (echoes.fieldpost.build_field_context)")
        import healpy as hp
        from .fieldpost import los_overdensity
        rng = np.random.default_rng(seed)
        fill_pix = np.flatnonzero(footprint.fill_weight > 0)
        if len(fill_pix) == 0:
            return _empty()
        theta, phi = hp.pix2ang(footprint.nside, fill_pix)
        pix_ra = np.degrees(phi); pix_dec = 90.0 - np.degrees(theta)
        zgrid = np.linspace(float(np.min(donor_z)), float(np.max(donor_z)), int(cr_nz))
        nbar_z = np.clip(footprint.nbar_z(zgrid), 0.0, None)
        # The conditional field is smooth at its ~degree correlation scale, so evaluate it
        # over the unique COARSE parent pixels (``field_nside``) and broadcast to the fine
        # fill pixels — this avoids one GP solve per fine pixel (the cost was O(N_fill), which
        # dominates at nside≥512). The fill WEIGHT (1−cover) stays at the fine footprint nside
        # to resolve thin stripes/holes; only the field MODULATION is coarsened.
        fns = min(int(field_nside), footprint.nside) if field_nside else footprint.nside
        if fns < footprint.nside:
            uniq, inv = np.unique(hp.ang2pix(fns, theta, phi), return_inverse=True)
            cth, cph = hp.pix2ang(fns, uniq)
            opd = los_overdensity(field_ctx, np.degrees(cph), 90.0 - np.degrees(cth),
                                  zgrid, n_samples=1, seed=seed)[:, 0, :][inv]
        else:
            opd = los_overdensity(field_ctx, pix_ra, pix_dec, zgrid, n_samples=1, seed=seed)[:, 0, :]
        # Tier-A non-Gaussian reshape of the per-pixel conditional field before it
        # sets the Poisson intensity (echoes.density_transform); None ⇒ Gaussian.
        if transform is not None:
            opd = transform(opd)
        opd = np.clip(opd, 0.0, None)
        pz_pix = nbar_z[None, :] * opd                          # (Npix, nz) unnormalised p(z)
        opd_ang = pz_pix.sum(1) / max(nbar_z.sum(), 1e-12)      # n(z)-weighted mean (1+δ) per pix
        # parent density per FULLY-covered pixel = N_gal / effective covered area
        # (Σ completeness, not pixel COUNT) — so a fractional fill_weight = (1−cover)
        # brings each partial pixel exactly to the full survey density (mass-conserving).
        eff_cov = max(float(np.clip(footprint.observed_cover, 0.0, 1.0).sum()), 1.0)
        nbar_ang = len(np.atleast_1d(donor_ra)) / eff_cov
        fw = footprint.fill_weight[fill_pix]
        # amplitude calibration: the field gives the RELATIVE structure (opd_ang); the parent
        # density sets the TOTAL over the zero-coverage CORE (count-calibrated). Tested
        # alternatives (fractional-rim fill, mass-conserving core boost) both inject worse
        # artifacts than the gentle rim under-fill, so the binary core is kept.
        lam_raw = fw * opd_ang
        target_total = nbar_ang * fw.sum() * density_boost
        lam = lam_raw * (target_total / max(lam_raw.sum(), 1e-12))
        counts = rng.poisson(np.clip(lam, 0.0, None))
        occ = np.flatnonzero(counts)
        if len(occ) == 0:
            return _empty()
        # Vectorised placement (no per-pixel Python loop): expand each occupied pixel to
        # its galaxies, jitter angles within the pixel, and draw z by per-pixel inverse-CDF.
        src = np.repeat(occ, counts[occ])                      # source fill-pixel per galaxy
        N = len(src)
        res = hp.nside2resol(footprint.nside)
        dth = (rng.random(N) - 0.5) * res
        dph = (rng.random(N) - 0.5) * res / np.clip(np.sin(theta[src]), 1e-3, None)
        dec = pix_dec[src] - np.degrees(dth)
        ra = (pix_ra[src] + np.degrees(dph)) % 360.0
        P = np.clip(pz_pix[occ], 0.0, None)                    # (n_occ, nz) per-pixel p(z)
        ssum = P.sum(1, keepdims=True)
        P = np.where(ssum > 0, P / np.maximum(ssum, 1e-30), 1.0 / len(zgrid))
        cdf = np.cumsum(P, axis=1)                             # (n_occ, nz)
        row = np.searchsorted(occ, src)                        # occ is sorted → row per galaxy
        zidx = (cdf[row] < rng.random(N)[:, None]).sum(1)
        z = zgrid[np.clip(zidx, 0, len(zgrid) - 1)]
        ra = ra.astype(np.float32); dec = dec.astype(np.float32); z = z.astype(np.float32)
    else:
        raise ValueError(f"inpaint mode {mode!r} not recognised ('cr', 'thin', or 'analog')")
    if len(ra) == 0:
        return _empty()
    unc = _uncert(ra, dec, donor_ra, donor_dec, uncert_scale_deg)
    return {"ra": ra, "dec": dec, "z": z,
            "prov": np.full(len(ra), PROV_INPAINT, np.int8), "uncert": unc}
