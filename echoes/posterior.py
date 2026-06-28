"""Compact, shareable posterior over completed BOSS CMASS catalogs + a fast sampler.

The completion's structure makes it cheaply shareable:

  * Every realization keeps the SAME observed galaxies (RA, Dec, spec-z). They are
    stored ONCE, not duplicated per sample.
  * Only the ~9% missing galaxies' redshifts vary between realizations, and each
    missing galaxy's redshift posterior p(z | n̂, colours) is FIXED (it is built
    from the observed field + colours, independent of the random seed). Only the
    DRAW from it changes. So we precompute each missing galaxy's posterior once,
    store it as a compact inverse-CDF (quantile function, uint16), and a "sample"
    is just seeded inverse-CDF sampling — microseconds, unlimited draws, ~zero
    storage per sample.

This module:
  * :func:`build_package` — runs the expensive local-density (KNN) posterior ONCE and
    returns a compact package (immutable observed base + missing positions + per-
    object inverse-CDF + systot weights + provenance).
  * :func:`write_package` / :func:`load_package` — heavily-compressed (.npz, uint16
    quantised) serialization; a few MB for the whole posterior.
  * :func:`draw` — reconstruct a full equal-weight realization from a seed. Fast
    and exactly equivalent (in distribution) to
    :func:`observed_ls.complete_catalog_photoz` with ``z_mode='field'``.

The samples are inherently RELATIVE: the shared observed base is the bulk of every
catalog and is stored once; a fixed reproducible ensemble of K realizations needs
only K seeds (or, if materialized, K small redshift-delta arrays for the missing
galaxies), never K copies of the observed galaxies.
"""
from __future__ import annotations

import numpy as np

from .completion import (_radec_to_nhat, _clpair_density, _systot_restore_extras,
                           measure_close_pair_dz, PROV)

QUANT_VERSION = 1
MAG_LO, MAG_HI = 10.0, 32.0          # uint16 quantisation window for ugriz model mags


def _quant_mags(m, lo=MAG_LO, hi=MAG_HI):
    """Quantise per-band mags to uint16 in [1,65535]; non-finite → sentinel 0 (so a bad
    single band, e.g. the frequently-negative u flux, is preserved as NaN without
    discarding the galaxy's good g/r/i/z). Precision ~3e-4 mag ≪ photometric error."""
    m = np.asarray(m, np.float64)
    q = np.clip(np.round((m - lo) / (hi - lo) * 65534.0) + 1.0, 1, 65535)
    q = np.where(np.isfinite(m), q, 0.0)
    return q.astype(np.uint16)


def _dequant_mags(q, lo=MAG_LO, hi=MAG_HI):
    out = lo + (np.asarray(q).astype(np.float32) - 1.0) / 65534.0 * (hi - lo)
    out[np.asarray(q) == 0] = np.nan
    return out.astype(np.float32)


def _assemble_base_mags(catalog, targets):
    """Fixed (n_obs+n_miss, 5) ugriz model mags — REAL for observed and missing (the
    missing targets are real imaging detections). ``None`` if photometry is absent."""
    if getattr(catalog, "mags_data", None) is None or targets.mags is None:
        return None
    return np.concatenate([np.asarray(catalog.mags_data, np.float64),
                           np.asarray(targets.mags, np.float64)], axis=0).astype(np.float32)


def _mag_cols(mags):
    """``{mags, colors, colors_finite}`` from a (N,5) ugriz array, or ``{}`` if None.
    Colors are the four adjacent differences (the ``fluxes_to_colors`` convention)."""
    if mags is None:
        return {}
    m = np.asarray(mags, np.float32)
    return {"mags": m, "colors": (m[:, :-1] - m[:, 1:]).astype(np.float32),
            "colors_finite": np.isfinite(m).all(axis=1)}


def _phi(x):
    """Standard-normal CDF Φ, pure numpy (Abramowitz & Stegun 7.1.26 erf, |err|≲1.5e-7).

    Used by the copula sampler to map a correlated Gaussian g (unit marginal variance)
    to a uniform u=Φ(g); kept dependency-free so the lightweight numpy-only reproducer
    (``data_release/draw_samples.py``) and ``echoes.draw`` produce IDENTICAL catalogs."""
    z = np.asarray(x, np.float64) / np.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * np.abs(z))
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
                + t * (-1.453152027 + t * 1.061405429))))
    erf = np.sign(z) * (1.0 - poly * np.exp(-z * z))
    return 0.5 * (1.0 + erf)


def _build_copula_modes(base_ra, base_dec, n_obs, invcdf, qlev, cov, *, K=128, max_m=40000, verbose=False):
    """Low-rank factor of the field-correlation copula over the missing galaxies.

    The released sampler draws ``z_i = invcdf_i(u_i)`` with IID ``u`` — a Gaussian
    copula with the IDENTITY correlation, which under-disperses the coherent
    large-scale completion variance (see ``validation/completion_covariance_shape``).
    Here we replace the identity with the MEASURED field correlation ``C_ij =
    ξ(|x_i-x_j|)/ξ(0)`` (the same ``(cov_bins,cov_vals)`` kernel the fieldpost engine
    uses), placing each missing galaxy at its angular position and marginal-median
    redshift. We store the leading ``K`` eigen-modes ``B = V_K √Λ_K`` plus a diagonal
    residual ``d`` with ``BᵀB`` rows + d² = 1, so ``g = Bη + d⊙ε`` has EXACT unit
    marginal variance ⇒ ``Φ(g_i)`` is exactly uniform ⇒ every per-object marginal
    (hence the per-object PIT calibration) is unchanged; only the JOINT law gains the
    cross-object dependence. The narrow data-conditioned marginals automatically damp
    the coupling near observed galaxies (the conditioning is in z-space)."""
    from .clustering import comoving_mpc_h
    from .field_posterior import _cov_matrix, _k0
    M = invcdf.shape[0]
    if M == 0 or cov is None:
        return None, None
    # The copula needs a dense (M,M) correlation eigendecomposition — O(M²) memory, O(M³) time.
    # Above ~40k missing objects this is infeasible (M=113k ⇒ ~100 GB + hours). Cap it: ship the IID
    # sampler (per-object marginals/PIT unchanged; only the cross-object dependence is dropped). The
    # scale path is a low-rank Nyström over landmark points — a follow-up; until then large-M datasets
    # (e.g. DESI) run copula-free rather than OOM. ``max_m=None`` forces the dense path.
    if max_m is not None and M > max_m:
        if verbose:
            print(f"[copula] M={M:,} > max_m={max_m:,}: dense eigh infeasible (O(M³)); shipping IID "
                  f"(marginals/PIT unchanged). Use a Nyström low-rank copula for large M.")
        return None, None
    ra_m = np.asarray(base_ra[n_obs:], np.float64)
    dec_m = np.asarray(base_dec[n_obs:], np.float64)
    qlev = np.asarray(qlev, np.float64)
    # marginal-median z per object (vectorised; qlev is shared, only invcdf rows differ)
    j = int(np.clip(np.searchsorted(qlev, 0.5), 1, len(qlev) - 1))
    f = (0.5 - qlev[j - 1]) / max(qlev[j] - qlev[j - 1], 1e-12)
    z_med = invcdf[:, j - 1] + (invcdf[:, j] - invcdf[:, j - 1]) * f
    X = comoving_mpc_h(z_med)[:, None] * _radec_to_nhat(ra_m, dec_m)          # (M,3) Mpc/h
    C = _cov_matrix(cov, X, X) / max(_k0(cov), 1e-30)                         # unit-diag correlation
    C = 0.5 * (C + C.T)
    w, V = np.linalg.eigh(C)
    K = int(min(K, M))
    idx = np.argsort(w)[::-1][:K]
    B = V[:, idx] * np.sqrt(np.clip(w[idx], 0.0, None))[None, :]              # (M,K)
    d = np.sqrt(np.clip(1.0 - (B ** 2).sum(1), 0.0, None))
    if verbose:
        var_expl = float(np.clip(w[idx], 0, None).sum() / max(np.clip(w, 0, None).sum(), 1e-30))
        print(f"[copula] M={M} K={K} captures {100*var_expl:.0f}% of field-correlation variance")
    return B.astype(np.float32), d.astype(np.float32)


def _grid_kde(z_pts, w_pts, *, zmin, dzg, ngrid, bw):
    """Per-row Gaussian KDE of weighted points ``z_pts`` (M,k) onto a length-``ngrid`` z-grid, via
    histogram-onto-grid + ``gaussian_filter1d`` — the KDE IS a Gaussian convolution, so this is one
    vectorised C call for all M objects instead of O(M·ngrid·k) brute-force exp (~10× faster). The
    bin snap (≤½ cell ≪ ``bw``) and the unnormalised KDE constant both wash out when the downstream
    CDF is normalised. Returns ``(M, ngrid)``."""
    from scipy.ndimage import gaussian_filter1d
    z_pts = np.asarray(z_pts); M = z_pts.shape[0]
    idx = np.clip(np.round((z_pts - zmin) / dzg), 0, ngrid - 1).astype(np.intp)
    rows = np.broadcast_to(np.arange(M)[:, None], z_pts.shape)
    hist = np.zeros((M, ngrid), np.float64)
    np.add.at(hist, (rows.ravel(), idx.ravel()), np.asarray(w_pts, np.float64).ravel())
    return gaussian_filter1d(hist, bw / dzg, axis=1, mode="constant", truncate=5.0)


def _broaden_close_pair_dz(dz_pool, z_o, frac):
    """Mix a fraction ``frac`` of the close-pair Δz pool with the chance-projection BACKGROUND (Δz of
    uncorrelated pairs ~ the n(z) self-convolution). The observed survivor close-pair pool is biased to
    physical Δz≈0, so the collided posterior z_host+Δz is too narrow → over-confident PIT; frac≈0.30
    calibrates it (seed-stable, wp-safe; validation/pit_closepair_prototype.py). Fixed RNG → reproducible."""
    if not frac or frac <= 0:
        return dz_pool
    rb = np.random.default_rng(0)
    n = len(dz_pool)
    bgd = z_o[rb.integers(len(z_o), size=n)] - z_o[rb.integers(len(z_o), size=n)]
    bgd = np.concatenate([bgd, -bgd])
    out = np.asarray(dz_pool, np.float64).copy()
    take = rb.random(n) < frac
    out[take] = rb.choice(bgd, int(take.sum()))
    return out


def _has_colours(targets):
    """True iff ``targets`` carries a non-empty colour feature array (i.e. a
    colour-space photo-z factor is available). DESI/Quaia targets carry ``(M,0)``
    colours → field-only / Gaussian posterior."""
    c = getattr(targets, "colors", None)
    return c is not None and np.asarray(c).ndim == 2 and np.asarray(c).shape[1] > 0


_KIND_TO_PROV = {"collided": PROV["collided"], "zfail": PROV["zfail"],
                 "coverage": PROV["coverage"], "photoz": PROV["photoz"],
                 "selfunc": PROV["selfunc"]}


def _anchor_pool(targets, anchor_frac):
    """Field neighbour pool for ``resample_redshift``: the lowest-error objects.

    The line-of-sight density prior must be built from the cleanest available
    redshifts. In resample mode every neighbour z is itself a photo-z, so seeding
    the prior from the raw photo-z point estimates would re-inject the very scatter
    we are trying to sharpen out. Instead keep only the lowest-``anchor_frac`` by
    ``photoz_sigma`` as "anchors" (the closest standin for a spectroscopic
    backbone) and seed the prior from their ``photoz_mu``."""
    sig = np.asarray(targets.photoz_sigma, np.float64)
    mu = np.asarray(targets.photoz_mu, np.float64)
    has_pz = np.isfinite(sig) & (sig > 0) & np.isfinite(mu)
    idx = np.where(has_pz)[0]
    n_anchor = max(int(round(anchor_frac * len(idx))), min(1000, len(idx)))
    anchor = idx[np.argsort(sig[idx])[:n_anchor]]
    return (np.asarray(targets.ra, np.float64)[anchor],
            np.asarray(targets.dec, np.float64)[anchor], mu[anchor])


def build_package(catalog, targets, photoz, *, mode="restore_missing", features_fn=None,
                  anchor_frac=0.30, photoz_err_scale=1.0, dz_pool=None, nq=65, ngrid=256,
                  K=150, K_zfail=20, dz_bg_frac=0.30, bw_f=0.004, bw_p=0.02,
                  jitter=None, verbose=False, copula=False, field_ctx=None, copula_modes=128):
    """Precompute the compact posterior package (the expensive step, done once).

    Two completion modes share this machinery (see :mod:`echoes.surveys.registry`):

    - ``mode="restore_missing"`` (BOSS, DESI): ``targets`` are the
      spectroscopically-missing objects to restore at their imaging positions. The
      observed catalogue is the fixed backbone; only the M missing redshifts vary.
      Each posterior is p(z|n̂,colours) ∝ (1+δ_g(n̂,z))·n̄(z)·p_photoz(z) (× close-pair
      prior for collisions).
    - ``mode="resample_redshift"`` (Quaia): every object is present with a fixed
      position but an uncertain redshift. ``targets`` is *every* object (self-hosted,
      via :func:`echoes.surveys.targets_generic.synth_resample_targets`); there is no
      fixed observed backbone (``n_obs=0``). Each posterior multiplies the per-object
      photo-z Gaussian N(``photoz_mu``, ``photoz_err_scale``·``photoz_sigma``) by the
      line-of-sight density prior, the latter seeded from the lowest-error anchors
      (:func:`_anchor_pool`). Target rows lacking a finite ``photoz_sigma`` (e.g.
      selection-function inpaint analogs) get a field-only posterior.

    ``features_fn(colors, mags) -> features`` maps the target photometry to the
    photo-z feature space; ``None`` uses the BOSS default (``photoz.photoz_features``,
    drops the u-band). If ``photoz`` is ``None`` or the targets carry no colours
    (DESI), the colour-space factor is dropped and the redshift posterior is
    **field-only** (``pp = 1``).

    ``copula=True`` additionally stores the field-correlation copula modes (from
    ``field_ctx.cov``, or a context built from ``catalog`` if not supplied) so
    :func:`draw` injects the coherent cross-object dependence the IID sampler lacks —
    marginals (hence calibration) unchanged (see :func:`_build_copula_modes`)."""
    if features_fn is None:
        from .photoz import photoz_features as features_fn
    resample = mode == "resample_redshift"

    host = np.asarray(targets.host_index)
    miss_kind = np.asarray(targets.miss_kind)
    M = len(host)

    if resample:
        # no fixed observed backbone: every object's redshift is resampled
        ra_o = np.empty(0); dec_o = np.empty(0); z_o = np.empty(0, np.float64); wsys_o = np.empty(0)
        z_host = np.full(M, np.nan)
        pz_mu = np.asarray(targets.photoz_mu, np.float64)
        pz_sig = photoz_err_scale * np.asarray(targets.photoz_sigma, np.float64)
        has_pz = np.isfinite(pz_sig) & (pz_sig > 0) & np.isfinite(pz_mu)
        pool_ra, pool_dec, pool_z = _anchor_pool(targets, anchor_frac)        # anchors seed the LOS prior
    else:
        ra_o = np.asarray(catalog.ra_data, np.float64)
        dec_o = np.asarray(catalog.dec_data, np.float64)
        z_o = np.asarray(catalog.z_data, np.float64)
        wsys_o = np.asarray(catalog.w_sys_data if catalog.w_sys_data is not None
                            else np.ones(len(ra_o)), np.float64)
        z_host = np.where(host >= 0, z_o[np.clip(host, 0, len(z_o) - 1)], np.nan)
        if (miss_kind == "collided").any():                                  # close-pair prior needed
            if dz_pool is None:
                dz_pool = measure_close_pair_dz(catalog)
            dz_pool = _broaden_close_pair_dz(np.asarray(dz_pool, np.float64), z_o, dz_bg_frac)
        else:                                                                # DESI coverage: no collisions
            dz_pool = np.zeros(1)
        pool_ra, pool_dec, pool_z = ra_o, dec_o, z_o                          # observed spec-z prior
    if jitter is None:
        jitter = bw_f * 0.5

    from scipy.spatial import cKDTree
    use_photoz = (not resample) and photoz is not None and _has_colours(targets)
    if use_photoz:
        feat = features_fn(targets.colors, targets.mags)
        zk, wk = photoz.posterior(feat)                  # (M, k)
    Kq = min(K, len(pool_z))
    _, nn = cKDTree(_radec_to_nhat(pool_ra, pool_dec)).query(
        _radec_to_nhat(np.asarray(targets.ra), np.asarray(targets.dec)), k=Kq, workers=-1)
    nn = np.atleast_2d(nn)
    if resample:
        zmin, zmax = float(np.nanmin(pz_mu[has_pz])), float(np.nanmax(pz_mu[has_pz]))
    else:
        zmin, zmax = float(z_o.min()), float(z_o.max())
    zgrid = np.linspace(zmin, zmax, ngrid)
    coll_i = (miss_kind == "collided") & (host >= 0)
    qlev = np.linspace(0.0, 1.0, nq)

    # Z-FAIL calibration: per-miss-kind field support. The local-field term pf is a KDE over the K
    # nearest observed redshifts (the LOS density); K=150 is too broad for z-fails → under-confident
    # PIT. Use only the ~K_zfail nearest for z-fails (per-neighbour weight 1/0); collided keep the full
    # K (their close-pair prior already pins z). (validation/pit_photoz_prototype.py.)
    K_use = np.where(miss_kind == "zfail", min(int(K_zfail), Kq), Kq).astype(np.int64)   # (M,)
    nbr_w = (np.arange(Kq)[None, :] < K_use[:, None]).astype(np.float64)                  # (M, Kq) 1/0
    znb_all = pool_z[nn]                                                                  # (M, Kq)

    # pf (local-field) and pp (photo-z) are Gaussian factors on zgrid — pf via _grid_kde (histogram +
    # gaussian_filter1d, ~10× over brute-force exp; see its docstring).
    dzg = (zmax - zmin) / (ngrid - 1)
    pf = _grid_kde(znb_all, nbr_w, zmin=zmin, dzg=dzg, ngrid=ngrid, bw=bw_f)               # (M, ngrid)
    if resample:
        # closed-form per-object Gaussian photo-z factor; selfunc inpaint rows (no σ) → flat field-only
        pp = np.ones((M, ngrid), np.float64)
        pp[has_pz] = np.exp(-0.5 * ((zgrid[None, :] - pz_mu[has_pz][:, None])
                                    / pz_sig[has_pz][:, None]) ** 2)
    elif use_photoz:
        ok_all = np.isfinite(wk) & (wk > 0)                                              # (M, k)
        pp = _grid_kde(np.where(ok_all, zk, zmin), np.where(ok_all, wk, 0.0),
                       zmin=zmin, dzg=dzg, ngrid=ngrid, bw=bw_p)                          # (M, ngrid)
        pp[~ok_all.any(1)] = 1.0                                                          # no photo-z → flat
    else:                                                                                # field-only (DESI)
        pp = np.ones((M, ngrid), np.float64)
    p = pf * pp
    if coll_i.any():                                                                      # close-pair prior
        pcl = _clpair_density(dz_pool)
        p[coll_i] *= pcl(zgrid[None, :] - z_host[coll_i][:, None])
    s = p.sum(1)
    cdf = np.maximum.accumulate(np.cumsum(p, axis=1) / np.where(s[:, None] > 0, s[:, None], 1.0), axis=1)

    invcdf = np.empty((M, nq), np.float64)
    fallback = s <= 0
    zfb = pz_mu if resample else z_host                  # per-object fallback central z
    zmed = float(np.median(pool_z)) if len(pool_z) else float(np.nanmedian(zfb))
    for i in range(M):
        if not fallback[i]:
            u, idx = np.unique(cdf[i], return_index=True)
            invcdf[i] = np.interp(qlev, u, zgrid[idx])
        else:
            invcdf[i] = zfb[i] if np.isfinite(zfb[i]) else zmed
    if verbose:
        print(f"[build_package] mode={mode} {M:,} posteriors, nq={nq}, n_obs={len(ra_o)}, "
              f"fallback={int(fallback.sum())}, K_zfail={K_zfail} dz_bg_frac={dz_bg_frac}")

    miss_prov = np.array([_KIND_TO_PROV.get(str(k), PROV["zfail"]) for k in miss_kind], np.int8)
    if not resample:
        miss_prov[fallback] = PROV["zhost"]              # fiber-collision host-z fallback
    miss_wsys = np.ones(M) if resample else wsys_o[np.clip(host, 0, len(z_o) - 1)]
    base_ra = np.concatenate([ra_o, np.asarray(targets.ra, np.float64)]).astype(np.float32)
    base_dec = np.concatenate([dec_o, np.asarray(targets.dec, np.float64)]).astype(np.float32)
    base_wsys = np.concatenate([wsys_o, miss_wsys]).astype(np.float32)
    base_prov = np.concatenate([np.full(len(ra_o), PROV["observed"], np.int8), miss_prov])

    pkg = {
        "n_obs": len(ra_o), "n_miss": M, "zmin": zmin, "zmax": zmax,
        "qlev": qlev.astype(np.float32), "jitter": float(jitter),
        "obs_z": z_o.astype(np.float32),                 # fixed observed redshifts (empty in resample)
        "base_ra": base_ra, "base_dec": base_dec,        # obs + missing positions (fixed)
        "base_wsys": base_wsys, "base_prov": base_prov,
        "base_mags": _assemble_base_mags(catalog, targets),   # (n_base,5) real ugriz, or None
        "invcdf": invcdf.astype(np.float32),             # (M, nq) per-object quantile fn
        "package_mode": mode,
    }
    if copula:
        _attach_copula(pkg, catalog, field_ctx, invcdf, qlev, copula_modes, verbose)
    return pkg


def _attach_copula(pkg, catalog, field_ctx, invcdf, qlev, copula_modes, verbose):
    """Build + attach the copula modes to ``pkg`` (no-op fallback to IID on failure)."""
    cov = getattr(field_ctx, "cov", None)
    if cov is None:
        try:
            from .fieldpost import build_field_context
            cov = build_field_context(catalog, verbose=verbose).cov
        except Exception as e:                                   # keep the package usable (IID)
            if verbose:
                print(f"[copula] no field kernel ({e}); shipping IID package")
            return
    cm, cd = _build_copula_modes(pkg["base_ra"], pkg["base_dec"], pkg["n_obs"],
                                 invcdf, qlev, cov, K=copula_modes, verbose=verbose)
    if cm is not None:
        pkg["cmodes"] = cm
        pkg["cdiag"] = cd


def build_package_generative(catalog, targets, photoz, gen_model, *, features_fn=None, dz_pool=None,
                             nq=65, ngrid=256, bw_p=0.02, dz_bg_frac=0.30, jitter=None,
                             verbose=False, copula=False, copula_modes=128):
    """Compact posterior package for the Tier-A generative engine.

    Identical layout to :func:`build_package` (same keys → the compact ``.npz`` and
    numpy-only ``data_release/draw_samples.py`` reproduce it with **zero changes**),
    but the per-object missing posterior is the **field-marginalized** generative one:

        p(z | n̂, colours) ∝ T(1+δ_post(n̂, z)) · n̄(z) · p_photoz(z)   (× close-pair),

    i.e. the fieldpost conditional posterior MEAN along each sightline (marginalised
    over field draws) pushed through the measured transform ``T`` — the same field
    the full engine samples, baked once into the inverse-CDF. Carries a
    ``package_engine='generative'`` tag (write/load preserve it, backward-compatibly).
    """
    from .fieldpost import los_overdensity
    if features_fn is None:
        from .photoz import photoz_features as features_fn

    ra_o = np.asarray(catalog.ra_data, np.float64)
    dec_o = np.asarray(catalog.dec_data, np.float64)
    z_o = np.asarray(catalog.z_data, np.float64)
    wsys_o = np.asarray(catalog.w_sys_data if catalog.w_sys_data is not None
                        else np.ones(len(ra_o)), np.float64)
    host = np.asarray(targets.host_index)
    z_host = np.where(host >= 0, z_o[np.clip(host, 0, len(z_o) - 1)], np.nan)
    miss_kind = np.asarray(targets.miss_kind)
    if dz_pool is None:
        dz_pool = measure_close_pair_dz(catalog)
    dz_pool = _broaden_close_pair_dz(np.asarray(dz_pool, np.float64), z_o, dz_bg_frac)  # COLLIDED fix
    if jitter is None:
        jitter = 0.002

    use_photoz = photoz is not None and _has_colours(targets)
    if use_photoz:
        feat = features_fn(targets.colors, targets.mags)
        zk, wk = photoz.posterior(feat)
    M = len(host)
    zmin, zmax = float(z_o.min()), float(z_o.max())
    zgrid = np.linspace(zmin, zmax, ngrid)
    pcl = _clpair_density(dz_pool)
    coll_i = (miss_kind == "collided") & (host >= 0)
    qlev = np.linspace(0.0, 1.0, nq)

    # field posterior MEAN along every missing sightline (marginalised over draws),
    # reshaped by the measured transform — the same field the full engine samples.
    fc = gen_model.field_ctx
    nbar_z = np.interp(zgrid, fc.z_centres, fc.nz_profile, left=0.0, right=0.0)
    opd_all = los_overdensity(fc, np.asarray(targets.ra, np.float64),
                              np.asarray(targets.dec, np.float64), zgrid)   # (M, ngrid)
    tf = gen_model.los_transform()
    if tf is not None:
        opd_all = tf(opd_all)

    # Vectorised (was a per-object Python loop). pf is the field × n̄(z) (already (M,ngrid)); pp is the
    # photo-z Gaussian KDE via _grid_kde (histogram + gaussian_filter1d, ~10× over brute-force exp).
    # NB the z-fail per-miss-kind-K fix does NOT apply here — pf is the generative FIELD, not a KNN.
    dzg = (zmax - zmin) / (ngrid - 1)
    pf = np.clip(opd_all, 0.0, None) * nbar_z[None, :]                                    # (M, ngrid)
    if use_photoz:
        ok_all = np.isfinite(wk) & (wk > 0)
        pp = _grid_kde(np.where(ok_all, zk, zmin), np.where(ok_all, wk, 0.0),
                       zmin=zmin, dzg=dzg, ngrid=ngrid, bw=bw_p)                          # (M, ngrid)
        pp[~ok_all.any(1)] = 1.0
    else:
        pp = np.ones((M, ngrid), np.float64)
    p = pf * pp
    if coll_i.any():
        p[coll_i] *= pcl(zgrid[None, :] - z_host[coll_i][:, None])
    s = p.sum(1)
    cdf = np.maximum.accumulate(np.cumsum(p, axis=1) / np.where(s[:, None] > 0, s[:, None], 1.0), axis=1)

    invcdf = np.empty((M, nq), np.float64)
    fallback = s <= 0
    zmed = float(np.median(z_o))
    for i in range(M):
        if not fallback[i]:
            u, idx = np.unique(cdf[i], return_index=True)
            invcdf[i] = np.interp(qlev, u, zgrid[idx])
        else:
            invcdf[i] = z_host[i] if np.isfinite(z_host[i]) else zmed
    if verbose:
        print(f"[build_package_generative] {M:,} posteriors, nq={nq}, "
              f"transform={gen_model.transform.kind}, fallback={int(fallback.sum())}, "
              f"dz_bg_frac={dz_bg_frac}")

    miss_prov = np.where(fallback, PROV["zhost"],
                         np.where(miss_kind == "collided", PROV["collided"], PROV["zfail"]))
    base_ra = np.concatenate([ra_o, np.asarray(targets.ra, np.float64)]).astype(np.float32)
    base_dec = np.concatenate([dec_o, np.asarray(targets.dec, np.float64)]).astype(np.float32)
    base_wsys = np.concatenate([wsys_o, wsys_o[np.clip(host, 0, len(z_o) - 1)]]).astype(np.float32)
    base_prov = np.concatenate([np.full(len(ra_o), PROV["observed"], np.int8), miss_prov.astype(np.int8)])

    pkg = {
        "n_obs": len(ra_o), "n_miss": M, "zmin": zmin, "zmax": zmax,
        "qlev": qlev.astype(np.float32), "jitter": float(jitter),
        "obs_z": z_o.astype(np.float32),
        "base_ra": base_ra, "base_dec": base_dec,
        "base_wsys": base_wsys, "base_prov": base_prov,
        "base_mags": _assemble_base_mags(catalog, targets),
        "invcdf": invcdf.astype(np.float32),
        "package_engine": "generative",
        "package_mode": "restore_missing",
    }
    if copula:
        _attach_copula(pkg, catalog, gen_model.field_ctx, invcdf, qlev, copula_modes, verbose)
    return pkg


def _quant(a, lo, hi):
    return np.clip(np.round((a - lo) / (hi - lo) * 65535.0), 0, 65535).astype(np.uint16)


def _dequant(u, lo, hi):
    return lo + u.astype(np.float32) / 65535.0 * (hi - lo)


def write_package(pkg, path):
    """Heavily-compressed serialization (.npz, uint16-quantised redshifts/CDF)."""
    zmin, zmax = pkg["zmin"], pkg["zmax"]
    extra = {}
    if pkg.get("cmodes") is not None:                                  # copula low-rank factor
        extra["cmodes"] = pkg["cmodes"].astype(np.float16)            # (M, K)
        extra["cdiag"] = pkg["cdiag"].astype(np.float16)             # (M,)
    if pkg.get("base_mags") is not None:                               # fixed ugriz photometry
        extra["base_mags_q"] = _quant_mags(pkg["base_mags"])          # uint16, per-band NaN sentinel
        extra["mag_lo"] = MAG_LO
        extra["mag_hi"] = MAG_HI
    np.savez_compressed(
        path,
        version=QUANT_VERSION, n_obs=pkg["n_obs"], n_miss=pkg["n_miss"],
        zmin=zmin, zmax=zmax, jitter=pkg["jitter"], qlev=pkg["qlev"].astype(np.float32),
        obs_z_q=_quant(pkg["obs_z"], zmin, zmax),                       # uint16
        invcdf_q=_quant(pkg["invcdf"], zmin, zmax),                     # uint16 (M, nq)
        base_ra=pkg["base_ra"], base_dec=pkg["base_dec"],              # float32 positions
        base_wsys=pkg["base_wsys"].astype(np.float16),
        base_prov=pkg["base_prov"],
        package_engine=str(pkg.get("package_engine", "field")),
        package_mode=str(pkg.get("package_mode", "restore_missing")),
        **extra,
    )


def load_package(path):
    d = np.load(path if str(path).endswith(".npz") else str(path) + ".npz")
    zmin, zmax = float(d["zmin"]), float(d["zmax"])
    pkg = {
        "n_obs": int(d["n_obs"]), "n_miss": int(d["n_miss"]), "zmin": zmin, "zmax": zmax,
        "qlev": d["qlev"].astype(np.float64), "jitter": float(d["jitter"]),
        "obs_z": _dequant(d["obs_z_q"], zmin, zmax),
        "invcdf": _dequant(d["invcdf_q"], zmin, zmax).astype(np.float64),
        "base_ra": d["base_ra"], "base_dec": d["base_dec"],
        "base_wsys": d["base_wsys"].astype(np.float32), "base_prov": d["base_prov"],
        "package_engine": str(d["package_engine"]) if "package_engine" in d.files else "field",
        "package_mode": str(d["package_mode"]) if "package_mode" in d.files else "restore_missing",
    }
    if "cmodes" in d.files:
        pkg["cmodes"] = d["cmodes"].astype(np.float32)
        pkg["cdiag"] = d["cdiag"].astype(np.float32)
    pkg["base_mags"] = None
    if "base_mags_q" in d.files:
        pkg["base_mags"] = _dequant_mags(d["base_mags_q"], float(d["mag_lo"]), float(d["mag_hi"]))
    return pkg


def draw(pkg, seed=0, *, systot=True, copula=None):
    """Draw one equal-weight completed realization from the package (fast).

    Returns ``dict(ra, dec, z, prov)``. Equivalent in distribution to
    ``complete_catalog_photoz(..., z_mode='field')``. With ``systot=False`` the
    WEIGHT_SYSTOT analog excess is skipped (just observed + missing).

    ``copula`` selects the missing-redshift dependence: ``None`` (default) uses the
    field-correlation copula iff the package carries copula modes, else IID; ``True``
    forces it (errors if absent); ``False`` forces the legacy IID draw (reproduces
    pre-copula catalogs bit-for-bit). The copula leaves every per-object marginal —
    hence per-object PIT calibration — unchanged; it only adds the coherent
    cross-object dependence that the IID draw under-disperses."""
    rng = np.random.default_rng(seed)
    n_obs, M = pkg["n_obs"], pkg["n_miss"]
    qlev, invcdf = pkg["qlev"], pkg["invcdf"]
    nq = len(qlev)

    # missing-redshift uniforms: field-correlation copula (Φ of a correlated Gaussian
    # with EXACT unit marginal variance ⇒ marginals/PIT unchanged) or legacy IID.
    has_modes = pkg.get("cmodes") is not None
    use_copula = has_modes if copula is None else bool(copula)
    if use_copula and not has_modes:
        raise ValueError("copula=True but the package carries no copula modes")
    if use_copula:
        cm, cd = pkg["cmodes"], pkg["cdiag"]
        g = cm @ rng.standard_normal(cm.shape[1]) + cd * rng.standard_normal(M)
        u = _phi(g)
    else:
        u = rng.random(M)
    j = np.clip(np.searchsorted(qlev, u), 1, nq - 1)
    q0, q1 = qlev[j - 1], qlev[j]
    v0 = invcdf[np.arange(M), j - 1]; v1 = invcdf[np.arange(M), j]
    z_miss = v0 + (v1 - v0) * (u - q0) / np.maximum(q1 - q0, 1e-12)
    z_miss = z_miss + rng.normal(0.0, pkg["jitter"], M)
    z_miss = np.clip(z_miss, pkg["zmin"], pkg["zmax"])

    base_ra, base_dec = pkg["base_ra"], pkg["base_dec"]
    base_z = np.concatenate([pkg["obs_z"], z_miss]).astype(np.float32)
    base_prov = pkg["base_prov"]
    base_mags = pkg.get("base_mags")                       # fixed ugriz (n_base,5) or None
    if not systot:
        return {"ra": base_ra.copy(), "dec": base_dec.copy(), "z": base_z, "prov": base_prov.copy(),
                "N": len(base_ra), **_mag_cols(base_mags)}

    wsys = pkg["base_wsys"]
    n_extra = np.floor(np.maximum(wsys - 1.0, 0.0) + rng.random(len(wsys))).astype(int)
    src = np.repeat(np.arange(len(base_ra)), n_extra)
    ex_ra, ex_dec, ex_z = _systot_restore_extras(
        base_ra.astype(np.float64), base_dec.astype(np.float64), base_z.astype(np.float64), src, rng)
    # systot extras inherit the SOURCE galaxy's photometry (copies base_mags[src], like base_z) —
    # consumes no RNG, so z/ra/dec stay byte-identical to the no-photometry draw.
    out_mags = np.concatenate([base_mags, base_mags[src]]) if base_mags is not None else None
    return {
        "ra": np.concatenate([base_ra, ex_ra]).astype(np.float32),
        "dec": np.concatenate([base_dec, ex_dec]).astype(np.float32),
        "z": np.concatenate([base_z, ex_z]).astype(np.float32),
        "prov": np.concatenate([base_prov, np.full(len(ex_ra), PROV["systot"], np.int8)]),
        "N": len(base_ra) + len(ex_ra),
        **_mag_cols(out_mags),
    }
