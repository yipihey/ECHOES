"""Experimental kNN2D redshift-completion field (Yuan, Abel & Wechsler 2024).

The third ECHOES redshift engine. Where ``z_mode='field'`` estimates the
line-of-sight density along a missing galaxy's sightline from a KDE of its K
nearest observed spec-z, and ``z_mode='graphgp'`` evaluates a conditional
Matheron GP density field there, this engine builds the local density from the
**2D angular kNN statistic** measured in pure observables (Δθ, z):

    (1 + δ)(n̂, z) = DD(n̂; θ, z) / RD(θ, z)            (Davis–Peebles, local)

- ``DD(n̂; θ, z)`` is the *per-sightline* neighbour profile: the count of
  observed galaxies within an angular cap of radius θ around the missing
  galaxy's known imaging position n̂, resolved into neighbour-redshift shells
  z. The missing galaxy's own (unknown) redshift never enters — only its
  neighbours' redshifts. Computed with the per-cap kernel
  :func:`echoes.knn._kernels._per_cap_count_kernel` (the kNN ladder θ).
- ``RD(θ, z)`` is the per-redshift *window expectation*: the mean count of
  observed galaxies in the same cap around a **random** footprint position —
  i.e. the no-clustering, selection-corrected normalisation, measured once
  globally (MC random queries, or the analytic separable-window form) in the
  regions the data actually cover. This is exactly the Yuan–Abel–Wechsler
  RD/DD construction applied locally, per missing sightline.

The resulting ``(1+δ)(n̂, z)`` replaces the KNN-KDE / GP local density in the
**same** posterior product used by the other two engines:

    p(z | n̂, colours) ∝ (1+δ)(n̂, z) · n̄(z) · p_photoz(z)   (× close-pair prior)

so the engine is a drop-in third path: cosmology-free (only (θ, z) observables),
evaluated identically, and self-closing (re-measure the kNN-CDF on the completed
catalog and recover the input — see ``validation/knn2d_closure.py``).

If it works, it continues the Banerjee & Abel nearest-neighbour series.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .geometry import _radec_to_nhat


@dataclass
class KNN2DFieldResult:
    """Measured kNN2D normalisation + observed-galaxy lookup for the engine.

    Built once by :func:`build_knn2d_field` and reused across an ensemble of
    completion seeds (the field measurement is deterministic; realization-to-
    realization variation comes from the per-object redshift draw, exactly as
    in ``z_mode='field'``).

    Attributes
    ----------
    theta_radii_rad : ``(n_theta,)``
        Ascending angular cap half-angles [rad] — the kNN ladder.
    z_n_edges, z_n_centres
        Neighbour-redshift shell edges / centres for the density profile.
    rd_cum : ``(n_theta, n_z_n)``
        Window-expectation cumulative cap count of observed galaxies per
        ``(theta, z_n)`` around a random footprint position (the RD
        normalisation). ``backend`` records how it was measured.
    aperture_index : int
        Index into ``theta_radii_rad`` of the aperture used for the
        ``reduce='aperture'`` density reduction.
    reduce : str
        ``'aperture'`` (single cap) or ``'ladder'`` (pooled over the θ ladder).
    bw_z : float
        Gaussian bandwidth [in z] for smoothing the per-sightline DD and the RD
        normalisation along the line of sight before forming ``(1+δ)``. The
        fine z_n shells are individually shot-noise dominated (a single cap holds
        ~hundreds of galaxies spread over the whole redshift range); smoothing
        pools adjacent shells into a stable local density profile, exactly as the
        KNN-KDE ``z_mode='field'`` engine KDE-smooths its 150 nearest spec-z.
    min_expected : float
        Below this RD cap count a ``(theta, z_n)`` cell is treated as
        not-well-covered; the local overdensity falls back to neutral
        ``(1+δ)=1`` (use n̄(z) only) there.
    nside_lookup : int
        HEALPix NSIDE of the observed-galaxy lookup grid.
    pix_starts, theta_g_sorted, phi_g_sorted, z_g_sorted, w_g_sorted
        Pixel-sorted observed-galaxy arrays feeding the per-cap kernel.
    n_obs : int
        Number of observed galaxies in the lookup.
    n_samples : int
        Field realizations (always 1 — see class docstring); kept for API
        symmetry with ``DensityFieldResult`` so the completion wiring is
        uniform across engines.
    backend : str
        ``'mc'`` or ``'analytic'`` — how ``rd_cum`` was measured.
    """

    theta_radii_rad: np.ndarray
    z_n_edges: np.ndarray
    z_n_centres: np.ndarray
    rd_cum: np.ndarray
    aperture_index: int
    reduce: str
    bw_z: float
    min_expected: float
    nside_lookup: int
    pix_starts: np.ndarray
    theta_g_sorted: np.ndarray
    phi_g_sorted: np.ndarray
    z_g_sorted: np.ndarray
    w_g_sorted: np.ndarray
    n_obs: int
    n_samples: int = 1
    backend: str = "mc"

    @property
    def n_theta(self) -> int:
        return self.theta_radii_rad.size

    @property
    def n_z_n(self) -> int:
        return self.z_n_centres.size


def _build_lookup(ra_deg, dec_deg, z, w, nside_lookup):
    """Pixel-sort the neighbour catalog for the per-cap kernel (mirrors the
    lookup grid built inside ``joint_knn_cdf``)."""
    import healpy as hp
    theta_g = np.deg2rad(90.0 - np.asarray(dec_deg, np.float64))
    phi_g = np.deg2rad(np.asarray(ra_deg, np.float64) % 360.0)
    ipix = hp.ang2pix(nside_lookup, theta_g, phi_g)
    order = np.argsort(ipix, kind="stable")
    ipix_s = ipix[order]
    npix = 12 * nside_lookup ** 2
    pix_starts = np.searchsorted(
        ipix_s, np.arange(npix + 1), side="left").astype(np.int64)
    return (pix_starts,
            np.ascontiguousarray(theta_g[order]),
            np.ascontiguousarray(phi_g[order]),
            np.ascontiguousarray(np.asarray(z, np.float64)[order]),
            np.ascontiguousarray(np.asarray(w, np.float64)[order]))


def _per_sightline_dd(field, ra_deg, dec_deg, n_threads=None):
    """Per-sightline cumulative DD profile ``DD(n̂; θ, z_n)`` for an array of
    query positions — one per-cap kernel call per sightline, parallelised over
    sightlines (the kernel is nogil). Returns ``(M, n_theta, n_z_n)``."""
    import healpy as hp
    from .knn._kernels import _per_cap_count_kernel
    ra_deg = np.asarray(ra_deg, np.float64)
    dec_deg = np.asarray(dec_deg, np.float64)
    theta_q = np.deg2rad(90.0 - dec_deg)
    phi_q = np.deg2rad(ra_deg % 360.0)
    vecs_q = hp.ang2vec(theta_q, phi_q)
    theta_max = float(field.theta_radii_rad.max())
    M = ra_deg.size
    out = np.zeros((M, field.n_theta, field.n_z_n), dtype=np.float64)

    def _one(i):
        ipix = hp.query_disc(field.nside_lookup, vecs_q[i], theta_max,
                             inclusive=True).astype(np.int64)
        if ipix.size == 0:
            return i, None
        return i, _per_cap_count_kernel(
            theta_q[i], phi_q[i], ipix, field.pix_starts,
            field.theta_g_sorted, field.phi_g_sorted,
            field.z_g_sorted, field.w_g_sorted,
            field.theta_radii_rad, field.z_n_edges)

    if n_threads is None:
        import os
        n_threads = os.cpu_count() or 1
    if n_threads > 1 and M > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            for i, mat in pool.map(_one, range(M)):
                if mat is not None:
                    out[i] = mat
    else:
        for i in range(M):
            _, mat = _one(i)
            if mat is not None:
                out[i] = mat
    return out


def _z_smoothing_matrix(z_centres, bw_z):
    """Row-normalised Gaussian smoothing matrix ``S[i, j] ∝ exp(-½((z_i-z_j)/bw)²)``
    over the neighbour-redshift shells. ``(profile @ S.T)[i]`` is the smoothed
    value at shell ``i`` — pools adjacent shells so per-cap counts are not
    individually shot-noise dominated."""
    d = (z_centres[:, None] - z_centres[None, :]) / bw_z
    S = np.exp(-0.5 * d * d)
    S /= S.sum(axis=1, keepdims=True)
    return S


def _one_plus_delta(dd_cum, field):
    """Reduce a per-sightline cumulative DD profile ``(…, n_theta, n_z_n)`` and
    the stored RD normalisation to a local overdensity ``(1+δ)(n̂, z_n)`` of
    shape ``(…, n_z_n)``.

    Both DD and RD are Gaussian-smoothed along z (bandwidth ``field.bw_z``)
    before the Davis–Peebles ratio — a single cap holds only a handful of
    galaxies per fine z_n shell, so the unsmoothed ratio is pure shot noise. The
    overdensity is held neutral (==1, i.e. fall back to n̄(z) only) where the
    smoothed RD is below ``min_expected`` (line-of-sight regions the survey does
    not actually cover well)."""
    S = _z_smoothing_matrix(field.z_n_centres, field.bw_z)     # (n_z_n, n_z_n)
    rd = field.rd_cum                                          # (n_theta, n_z_n)
    if field.reduce == "ladder":
        # pool the whole kNN ladder: Σ_θ DD / Σ_θ RD (count-weighted).
        dd = dd_cum.sum(axis=-2)                               # (…, n_z_n)
        rdr = rd.sum(axis=0)                                   # (n_z_n,)
    else:                                                      # 'aperture'
        t = field.aperture_index
        dd = dd_cum[..., t, :]                                 # (…, n_z_n)
        rdr = rd[t]                                            # (n_z_n,)
    dd = dd @ S.T                                              # smooth along z
    rdr = S @ rdr
    covered = rdr >= field.min_expected
    safe = np.where(covered, rdr, np.inf)
    opd = dd / safe                                            # 0 where uncovered (rdr→inf)
    # neutral fallback where not well-covered: use n̄(z) only (1+δ=1).
    opd = np.where(covered[(None,) * (opd.ndim - 1) + (slice(None),)], opd, 1.0)
    return opd


def build_knn2d_field(
    catalog,
    *,
    theta_edges_deg: Optional[np.ndarray] = None,
    n_z_n: int = 48,
    z_range: Optional[tuple] = None,
    aperture_deg: float = 1.0,
    reduce: str = "aperture",
    bw_z: float = 0.008,
    rd_source: str = "mc",
    n_rd_factor: int = 4,
    min_expected: float = 1.0,
    nside_lookup: int = 512,
    sel_map: Optional[np.ndarray] = None,
    nside: Optional[int] = None,
    n_samples: int = 1,
    seed: int = 0,
    verbose: bool = False,
):
    """Measure the kNN2D normalisation field once for an ensemble of seeds.

    Builds the observed-galaxy per-cap lookup and the window-expectation RD
    profile ``RD(θ, z_n)``; reusable across many :func:`complete_catalog_photoz`
    calls (pass as ``knn2d_field=``) to amortise the RD measurement.

    Parameters
    ----------
    catalog
        ECHOES catalog (needs ``ra_data``, ``dec_data``, ``z_data``; and
        ``sel_map``/``nside`` for the RD window expectation, unless passed
        explicitly via ``sel_map=``/``nside=`` — e.g. for a mock-observed
        subset that shares the survey footprint but does not carry the mask).
    theta_edges_deg
        Angular cap radii [deg] defining the kNN ladder; default
        ``geomspace(0.03, 0.3, 7)``.
    n_z_n, z_range
        Neighbour-redshift shells: ``n_z_n`` uniform bins over ``z_range``
        (default the observed redshift span).
    aperture_deg, reduce
        Density reduction over the θ ladder — single cap nearest
        ``aperture_deg`` (``reduce='aperture'``, default ~1° so the cap holds
        ~150 galaxies, matching the KNN-KDE engine's K) or count-weighted pool
        over the whole ladder (``reduce='ladder'``).
    bw_z
        Gaussian z-bandwidth for smoothing DD/RD before the ratio (default
        0.008, between the KNN-KDE field bandwidth 0.004 and the photo-z 0.02).
    rd_source
        ``'mc'`` (default): RD from ``n_rd_factor·N_data`` random footprint
        queries vs the observed galaxies (captures the true window). ``'analytic'``:
        separable-window form ``N_data(z_n)·⟨A_cap(θ)⟩/Ω_footprint`` (fast; exact
        for a binary mask, e.g. CMASS-South COMP≈0.99).
    min_expected
        RD cap-count floor below which a cell is not-well-covered (neutral
        fallback ``(1+δ)=1``).
    nside_lookup
        HEALPix NSIDE for the neighbour lookup (pixels ≪ smallest cap).

    Returns
    -------
    KNN2DFieldResult
    """
    ra_o = np.asarray(catalog.ra_data, np.float64)
    dec_o = np.asarray(catalog.dec_data, np.float64)
    z_o = np.asarray(catalog.z_data, np.float64)
    n_obs = ra_o.size

    if theta_edges_deg is None:
        theta_edges_deg = np.geomspace(0.05, 1.5, 8)
    theta_radii_rad = np.deg2rad(np.asarray(theta_edges_deg, np.float64))
    if z_range is None:
        z_range = (float(z_o.min()), float(z_o.max()))
    z_n_edges = np.linspace(z_range[0], z_range[1], n_z_n + 1)
    z_n_centres = 0.5 * (z_n_edges[1:] + z_n_edges[:-1])
    aperture_index = int(np.argmin(np.abs(theta_radii_rad
                                          - np.deg2rad(aperture_deg))))

    # observed-galaxy lookup for the per-sightline DD kernel (uniform weights:
    # the local density field is the equal-weight observed spec-z field, as in
    # z_mode='field').
    w_o = np.ones(n_obs, np.float64)
    pix_starts, tg_s, pg_s, zg_s, wg_s = _build_lookup(
        ra_o, dec_o, z_o, w_o, nside_lookup)

    # RD(θ, z_n): window expectation of observed-galaxy cap counts. The footprint
    # mask may be supplied explicitly (e.g. a mock-observed subset) or read off
    # the catalog.
    if sel_map is None:
        sel_map = getattr(catalog, "sel_map", None)
    if nside is None:
        nside = getattr(catalog, "nside", None)
    if sel_map is None:
        raise ValueError(
            "build_knn2d_field needs the survey footprint for the RD "
            "normalisation — pass sel_map= (and nside=) or use a catalog "
            "with a .sel_map attribute.")
    sel_map = np.asarray(sel_map)
    rng = np.random.default_rng(seed)
    if rd_source == "analytic":
        from .knn.analytic_rr import analytic_rr_cube
        z_q_edges = np.array([z_range[0], z_range[1]])
        res = analytic_rr_cube(
            sel_map=sel_map, z_data=z_o,
            theta_radii_rad=theta_radii_rad,
            z_q_edges=z_q_edges, z_n_edges=z_n_edges,
            n_q_per_shell=np.array([1], dtype=np.int64),
            n_random_total=n_obs, nside=nside)
        rd_cum = res.sum_n[:, 0, :]                            # (n_theta, n_z_n)
    else:                                                      # 'mc'
        from .knn import joint_knn_cdf
        from .randoms import make_random_from_selection_function
        n_rd = int(n_rd_factor * n_obs)
        ra_r, dec_r, z_r = make_random_from_selection_function(
            sel_map=sel_map, n_random=n_rd, z_data=z_o, nside=nside, rng=rng)
        z_q_edges = np.array([z_range[0], z_range[1]])
        res = joint_knn_cdf(
            np.asarray(ra_r, np.float64), np.asarray(dec_r, np.float64),
            np.asarray(z_r, np.float64), ra_o, dec_o, z_o,
            theta_radii_rad, z_q_edges, z_n_edges, k_max=0,
            flavor="RD", nside_lookup=nside_lookup)
        from .knn import derived
        rd_cum = derived.mean_count(res)[:, 0, :]              # (n_theta, n_z_n)

    if verbose:
        S = _z_smoothing_matrix(z_n_centres, bw_z)
        rdr_sm = (rd_cum.sum(0) if reduce == "ladder"
                  else rd_cum[aperture_index]) @ S.T
        cov = float((rdr_sm >= min_expected).mean())
        print(f"[knn2d-field] N_obs={n_obs:,} theta={len(theta_radii_rad)} "
              f"z_n={n_z_n} rd_source={rd_source} reduce={reduce} "
              f"aperture={aperture_deg}deg bw_z={bw_z} "
              f"<cap RD>={float(np.median(rdr_sm)):.1f} well-covered shells="
              f"{cov*100:.0f}%")

    return KNN2DFieldResult(
        theta_radii_rad=theta_radii_rad,
        z_n_edges=z_n_edges, z_n_centres=z_n_centres,
        rd_cum=np.ascontiguousarray(rd_cum),
        aperture_index=aperture_index, reduce=reduce, bw_z=float(bw_z),
        min_expected=float(min_expected), nside_lookup=nside_lookup,
        pix_starts=pix_starts, theta_g_sorted=tg_s, phi_g_sorted=pg_s,
        z_g_sorted=zg_s, w_g_sorted=wg_s, n_obs=n_obs,
        n_samples=int(max(1, n_samples)), backend=rd_source)


def _knn2d_zmiss(targets, photoz, dz_pool, knn2d_field, draw_index,
                 z_o, z_host, miss_kind, rng):
    """Missing-galaxy redshifts from the 2D-kNN local density field, evaluated
    along each missing galaxy's sightline:

        p(z | n̂, colours) ∝ (1+δ_kNN(n̂, z)) · n̄(z) · p_photoz(z)   (× close-pair)

    i.e. ``z_mode='field'`` with the Yuan–Abel–Wechsler RD/DD local overdensity
    replacing the KNN-KDE local density. Mirrors :func:`completion._graphgp_zmiss`
    exactly. Returns ``(z_miss, zhost_fallback)``.
    """
    from .photoz import photoz_features
    from .completion import _clpair_density
    field = knn2d_field
    ra_m = np.asarray(targets.ra, np.float64)
    dec_m = np.asarray(targets.dec, np.float64)
    host = np.asarray(targets.host_index)
    coll = (miss_kind == "collided") & (host >= 0)
    feat = photoz_features(targets.colors, targets.mags)
    zk, wk = photoz.posterior(feat)
    pcl = _clpair_density(dz_pool)

    zc = field.z_n_centres
    zgrid = np.linspace(z_o.min(), z_o.max(), 256)
    nbar_z = np.interp(zgrid, zc,
                       np.histogram(z_o, bins=field.z_n_edges)[0].astype(float),
                       left=0.0, right=0.0)

    # per-sightline DD profile -> local overdensity (1+δ)(n̂, z_n).
    dd_cum = _per_sightline_dd(field, ra_m, dec_m)             # (M, n_theta, n_z_n)
    opd_zn = _one_plus_delta(dd_cum, field)                   # (M, n_z_n)

    bw_p = 0.02
    M = len(ra_m)
    z_miss = np.empty(M)
    fb = np.zeros(M, bool)
    for i in range(M):
        # kNN local density × n̄(z) along this sightline.
        pf = np.interp(zgrid, zc, opd_zn[i], left=0.0, right=0.0) * nbar_z
        w = wk[i]; ok = np.isfinite(w) & (w > 0)
        pp = ((w[ok][None, :] * np.exp(-0.5 * ((zgrid[:, None]
              - zk[i][ok][None, :]) / bw_p) ** 2)).sum(1)
              if ok.any() else np.ones_like(zgrid))
        p = pf * pp
        if coll[i]:
            p = p * pcl(zgrid - z_host[i])
        s = p.sum()
        if s > 0:
            z_miss[i] = rng.choice(zgrid, p=p / s)
        else:
            z_miss[i] = z_host[i] if np.isfinite(z_host[i]) else float(np.median(z_o))
            fb[i] = True
    return z_miss, fb


def build_knn2d_field_from_catalog(catalog, *, n_samples=1, seed=0,
                                   verbose=False, **kwargs):
    """Convenience builder for ``z_mode='knn2d'`` (parallels
    :func:`completion.build_gp_field`). Build ONCE and pass as
    ``knn2d_field=`` to :func:`complete_catalog_photoz`."""
    return build_knn2d_field(catalog, n_samples=n_samples, seed=seed,
                             verbose=verbose, **kwargs)
