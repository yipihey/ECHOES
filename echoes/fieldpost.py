"""Field-level conditional redshift completion (``z_mode='fieldpost'``).

The third-generation ECHOES redshift engine: instead of a local-density estimate
along each missing sightline, it evaluates the **proper conditional posterior of
the galaxy overdensity field** there, conditioned on the nearby observed galaxies
through the log-Gaussian-Cox-process linearization (:func:`echoes.field_posterior.
conditional_overdensity_los`) with the ξ-tabulated graphGP kernel. The missing
redshift is then drawn from

    p(z | n̂, colours) ∝ (1 + δ_post(n̂, z)) · n̄(z) · p_photoz(z)   (× close-pair),

with ``1+δ_post`` the field posterior mean — which carries the field's full
correlation structure and reverts toward unity (via the kernel) in data-poor
stretches, rather than the fixed-aperture / K-nearest density of the earlier
engines. The selection model (:mod:`echoes.selection_model`) supplies the
density-coupled missingness and the imaging photo-z likelihood.

``build_field_context`` measures ξ(r) → kernel once and prepares the
observed-galaxy comoving positions + angular lookup; ``_fieldpost_zmiss`` mirrors
``completion._graphgp_zmiss`` so it drops into ``complete_catalog_photoz``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .geometry import _radec_to_nhat
from .clustering import comoving_mpc_h
from .field_posterior import conditional_overdensity_los


@dataclass
class FieldContext:
    """Everything the fieldpost engine needs, built once per catalog."""
    x_obs: np.ndarray            # (N, 3) observed galaxy comoving positions [Mpc/h]
    nhat_obs: np.ndarray         # (N, 3) observed angular unit vectors
    cov: tuple                   # (cov_bins, cov_vals) ξ-tabulated graphGP kernel
    nbar: float                  # mean comoving number density [ (Mpc/h)^-3 ]
    z_centres: np.ndarray        # n(z) profile grid
    nz_profile: np.ndarray       # n(z) counts on z_centres
    neigh_chord: float           # angular cylinder radius (chord) for LOS neighbours
    max_neigh: int               # cap on neighbours per sightline (dense tractability)
    n_samples: int = 1
    _tree: object = None         # cKDTree on nhat_obs (lazy)

    def tree(self):
        if self._tree is None:
            from scipy.spatial import cKDTree
            self._tree = cKDTree(self.nhat_obs)
        return self._tree


def build_field_context(
    catalog,
    *,
    r_edges: Optional[np.ndarray] = None,
    n_rand_factor: int = 3,
    neigh_radius_deg: float = 1.2,
    max_neigh: int = 250,
    n_samples: int = 1,
    seed: int = 0,
    nthreads: int = 8,
    sel_map: Optional[np.ndarray] = None,
    nside: Optional[int] = None,
    verbose: bool = False,
):
    """Measure ξ(r) → kernel and assemble the observed-galaxy context for the
    fieldpost engine. ``neigh_radius_deg`` is the angular cylinder around each
    missing sightline whose observed galaxies condition the field."""
    from .field_kernel import tabulate_kernel
    from .ls_corrfunc import xi_landy_szalay
    from .randoms import make_random_from_selection_function

    ra = np.asarray(catalog.ra_data, np.float64)
    dec = np.asarray(catalog.dec_data, np.float64)
    z = np.asarray(catalog.z_data, np.float64)
    nhat = _radec_to_nhat(ra, dec)
    chi = comoving_mpc_h(z)
    x_obs = chi[:, None] * nhat                                   # comoving Mpc/h
    N = len(ra)

    if sel_map is None:
        sel_map = getattr(catalog, "sel_map", None)
    if nside is None:
        nside = getattr(catalog, "nside", None)
    if r_edges is None:
        r_edges = np.logspace(np.log10(1.0), np.log10(60.0), 36)
    rng = np.random.default_rng(seed)
    rar, decr, zr = make_random_from_selection_function(
        sel_map=sel_map, n_random=n_rand_factor * N, z_data=z, nside=nside, rng=rng)
    x_rand = comoving_mpc_h(zr)[:, None] * _radec_to_nhat(np.asarray(rar), np.asarray(decr))
    r_centers, xi_j = xi_landy_szalay(x_obs, x_rand, r_edges=r_edges, nthreads=nthreads)[:2]
    cov, _kfit = tabulate_kernel(r_centers, xi_j)        # (cov_bins, cov_vals), (A,r0,alpha)
    cov = (np.asarray(cov[0], np.float64), np.asarray(cov[1], np.float64))

    # mean comoving number density: N / (footprint solid angle × radial shell volume).
    omega = (4.0 * np.pi * (np.asarray(sel_map) > 0).mean()
             if sel_map is not None else 4.0 * np.pi)
    vol = omega / 3.0 * (chi.max() ** 3 - chi.min() ** 3)
    nbar = float(N / max(vol, 1.0))

    nz_grid = np.linspace(z.min(), z.max(), 64)
    nz = np.histogram(z, bins=np.linspace(z.min(), z.max(), 65))[0].astype(float)
    neigh_chord = 2.0 * np.sin(np.radians(neigh_radius_deg) / 2.0)

    if verbose:
        print(f"[fieldpost] N={N:,}  ξ(r) over {len(r_centers)} bins, K(0)={cov[1][0]:.3f}  "
              f"nbar={nbar:.3e} (Mpc/h)^-3  neigh<{neigh_radius_deg}deg cap {max_neigh}")
    return FieldContext(x_obs=x_obs, nhat_obs=nhat, cov=cov, nbar=nbar,
                        z_centres=nz_grid, nz_profile=nz, neigh_chord=neigh_chord,
                        max_neigh=max_neigh, n_samples=int(max(1, n_samples)))


def los_overdensity(field_ctx, ra_m, dec_m, zgrid):
    """Conditional field overdensity ``1+δ_post(n̂, z)`` along each sightline.

    For each angular position ``(ra_m, dec_m)``, gathers the observed galaxies in
    the angular cylinder and conditions the GP field at the comoving points along
    the sightline (``zgrid``). Returns ``(M, n_z)``; rows with too few neighbours
    are 1 (neutral). The reusable core of the fieldpost engine — used both to
    sample redshifts and to score the per-galaxy posterior."""
    fc = field_ctx
    nhat_m = _radec_to_nhat(np.asarray(ra_m, np.float64), np.asarray(dec_m, np.float64))
    chi_grid = comoving_mpc_h(np.asarray(zgrid, np.float64))
    tree = fc.tree()
    M = len(nhat_m)
    opd = np.ones((M, len(zgrid)), np.float64)
    for i in range(M):
        idx = tree.query_ball_point(nhat_m[i], fc.neigh_chord)
        if len(idx) > fc.max_neigh:
            _, idx = tree.query(nhat_m[i], k=fc.max_neigh); idx = np.asarray(idx)
        if len(idx) >= 6:
            x_pred = chi_grid[:, None] * nhat_m[i][None, :]
            opd[i], _ = conditional_overdensity_los(fc.x_obs[np.asarray(idx)], fc.nbar,
                                                    x_pred, fc.cov)
    return opd


def _fieldpost_zmiss(targets, photoz, dz_pool, field_ctx, draw_index,
                     z_o, z_host, miss_kind, rng):
    """Missing-galaxy redshifts from the conditional field posterior along each
    sightline. Mirrors :func:`completion._graphgp_zmiss`."""
    from .photoz import photoz_features
    from .completion import _clpair_density
    fc = field_ctx
    ra_m = np.asarray(targets.ra, np.float64)
    dec_m = np.asarray(targets.dec, np.float64)
    nhat_m = _radec_to_nhat(ra_m, dec_m)
    host = np.asarray(targets.host_index)
    coll = (miss_kind == "collided") & (host >= 0)
    feat = photoz_features(targets.colors, targets.mags)
    zk, wk = photoz.posterior(feat)
    pcl = _clpair_density(dz_pool)

    zgrid = np.linspace(z_o.min(), z_o.max(), 160)
    nbar_z = np.interp(zgrid, fc.z_centres, fc.nz_profile, left=0.0, right=0.0)
    opd_all = los_overdensity(fc, ra_m, dec_m, zgrid)         # (M, n_z) field 1+δ
    bw_p = 0.02
    M = len(ra_m)
    z_miss = np.empty(M); fb = np.zeros(M, bool)
    for i in range(M):
        pf = opd_all[i] * nbar_z
        w = wk[i]; ok = np.isfinite(w) & (w > 0)
        pp = ((w[ok][None, :] * np.exp(-0.5 * ((zgrid[:, None] - zk[i][ok][None, :]) / bw_p) ** 2)).sum(1)
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
