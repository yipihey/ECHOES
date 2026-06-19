"""Numba per-cap counting kernels for the angular kNN-CDF estimator.

Surgically extracted from the graphGP-cosmology ``sigma^2`` cone-shell
estimator (the research code) — only the two ``@numba.njit(nogil=True)``
kernels and the ``_NUMBA_OK`` flag are carried over, mirroring the same
self-contained extraction used for :mod:`echoes.geometry`. The full
``sigma^2`` cone-shell estimator is intentionally *not* ported; the ECHOES
2D-kNN engine needs only the per-cap counting primitive.

``_per_cap_count_kernel`` is the single hot loop of the cone-shell pipeline:
for one cap centre ``(theta_c, phi_c)`` and the pre-fetched HEALPix disc of
neighbour pixels, it accumulates weighted neighbour counts into a
``(n_theta, n_z)`` matrix whose ``[t, k]`` entry is the count of neighbours
with angular separation ``<= theta_radii[t]`` and redshift in shell ``k``
(cumulative along the theta axis). It releases the GIL so the outer query
loop can run in a thread pool. ``_per_cap_count_kernel_per_region`` is the
jackknife variant that splits counts by a per-neighbour region label.
"""

from __future__ import annotations

import numpy as np


try:
    import numba

    @numba.njit(cache=True, nogil=True)
    def _per_cap_count_kernel(
        theta_c, phi_c,
        ipix_disc,
        pix_starts,
        theta_g_sorted, phi_g_sorted, z_g_sorted, w_g_sorted,
        theta_radii, z_edges,
    ):
        """Numba-JIT per-cap kernel.

        For a single cap centre ``(theta_c, phi_c)`` and the set of
        ``ipix_disc`` pixels returned by ``hp.query_disc`` at the
        largest theta, accumulate weighted counts into a (n_theta, n_z)
        matrix where the (t, k) entry is the count of galaxies with
        angular separation <= ``theta_radii[t]`` and redshift in shell
        ``k``.

        Releases the GIL (``nogil=True``) so the outer loop can run
        across many caps in a thread pool.
        """
        n_theta = theta_radii.shape[0]
        n_z = z_edges.shape[0] - 1
        out = np.zeros((n_theta, n_z))
        sin_tc = np.sin(theta_c)
        cos_tc = np.cos(theta_c)
        z_lo_total = z_edges[0]
        z_hi_total = z_edges[n_z]
        for ip_idx in range(ipix_disc.shape[0]):
            ip = ipix_disc[ip_idx]
            s = pix_starts[ip]
            e = pix_starts[ip + 1]
            for j in range(s, e):
                tg = theta_g_sorted[j]
                pg = phi_g_sorted[j]
                cs = (sin_tc * np.sin(tg) * np.cos(phi_c - pg)
                        + cos_tc * np.cos(tg))
                if cs > 1.0:
                    cs = 1.0
                elif cs < -1.0:
                    cs = -1.0
                sep = np.arccos(cs)
                # binary search for smallest t with theta_radii[t] >= sep
                lo = 0
                hi = n_theta
                while lo < hi:
                    mid = (lo + hi) // 2
                    if theta_radii[mid] < sep:
                        lo = mid + 1
                    else:
                        hi = mid
                t_bin = lo
                if t_bin >= n_theta:
                    continue
                # binary search for z bin: smallest k with z_edges[k+1] > zj
                zj = z_g_sorted[j]
                if zj < z_lo_total or zj >= z_hi_total:
                    continue
                lo = 0
                hi = n_z
                while lo < hi:
                    mid = (lo + hi) // 2
                    if z_edges[mid + 1] <= zj:
                        lo = mid + 1
                    else:
                        hi = mid
                z_bin = lo
                out[t_bin, z_bin] += w_g_sorted[j]
        # cumulative sum across the theta axis -- a galaxy at
        # angular separation sep contributes to all caps t with
        # theta_radii[t] >= sep, so we accumulate in-place.
        for k in range(n_z):
            acc = 0.0
            for t in range(n_theta):
                acc += out[t, k]
                out[t, k] = acc
        return out

    @numba.njit(cache=True, nogil=True)
    def _per_cap_count_kernel_per_region(
        theta_c, phi_c,
        ipix_disc,
        pix_starts,
        theta_g_sorted, phi_g_sorted, z_g_sorted, w_g_sorted,
        gal_region_sorted,
        theta_radii, z_edges, n_regions,
    ):
        """Like ``_per_cap_count_kernel`` but accumulates counts split
        by per-galaxy ``gal_region_sorted`` label, returning a
        ``(n_theta, n_z, n_regions)`` cube.

        Used by the single-pass jackknife to enable a fold cost equal to
        one full count pass: ``N_minus_k = cube.sum(-1) - cube[..., k]``.
        """
        n_theta = theta_radii.shape[0]
        n_z = z_edges.shape[0] - 1
        out = np.zeros((n_theta, n_z, n_regions))
        sin_tc = np.sin(theta_c)
        cos_tc = np.cos(theta_c)
        z_lo_total = z_edges[0]
        z_hi_total = z_edges[n_z]
        for ip_idx in range(ipix_disc.shape[0]):
            ip = ipix_disc[ip_idx]
            s = pix_starts[ip]
            e = pix_starts[ip + 1]
            for j in range(s, e):
                tg = theta_g_sorted[j]
                pg = phi_g_sorted[j]
                cs = (sin_tc * np.sin(tg) * np.cos(phi_c - pg)
                        + cos_tc * np.cos(tg))
                if cs > 1.0:
                    cs = 1.0
                elif cs < -1.0:
                    cs = -1.0
                sep = np.arccos(cs)
                lo = 0
                hi = n_theta
                while lo < hi:
                    mid = (lo + hi) // 2
                    if theta_radii[mid] < sep:
                        lo = mid + 1
                    else:
                        hi = mid
                t_bin = lo
                if t_bin >= n_theta:
                    continue
                zj = z_g_sorted[j]
                if zj < z_lo_total or zj >= z_hi_total:
                    continue
                lo = 0
                hi = n_z
                while lo < hi:
                    mid = (lo + hi) // 2
                    if z_edges[mid + 1] <= zj:
                        lo = mid + 1
                    else:
                        hi = mid
                z_bin = lo
                r = gal_region_sorted[j]
                if r < 0 or r >= n_regions:
                    continue
                out[t_bin, z_bin, r] += w_g_sorted[j]
        # cumulative sum across the theta axis -- per (z, region).
        for k in range(n_z):
            for r in range(n_regions):
                acc = 0.0
                for t in range(n_theta):
                    acc += out[t, k, r]
                    out[t, k, r] = acc
        return out

    _NUMBA_OK = True
except ImportError:                                            # pragma: no cover
    _NUMBA_OK = False
    _per_cap_count_kernel = None
    _per_cap_count_kernel_per_region = None
