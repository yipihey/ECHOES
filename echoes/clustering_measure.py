"""Survey-geometry clustering estimators for ECHOES covariance work.

Two Landy--Szalay statistics, both weight-aware (FKP x completeness), used to turn a
mock-catalog ENSEMBLE into a clustering covariance:

  * ``measure_wp``       projected correlation w_p(r_p)  (RSD-marginalised; isotropic part).
  * ``measure_xi_ell``   redshift-space multipoles xi_0(s), xi_2(s)  (the quadrupole carries
                         the anisotropy the LGCP field is built to reproduce).

Backend: `Corrfunc.mocks` (DDrppi_mocks / DDsmu_mocks) when importable — the standard,
fast, OpenMP survey-geometry pair counter. A pure ``scipy.spatial.cKDTree`` Landy--Szalay
fallback is provided so the machinery runs without Corrfunc (slower; intended for small N).

Inputs are sky coordinates + comoving distance ``cz_*`` (Mpc/h; we pass
``is_comoving_dist=True`` so no internal cosmology is applied). Per-object weights are
optional and default to 1. The LS normalisations use the *weighted* sums Sum(w), so the
estimator is unbiased under the supplied weights (FKP downweighting of dense regions).

The estimator amortises the random--random pair count: ``RandomPairs`` precomputes the
weighted RR (and the random normalisations) once for a fixed random catalog, and every
realization reuses it via ``rr=`` — RR is the dominant cost and the randoms never change
across the ensemble.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks as _DDrppi
    from Corrfunc.mocks.DDsmu_mocks import DDsmu_mocks as _DDsmu
    _HAS_CORRFUNC = True
except Exception:  # pragma: no cover - import guard
    _HAS_CORRFUNC = False

# Corrfunc "cosmology" selector is irrelevant when is_comoving_dist=True (distances are
# supplied directly), but the argument is mandatory; 2 = Planck.
_COSMO = 2


# ----------------------------------------------------------------------------- weights
def _w(w, n):
    if w is None:
        return np.ones(n, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if w.shape[0] != n:
        raise ValueError(f"weight length {w.shape[0]} != n_points {n}")
    return w


def _weighted_pairs(res):
    """Weighted pair count per bin from a Corrfunc result: npairs * weightavg.

    Corrfunc returns ``weightavg`` = mean of (w_i*w_j) over pairs in the bin (pair_product
    weighting), so the *sum* of pair weights is npairs * weightavg. With unit weights this
    reduces to npairs.
    """
    npairs = res["npairs"].astype(np.float64)
    if "weightavg" in res.dtype.names:
        return npairs * res["weightavg"].astype(np.float64)
    return npairs


# ----------------------------------------------------------------------- random cache
@dataclass
class RandomPairs:
    """Precomputed weighted RR pair counts for a fixed random catalog.

    Built once and reused across every realization (the randoms are common to the whole
    ensemble, and RR dominates the pair-count cost). Holds both the wp (rp, pi) RR grid and
    the xi_ell (s, mu) RR grid so a single random set serves both estimators.
    """
    nr_w: float              # weighted random normalisation Sum(w_r)
    rp_edges: np.ndarray
    pimax: float
    npibins: int
    RR_rppi: np.ndarray      # weighted RR over (rp, pi)
    s_edges: np.ndarray
    nmu: int
    RR_smu: np.ndarray       # weighted RR over (s, mu)


def build_random_pairs(ra_r, dec_r, cz_r, w_r, *, rp_edges, pimax, npibins,
                       s_edges, nmu, nthreads=8):
    """Precompute the weighted RR grids for wp and xi_ell from one random catalog."""
    if not _HAS_CORRFUNC:
        raise RuntimeError("build_random_pairs requires Corrfunc; the cKDTree fallback "
                           "computes RR inline per call instead.")
    ra_r = np.ascontiguousarray(ra_r, np.float64)
    dec_r = np.ascontiguousarray(dec_r, np.float64)
    cz_r = np.ascontiguousarray(cz_r, np.float64)
    w_r = _w(w_r, len(ra_r))

    rr_rppi = _DDrppi(1, _COSMO, nthreads, float(pimax), rp_edges,
                      ra_r, dec_r, cz_r, weights1=w_r,
                      is_comoving_dist=True, weight_type="pair_product")
    RR_rppi = _weighted_pairs(rr_rppi).reshape(len(rp_edges) - 1, npibins)

    rr_smu = _DDsmu(1, _COSMO, nthreads, 1.0, nmu, s_edges,
                    ra_r, dec_r, cz_r, weights1=w_r,
                    is_comoving_dist=True, weight_type="pair_product")
    RR_smu = _weighted_pairs(rr_smu).reshape(len(s_edges) - 1, nmu)

    return RandomPairs(nr_w=float(w_r.sum()), rp_edges=np.asarray(rp_edges),
                       pimax=float(pimax), npibins=int(npibins), RR_rppi=RR_rppi,
                       s_edges=np.asarray(s_edges), nmu=int(nmu), RR_smu=RR_smu)


# ------------------------------------------------------------------- LS normalisation
def _ls_cf(DD, DR, RR, nd, nr):
    """Weighted Landy--Szalay xi from weighted pair sums and weighted normalisations.

      xi = (DD/NDD - 2 DR/NDR + RR/NRR) / (RR/NRR)
    with NDD = nd^2, NDR = nd*nr, NRR = nr^2 (weighted-count normalisation; the self-pair
    correction is negligible for these N and cancels to leading order in the ratio).
    """
    ndd, ndr, nrr = nd * nd, nd * nr, nr * nr
    dd, dr, rr = DD / ndd, DR / ndr, RR / nrr
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = np.where(rr > 0, (dd - 2.0 * dr + rr) / rr, 0.0)
    return xi


# --------------------------------------------------------------------------- wp(r_p)
def measure_wp(ra_d, dec_d, cz_d, w_d, ra_r, dec_r, cz_r, w_r, *,
               rp_edges, pimax=80.0, npibins=None, nthreads=8, rr: Optional[RandomPairs] = None):
    """Projected correlation function w_p(r_p) via Landy--Szalay (survey geometry).

    ``cz_*`` are comoving distances (Mpc/h). ``pimax`` is the line-of-sight integration
    limit (Mpc/h), with ``npibins`` linear pi bins of width pimax/npibins (default
    int(pimax), i.e. 1 Mpc/h). Returns ``(rp_centers, wp)``.
    """
    rp_edges = np.asarray(rp_edges, np.float64)
    if npibins is None:
        npibins = int(round(pimax))
    rp_c = 0.5 * (rp_edges[:-1] + rp_edges[1:])

    if not _HAS_CORRFUNC:
        return _wp_fallback(ra_d, dec_d, cz_d, w_d, ra_r, dec_r, cz_r, w_r,
                            rp_edges=rp_edges, pimax=pimax, npibins=npibins, rp_c=rp_c)

    ra_d = np.ascontiguousarray(ra_d, np.float64); dec_d = np.ascontiguousarray(dec_d, np.float64)
    cz_d = np.ascontiguousarray(cz_d, np.float64); w_d = _w(w_d, len(ra_d))
    ra_r = np.ascontiguousarray(ra_r, np.float64); dec_r = np.ascontiguousarray(dec_r, np.float64)
    cz_r = np.ascontiguousarray(cz_r, np.float64); w_r = _w(w_r, len(ra_r))
    nd, nr = float(w_d.sum()), float(w_r.sum())

    dd = _DDrppi(1, _COSMO, nthreads, float(pimax), rp_edges, ra_d, dec_d, cz_d,
                 weights1=w_d, is_comoving_dist=True, weight_type="pair_product")
    DD = _weighted_pairs(dd).reshape(len(rp_edges) - 1, npibins)
    dr = _DDrppi(0, _COSMO, nthreads, float(pimax), rp_edges, ra_d, dec_d, cz_d,
                 weights1=w_d, RA2=ra_r, DEC2=dec_r, CZ2=cz_r, weights2=w_r,
                 is_comoving_dist=True, weight_type="pair_product")
    DR = _weighted_pairs(dr).reshape(len(rp_edges) - 1, npibins)
    if rr is not None:
        RR = rr.RR_rppi
        nr = rr.nr_w
    else:
        rrres = _DDrppi(1, _COSMO, nthreads, float(pimax), rp_edges, ra_r, dec_r, cz_r,
                        weights1=w_r, is_comoving_dist=True, weight_type="pair_product")
        RR = _weighted_pairs(rrres).reshape(len(rp_edges) - 1, npibins)

    xi_rppi = _ls_cf(DD, DR, RR, nd, nr)        # (n_rp, n_pi)
    dpi = pimax / npibins
    wp = 2.0 * dpi * xi_rppi.sum(axis=1)        # integrate over pi (both signs folded by Corrfunc)
    return rp_c, wp


# ----------------------------------------------------------------- xi_0, xi_2 (s, mu)
def measure_xi_ell(ra_d, dec_d, cz_d, w_d, ra_r, dec_r, cz_r, w_r, *,
                   s_edges, nmu=100, nthreads=8, rr: Optional[RandomPairs] = None):
    """Redshift-space multipoles xi_0(s), xi_2(s) via Landy--Szalay in (s, mu).

    Computes xi(s, mu) on a [0,1] mu grid (Corrfunc folds +/-mu), then projects onto the
    Legendre moments L_0=1 and L_2=(3 mu^2-1)/2:  xi_l = (2l+1) <xi(s,mu) L_l(mu)>_mu.
    Returns ``(s_centers, xi0, xi2)``.
    """
    s_edges = np.asarray(s_edges, np.float64)
    s_c = 0.5 * (s_edges[:-1] + s_edges[1:])

    if not _HAS_CORRFUNC:
        return _xiell_fallback(ra_d, dec_d, cz_d, w_d, ra_r, dec_r, cz_r, w_r,
                               s_edges=s_edges, nmu=nmu, s_c=s_c)

    ra_d = np.ascontiguousarray(ra_d, np.float64); dec_d = np.ascontiguousarray(dec_d, np.float64)
    cz_d = np.ascontiguousarray(cz_d, np.float64); w_d = _w(w_d, len(ra_d))
    ra_r = np.ascontiguousarray(ra_r, np.float64); dec_r = np.ascontiguousarray(dec_r, np.float64)
    cz_r = np.ascontiguousarray(cz_r, np.float64); w_r = _w(w_r, len(ra_r))
    nd, nr = float(w_d.sum()), float(w_r.sum())

    dd = _DDsmu(1, _COSMO, nthreads, 1.0, nmu, s_edges, ra_d, dec_d, cz_d,
                weights1=w_d, is_comoving_dist=True, weight_type="pair_product")
    DD = _weighted_pairs(dd).reshape(len(s_edges) - 1, nmu)
    dr = _DDsmu(0, _COSMO, nthreads, 1.0, nmu, s_edges, ra_d, dec_d, cz_d,
                weights1=w_d, RA2=ra_r, DEC2=dec_r, CZ2=cz_r, weights2=w_r,
                is_comoving_dist=True, weight_type="pair_product")
    DR = _weighted_pairs(dr).reshape(len(s_edges) - 1, nmu)
    if rr is not None:
        RR = rr.RR_smu
        nr = rr.nr_w
    else:
        rrres = _DDsmu(1, _COSMO, nthreads, 1.0, nmu, s_edges, ra_r, dec_r, cz_r,
                       weights1=w_r, is_comoving_dist=True, weight_type="pair_product")
        RR = _weighted_pairs(rrres).reshape(len(s_edges) - 1, nmu)

    xi_smu = _ls_cf(DD, DR, RR, nd, nr)         # (n_s, n_mu)
    return _project_multipoles(xi_smu, s_c, nmu)


def _project_multipoles(xi_smu, s_c, nmu):
    """xi(s,mu) -> xi0, xi2 by Gauss-Legendre-free uniform-mu Simpson-ish average."""
    mu = (np.arange(nmu) + 0.5) / nmu          # bin centers on [0,1]
    dmu = 1.0 / nmu
    L0 = np.ones_like(mu)
    L2 = 0.5 * (3.0 * mu ** 2 - 1.0)
    xi0 = (2 * 0 + 1) * (xi_smu * L0[None, :]).sum(axis=1) * dmu
    xi2 = (2 * 2 + 1) * (xi_smu * L2[None, :]).sum(axis=1) * dmu
    return s_c, xi0, xi2


# ===================================================================== scipy fallback
# Pure cKDTree Landy--Szalay (no Corrfunc). Slower; for small-N machinery checks only.

def _radec_cz_to_xyz(ra, dec, cz):
    ra = np.deg2rad(np.asarray(ra, np.float64)); dec = np.deg2rad(np.asarray(dec, np.float64))
    r = np.asarray(cz, np.float64)
    cd = np.cos(dec)
    return np.stack([r * cd * np.cos(ra), r * cd * np.sin(ra), r * np.sin(dec)], axis=1)


def _pair_hist_rppi(Pa, wa, Pb, wb, rp_edges, pimax, npibins, autocorr):
    """Weighted (rp, pi) pair histogram via cKDTree, mid-point line of sight."""
    from scipy.spatial import cKDTree
    rmax = float(np.hypot(rp_edges[-1], pimax))
    ta, tb = cKDTree(Pa), cKDTree(Pb)
    pairs = ta.query_ball_tree(tb, rmax)
    out = np.zeros((len(rp_edges) - 1, npibins))
    for i, js in enumerate(pairs):
        if not js:
            continue
        js = np.asarray(js)
        if autocorr:
            js = js[js > i]
            if js.size == 0:
                continue
        s = Pa[i] - Pb[js]
        lvec = 0.5 * (Pa[i] + Pb[js])
        lnorm = np.linalg.norm(lvec, axis=1)
        pi = np.abs((s * lvec).sum(axis=1) / np.where(lnorm > 0, lnorm, 1.0))
        rp = np.sqrt(np.maximum(np.einsum("ij,ij->i", s, s) - pi ** 2, 0.0))
        wpair = wa[i] * wb[js]
        h, _, _ = np.histogram2d(rp, pi, bins=[rp_edges, np.linspace(0, pimax, npibins + 1)],
                                 weights=wpair)
        out += h
    if autocorr:
        out *= 2.0  # restore double counting convention of Corrfunc autocorr
    return out


def _wp_fallback(ra_d, dec_d, cz_d, w_d, ra_r, dec_r, cz_r, w_r, *, rp_edges, pimax, npibins, rp_c):
    Pd = _radec_cz_to_xyz(ra_d, dec_d, cz_d); w_d = _w(w_d, len(Pd))
    Pr = _radec_cz_to_xyz(ra_r, dec_r, cz_r); w_r = _w(w_r, len(Pr))
    nd, nr = float(w_d.sum()), float(w_r.sum())
    DD = _pair_hist_rppi(Pd, w_d, Pd, w_d, rp_edges, pimax, npibins, autocorr=True)
    DR = _pair_hist_rppi(Pd, w_d, Pr, w_r, rp_edges, pimax, npibins, autocorr=False)
    RR = _pair_hist_rppi(Pr, w_r, Pr, w_r, rp_edges, pimax, npibins, autocorr=True)
    xi = _ls_cf(DD, DR, RR, nd, nr)
    dpi = pimax / npibins
    return rp_c, 2.0 * dpi * xi.sum(axis=1)


def _pair_hist_smu(Pa, wa, Pb, wb, s_edges, nmu, autocorr):
    from scipy.spatial import cKDTree
    rmax = float(s_edges[-1])
    ta, tb = cKDTree(Pa), cKDTree(Pb)
    pairs = ta.query_ball_tree(tb, rmax)
    out = np.zeros((len(s_edges) - 1, nmu))
    mu_edges = np.linspace(0.0, 1.0, nmu + 1)
    for i, js in enumerate(pairs):
        if not js:
            continue
        js = np.asarray(js)
        if autocorr:
            js = js[js > i]
            if js.size == 0:
                continue
        svec = Pa[i] - Pb[js]
        lvec = 0.5 * (Pa[i] + Pb[js])
        lnorm = np.linalg.norm(lvec, axis=1)
        snorm = np.linalg.norm(svec, axis=1)
        mu = np.abs((svec * lvec).sum(axis=1) / np.where(snorm * lnorm > 0, snorm * lnorm, 1.0))
        wpair = wa[i] * wb[js]
        h, _, _ = np.histogram2d(snorm, mu, bins=[s_edges, mu_edges], weights=wpair)
        out += h
    if autocorr:
        out *= 2.0
    return out


def _xiell_fallback(ra_d, dec_d, cz_d, w_d, ra_r, dec_r, cz_r, w_r, *, s_edges, nmu, s_c):
    Pd = _radec_cz_to_xyz(ra_d, dec_d, cz_d); w_d = _w(w_d, len(Pd))
    Pr = _radec_cz_to_xyz(ra_r, dec_r, cz_r); w_r = _w(w_r, len(Pr))
    nd, nr = float(w_d.sum()), float(w_r.sum())
    DD = _pair_hist_smu(Pd, w_d, Pd, w_d, s_edges, nmu, autocorr=True)
    DR = _pair_hist_smu(Pd, w_d, Pr, w_r, s_edges, nmu, autocorr=False)
    RR = _pair_hist_smu(Pr, w_r, Pr, w_r, s_edges, nmu, autocorr=True)
    xi_smu = _ls_cf(DD, DR, RR, nd, nr)
    return _project_multipoles(xi_smu, s_c, nmu)


def has_corrfunc() -> bool:
    return _HAS_CORRFUNC
