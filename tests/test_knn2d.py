"""Unit tests for the experimental kNN2D engine subpackage and field builder.

Self-contained (synthetic catalogs, no survey data download). Covers:
  * the ported joint angular kNN-CDF primitive — DD/RD cube shapes and the
    Davis-Peebles ξ ≈ 0 on a uniform field,
  * the field builder + per-sightline (1+δ) reduction — finite, positive RD,
    and (1+δ) ≈ 1 on a uniform field (no spurious structure).
"""
import types

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")
pytest.importorskip("numba")


# A uniform random patch + matching binary sel_map at low NSIDE.
NSIDE = 64
RA_LO, RA_HI, DEC_LO, DEC_HI = 150.0, 165.0, 5.0, 20.0
Z_LO, Z_HI = 0.45, 0.65


def _uniform_patch(n, seed=0):
    rng = np.random.default_rng(seed)
    ra = rng.uniform(RA_LO, RA_HI, n)
    dec = rng.uniform(DEC_LO, DEC_HI, n)
    z = rng.uniform(Z_LO, Z_HI, n)
    return ra, dec, z


def _sel_map():
    npix = 12 * NSIDE ** 2
    theta, phi = healpy.pix2ang(NSIDE, np.arange(npix))
    ra = np.degrees(phi); dec = 90.0 - np.degrees(theta)
    sel = ((ra >= RA_LO) & (ra <= RA_HI) & (dec >= DEC_LO) & (dec <= DEC_HI)).astype(float)
    return sel


def test_joint_knn_cdf_shapes_and_uniform_xi():
    from echoes.knn import joint_knn_cdf, derived
    ra, dec, z = _uniform_patch(4000)
    theta = np.radians(np.geomspace(0.05, 0.8, 6))
    z_q = np.array([Z_LO, Z_HI]); z_n = np.linspace(Z_LO, Z_HI, 9)
    dd = joint_knn_cdf(ra, dec, z, ra, dec, z, theta, z_q, z_n, k_max=6, flavor="DD")
    assert dd.H_geq_k.shape == (6, 1, 8, 6)
    assert dd.sum_n.shape == (6, 1, 8)
    assert dd.N_q[0] == len(ra)
    rar, decr, zr = _uniform_patch(4 * len(ra), seed=1)
    rd = joint_knn_cdf(rar, decr, zr, ra, dec, z, theta, z_q, z_n, k_max=6, flavor="RD")
    xi = derived.xi_dp(dd, rd)
    # uniform field: Davis-Peebles xi ~ 0 (shot noise only).
    assert np.nanmedian(np.abs(xi)) < 0.15
    mc = derived.mean_count(rd)
    assert np.isfinite(mc).all() and (mc >= 0).all()


def test_build_field_and_uniform_overdensity():
    from echoes.knn2d_field import (build_knn2d_field, _per_sightline_dd,
                                    _one_plus_delta)
    ra, dec, z = _uniform_patch(20000)
    cat = types.SimpleNamespace(ra_data=ra, dec_data=dec, z_data=z)
    sel = _sel_map()
    field = build_knn2d_field(
        cat, sel_map=sel, nside=NSIDE, rd_source="mc", n_rd_factor=4,
        aperture_deg=0.6, theta_edges_deg=np.geomspace(0.05, 1.0, 6),
        n_z_n=24, min_expected=0.5, nside_lookup=256, seed=0)
    assert field.rd_cum.shape == (6, 24)
    assert np.isfinite(field.rd_cum).all() and (field.rd_cum >= 0).all()
    assert field.rd_cum[field.aperture_index].max() > 0

    # per-sightline (1+delta) on a sample of interior query positions.
    rq, dq, _ = _uniform_patch(300, seed=7)
    inside = (rq > RA_LO + 1) & (rq < RA_HI - 1) & (dq > DEC_LO + 1) & (dq < DEC_HI - 1)
    rq, dq = rq[inside], dq[inside]
    dd = _per_sightline_dd(field, rq, dq, n_threads=1)
    assert dd.shape == (len(rq), 6, 24)
    opd = _one_plus_delta(dd, field)
    assert opd.shape == (len(rq), 24)
    # uniform field: well-covered cells have (1+delta) ~ 1 on average.
    covered = opd[opd > 0]
    assert covered.size > 0
    assert 0.6 < float(np.median(covered)) < 1.5


def test_completion_knn2d_schema():
    """The 'knn2d' branch returns the documented schema on a synthetic catalog
    with synthetic targets + a trivial photo-z posterior."""
    from echoes.completion import complete_catalog_photoz, PROV
    from echoes.knn2d_field import build_knn2d_field
    ra, dec, z = _uniform_patch(12000)
    cat = types.SimpleNamespace(
        ra_data=ra, dec_data=dec, z_data=z, w_sys_data=None,
        w_cp_data=None, w_noz_data=None)
    sel = _sel_map()
    field = build_knn2d_field(cat, sel_map=sel, nside=NSIDE, rd_source="mc",
                              aperture_deg=0.6, theta_edges_deg=np.geomspace(0.05, 1.0, 6),
                              n_z_n=24, nside_lookup=256, seed=0)
    # synthetic missing targets at known positions (no host: pure field draw).
    rng = np.random.default_rng(3)
    M = 200
    ra_m, dec_m, _ = _uniform_patch(M, seed=9)
    targets = types.SimpleNamespace(
        ra=ra_m, dec=dec_m, N=M, host_index=np.full(M, -1),
        miss_kind=np.array(["zfail"] * M),
        colors=np.zeros((M, 4)), mags=np.zeros((M, 5)))

    class _PZ:
        def posterior(self, feat):
            n = len(feat)
            zk = np.tile(np.linspace(Z_LO, Z_HI, 16), (n, 1))
            wk = np.ones_like(zk)
            return zk, wk

    out = complete_catalog_photoz(cat, targets, _PZ(), seed=0, z_mode="knn2d",
                                  knn2d_field=field,
                                  dz_pool=np.array([0.0, 0.001, -0.001]))
    for key in ("ra", "dec", "z", "N", "prov"):
        assert key in out
    assert out["N"] == len(out["ra"]) == len(out["z"])
    zc = np.asarray(out["z"])
    assert np.isfinite(zc).all()
    assert (zc.min() >= Z_LO - 0.05) and (zc.max() <= Z_HI + 0.05)
    assert set(np.unique(out["prov"])).issubset(set(PROV.values()))
