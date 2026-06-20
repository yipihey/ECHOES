"""M1: generative inpaint sampler fills interior holes with PROV=5 galaxies."""
import numpy as np

from echoes.fill_footprint import build_fill_footprint
from echoes.inpaint_field import sample_inpaint_catalog, PROV_INPAINT
from echoes.completion import fill_regime, PROV


def _patch(n=200_000, lo=0.0, hi=12.0, hole=(6.0, 6.0, 1.5), seed=0):
    rng = np.random.default_rng(seed)
    ra = rng.uniform(lo, hi, n); dec = rng.uniform(lo, hi, n)
    cra, cdec, rad = hole
    d = np.hypot((ra - cra) * np.cos(np.radians(dec)), dec - cdec)
    keep = d > rad
    return ra[keep], dec[keep]


def test_analog_inpaint_fills_interior_hole():
    # randoms + donor galaxies share the same punched hole
    ra_r, dec_r = _patch(seed=0)
    ra_d, dec_d = _patch(n=60_000, seed=1)
    z_d = np.random.default_rng(2).uniform(0.45, 0.70, len(ra_d))
    fp = build_fill_footprint(ra_random=ra_r, dec_random=dec_r, z_data=z_d,
                              nside=64, mangle_ply=None, mangle_npy=None, lss_clip_deg=3.0)
    assert fp.holes, "expected an interior hole"

    out = sample_inpaint_catalog(fp, donor_ra=ra_d, donor_dec=dec_d, donor_z=z_d,
                                 rand_ra=ra_r, rand_dec=dec_r, mode="analog", seed=0)
    assert len(out["ra"]) > 0
    assert np.all(out["prov"] == PROV_INPAINT)
    # inpainted galaxies land inside the (6,6) hole
    d = np.hypot((out["ra"] - 6.0) * np.cos(np.radians(out["dec"])), out["dec"] - 6.0)
    assert d.max() < 2.0
    # redshifts are drawn from the donor distribution (transplant), in range
    assert out["z"].min() >= 0.45 - 1e-6 and out["z"].max() <= 0.70 + 1e-6
    # uncert is a valid [0,1] prior-dominance flag that RISES toward the hole centre
    # (deeper into the hole = farther from any data = more prior-dominated)
    assert out["uncert"].min() >= 0.0 and out["uncert"].max() <= 1.0
    d_center = np.hypot((out["ra"] - 6.0) * np.cos(np.radians(out["dec"])), out["dec"] - 6.0)
    assert np.corrcoef(d_center, out["uncert"])[0, 1] < -0.2   # near centre -> higher uncert


def test_no_holes_returns_empty():
    ra_r, dec_r = _patch(hole=(6.0, 6.0, 0.0))   # complete, no hole
    z = np.random.default_rng(3).uniform(0.45, 0.70, 30_000)
    fp = build_fill_footprint(ra_random=ra_r, dec_random=dec_r, z_data=z,
                              nside=64, mangle_ply=None, mangle_npy=None, lss_clip_deg=3.0)
    out = sample_inpaint_catalog(fp, donor_ra=ra_r[:30_000], donor_dec=dec_r[:30_000],
                                 donor_z=z, rand_ra=ra_r, rand_dec=dec_r, mode="analog")
    assert len(out["ra"]) == 0 and out["prov"].dtype == np.int8


def test_fill_regime_flags_prior_inpaint():
    prov = np.array([PROV["observed"], PROV["collided"], PROV["inpaint"], PROV["inpaint"]])
    uncert = np.array([0.0, 0.0, 0.1, 0.9])
    regime, is_prior = fill_regime(prov, uncert, prior_thresh=0.5)
    assert list(regime) == ["observed", "completed", "inpaint_data", "inpaint_prior"]
    assert list(is_prior) == [False, False, False, True]
    # without uncert, inpaint galaxies are conservatively flagged prior
    _, is_prior2 = fill_regime(prov)
    assert is_prior2.sum() == 2
