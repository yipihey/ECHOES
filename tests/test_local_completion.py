"""True-3D local completion primitives (branch data/local-neighborhood)."""
import numpy as np
import pytest

from echoes.local_completion import (galactic_b, radial_nbar, _calibrate_bias,
                                     _ztransplant_kmag)


def test_galactic_b_matches_astropy():
    pytest.importorskip("astropy")
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    ra = np.array([0.0, 90.0, 180.0, 266.4, 12.3]); dec = np.array([0.0, 30.0, -30.0, -28.9, 41.0])
    ref = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs").galactic.b.deg
    np.testing.assert_allclose(galactic_b(ra, dec), ref, atol=0.4)   # NGP formula ~arcmin-accurate


def test_radial_nbar_uniform_sphere():
    # uniform-density sphere over the full sky (f_sky=1) -> roughly constant n̄(d)
    rng = np.random.default_rng(0)
    d = rng.random(200000) ** (1 / 3) * 200.0          # uniform in volume out to 200 Mpc
    edges = np.linspace(20, 180, 9)
    dctr, nbar = radial_nbar(d, 1.0, edges)
    assert np.all(nbar > 0)
    assert nbar.std() / nbar.mean() < 0.1              # ~constant density


def test_calibrate_bias_hits_target():
    rng = np.random.default_rng(1)
    opd = np.clip(1.0 + rng.normal(0, 1.0, 80000), 0, None)     # a 1+δ field, mean ~1
    for target in (1.4, 1.7, 2.0):                              # above the void gap, within reach
        b = _calibrate_bias(opd, target)
        w = opd ** b
        got = (opd * w).sum() / w.sum()
        assert abs(got - target) < 0.05
    assert _calibrate_bias(opd, 99.0) == pytest.approx(2.0)     # unreachable target -> clamp to hi


def test_ztransplant_kmag_distance_matched():
    rng = np.random.default_rng(2)
    donor_d = np.linspace(10, 300, 5000)
    donor_k = 9.0 + 5 * np.log10(donor_d)              # K grows with distance (flux limit)
    k = _ztransplant_kmag(np.array([50.0, 250.0]), donor_d, donor_k, rng, K=50)
    assert k[0] < k[1]                                 # nearer galaxy is brighter (smaller K)
    assert np.isfinite(k).all()
