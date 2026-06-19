"""The generative selection model: probability terms well-formed + self-consistent."""
import numpy as np
import pytest

from echoes.selection_model import SelectionModel, local_close_pair_count


def test_probability_terms_well_formed():
    sm = SelectionModel(coll_frac=0.6, zfail_frac=0.014, zfail_faint_bias=1.5)
    # imaging detection: w_systot=1 -> 1; w_systot=2 -> 0.5; w_systot<1 -> 1
    assert np.isclose(sm.p_img(1.0), 1.0)
    assert np.isclose(sm.p_img(2.0), 0.5)
    assert np.isclose(sm.p_img(0.8), 1.0)
    # collision loss: 0 with no neighbour, monotone increasing, in [0,1]
    n = np.array([0, 1, 2, 5, 10.0])
    pc = sm.p_collision(n)
    assert pc[0] == 0.0
    assert np.all(np.diff(pc) > 0) and np.all((pc >= 0) & (pc <= 1))
    assert np.isclose(pc[1], 0.3)        # coll_frac/2: one of the pair is removed
    # p_observed decreases with collisions and with w_systot
    p0 = sm.p_observed(1.0, np.array([0.0]), imag=np.array([20.0]))
    p1 = sm.p_observed(1.0, np.array([3.0]), imag=np.array([20.0]))
    p2 = sm.p_observed(1.6, np.array([0.0]), imag=np.array([20.0]))
    assert p1[0] < p0[0] and p2[0] < p0[0]
    assert np.all((p0 >= 0) & (p0 <= 1))


def test_photoz_loglike_peaks_at_truth():
    sm = SelectionModel(photoz_sigma=0.03)
    zg = np.linspace(0.4, 0.7, 200)
    ll = sm.photoz_loglike(zg, np.array([0.55, 0.62]))
    assert ll.shape == (2, 200)
    assert np.isclose(zg[ll[0].argmax()], 0.55, atol=0.005)
    assert np.isclose(zg[ll[1].argmax()], 0.62, atol=0.005)


def test_missing_plus_observed_equals_true_intensity():
    sm = SelectionModel()
    rng = np.random.default_rng(0)
    lam = rng.uniform(0.5, 2.0, 50)
    w = rng.uniform(0.85, 1.5, 50); nc = rng.integers(0, 4, 50).astype(float)
    obs = sm.observed_thinning(lam, w, nc)
    miss = sm.missing_intensity(lam, w, nc)
    assert np.allclose(obs + miss, lam)        # partition of the true intensity


@pytest.mark.parametrize("scale", [62.0 / 3600.0])
def test_local_close_pair_count(scale):
    # a tight clump of 4 + isolated points: clump members each see >=3 neighbours
    ra = np.r_[np.full(4, 150.0) + 1e-3 * np.arange(4), 160.0, 170.0]
    dec = np.r_[np.full(4, 10.0), 20.0, 30.0]
    nc = local_close_pair_count(ra, dec, scale)
    assert nc.shape == (6,)
    assert np.all(nc[:4] >= 3) and nc[4] == 0 and nc[5] == 0
