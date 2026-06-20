"""Tier-A generative engine: the measured transform reshapes the one-point PDF
while preserving rank order (calibration), and identity == fieldpost passthrough.
Data-free (hand-built FieldContext + synthetic PDF), no Corrfunc/real catalogs."""
import numpy as np
import pytest

from echoes.density_transform import (DensityTransform, fit_density_transform, _moments)


def test_empirical_transform_recovers_pdf_and_is_monotone():
    rng = np.random.default_rng(0)
    target = np.exp(0.6 * rng.normal(size=100_000) - 0.5 * 0.6 ** 2)   # skewed, mean 1
    dt = fit_density_transform(target, kind="empirical")
    out = dt.T(rng.normal(size=100_000))
    mt, vt, st = _moments(target); mo, vo, so = _moments(out)
    assert abs(mo - mt) < 0.03 and abs(vo - vt) < 0.05 and abs(so - st) < 0.3
    # strictly monotone (rank/PIT preserving) + clean round-trip on the support
    g = np.linspace(-2.5, 2.5, 200)
    Tg = dt.T(g)
    assert np.all(np.diff(Tg) >= -1e-9)
    assert np.max(np.abs(dt.T_inv(Tg) - g)) < 1e-2


def test_lognormal_transform_matches_variance():
    rng = np.random.default_rng(1)
    target = np.exp(0.5 * rng.normal(size=80_000) - 0.5 * 0.5 ** 2)
    dt = fit_density_transform(target, kind="lognormal")
    out = dt.T(rng.normal(size=80_000))
    assert abs(_moments(out)[1] - _moments(target)[1]) < 0.05
    assert dt.sigma_g > 0


def test_identity_is_exact_passthrough():
    dt = DensityTransform(kind="identity")
    x = np.linspace(-3, 5, 50)
    assert np.array_equal(dt.T(x), x)
    assert np.array_equal(dt.apply_to_field(x), x)


def test_generative_model_identity_hook_is_none():
    from echoes.generative import GenerativeModel
    gm = GenerativeModel(field_ctx=None, transform=DensityTransform(kind="identity"))
    assert gm.los_transform() is None                       # exact fieldpost parity
    assert gm.n_samples == 1


def test_los_transform_reshapes_field_preserving_rank():
    from echoes.generative import GenerativeModel
    rng = np.random.default_rng(2)
    target = np.exp(0.7 * rng.normal(size=50_000) - 0.5 * 0.7 ** 2)
    dt = fit_density_transform(target, kind="empirical")
    gm = GenerativeModel(field_ctx=None, transform=dt, sigma_ref=0.5)
    tf = gm.los_transform()
    assert tf is not None
    # a Gaussian-ish field (1+δ around 1) -> reshaped to a skewed marginal
    field = 1.0 + 0.5 * rng.normal(size=(40, 30))
    out = tf(field)
    assert out.shape == field.shape
    assert np.all(out >= 0)
    assert _moments(out.ravel())[2] > _moments(field.ravel())[2] + 0.5   # skew increased
    # monotone per element: ranking within a sightline is preserved
    i = field[0].argsort()
    assert np.all(np.diff(out[0][i]) >= -1e-6)


def test_cic_overdensity_shape():
    from types import SimpleNamespace
    from echoes.generative import _cic_overdensity
    rng = np.random.default_rng(3)
    cat = SimpleNamespace(
        ra_data=rng.uniform(150, 155, 2000), dec_data=rng.uniform(0, 5, 2000),
        z_data=rng.uniform(0.45, 0.6, 2000),
        ra_random=rng.uniform(150, 155, 20000), dec_random=rng.uniform(0, 5, 20000),
        z_random=rng.uniform(0.45, 0.6, 20000))
    opd = _cic_overdensity(cat, R=8.0, n_cells=1000)
    assert opd.shape == (1000,) and np.all(opd >= 0) and abs(opd.mean() - 1.0) < 0.2


def test_cic_overdensity_without_randoms():
    """A catalog lacking ``ra_random`` (e.g. a mock observed object) must not crash:
    explicit randoms, then the data-as-cells fallback."""
    from types import SimpleNamespace
    from echoes.generative import _cic_overdensity
    rng = np.random.default_rng(4)
    obs = SimpleNamespace(ra_data=rng.uniform(150, 155, 2000), dec_data=rng.uniform(0, 5, 2000),
                          z_data=rng.uniform(0.45, 0.6, 2000))           # no ra_random
    rnd = (rng.uniform(150, 155, 20000), rng.uniform(0, 5, 20000), rng.uniform(0.45, 0.6, 20000))
    o1 = _cic_overdensity(obs, randoms=rnd, n_cells=800)
    o2 = _cic_overdensity(obs, n_cells=800)                              # data fallback
    assert o1.shape == (800,) and o2.shape == (800,)
    assert abs(o1.mean() - 1.0) < 0.2 and np.all(o1 >= 0) and np.all(o2 >= 0)
