"""The field-correlation copula injects cross-object dependence while leaving every
per-object marginal (hence PIT calibration) unchanged, and stays reproducible."""
import numpy as np
import pytest
from echoes.posterior import draw, write_package, load_package, _phi


def _toy_pkg(a=None):
    """2 missing objects, identity quantile (z_i = u_i), no observed/systot. If ``a``
    is given, attach a 1-mode copula with Gaussian cross-correlation a² between them."""
    nq = 65
    qlev = np.linspace(0.0, 1.0, nq).astype(np.float32)
    invcdf = np.repeat(qlev[None, :], 2, axis=0).astype(np.float32)   # invcdf_i(u)=u ⇒ z=u
    pkg = {"n_obs": 0, "n_miss": 2, "zmin": 0.0, "zmax": 1.0,
           "qlev": qlev.astype(np.float64), "jitter": 0.0,
           "obs_z": np.zeros(0, np.float32),
           "base_ra": np.zeros(2, np.float32), "base_dec": np.zeros(2, np.float32),
           "base_wsys": np.ones(2, np.float32),
           "base_prov": np.zeros(2, np.int8), "invcdf": invcdf.astype(np.float64)}
    if a is not None:
        pkg["cmodes"] = np.full((2, 1), a, np.float32)
        pkg["cdiag"] = np.full(2, np.sqrt(max(1.0 - a * a, 0.0)), np.float32)
    return pkg


def _z(pkg, seeds, **kw):
    return np.array([draw(pkg, seed=s, systot=False, **kw)["z"] for s in seeds])


def test_phi_is_standard_normal_cdf():
    from math import erf
    x = np.linspace(-4, 4, 17)
    ref = np.array([0.5 * (1 + erf(v / np.sqrt(2))) for v in x])
    np.testing.assert_allclose(_phi(x), ref, atol=2e-7)


def test_copula_preserves_uniform_marginal():
    # z_i = u_i = Φ(g_i); g_i has exact unit variance ⇒ u_i ~ U(0,1) per object.
    Z = _z(_toy_pkg(a=np.sqrt(0.8)), range(4000))
    assert abs(Z[:, 0].mean() - 0.5) < 0.02 and abs(Z[:, 1].mean() - 0.5) < 0.02
    assert abs(Z[:, 0].std() - 1 / np.sqrt(12)) < 0.02          # U(0,1) std = 0.2887
    # KS vs uniform: marginal is the SAME law the IID draw targets
    from scipy.stats import kstest
    assert kstest(Z[:, 0], "uniform").pvalue > 0.01


def test_copula_injects_cross_object_correlation():
    seeds = range(4000)
    z_iid = _z(_toy_pkg(a=None), seeds)                          # no modes ⇒ IID
    z_cop = _z(_toy_pkg(a=np.sqrt(0.8)), seeds)                  # ρ_Gauss = 0.8
    assert abs(np.corrcoef(z_iid[:, 0], z_iid[:, 1])[0, 1]) < 0.06
    assert np.corrcoef(z_cop[:, 0], z_cop[:, 1])[0, 1] > 0.6     # ≈ (6/π)arcsin(0.4)=0.78


def test_copula_false_reproduces_legacy_iid():
    seeds = range(50)
    np.testing.assert_array_equal(_z(_toy_pkg(a=0.9), seeds, copula=False),
                                  _z(_toy_pkg(a=None), seeds))   # forced IID == no-modes
    with pytest.raises(ValueError):
        draw(_toy_pkg(a=None), seed=0, systot=False, copula=True)  # no modes to force


def test_write_load_roundtrip_preserves_modes(tmp_path):
    pkg = _toy_pkg(a=np.sqrt(0.8))
    p = tmp_path / "toy.npz"
    write_package(pkg, str(p))
    lp = load_package(str(p))
    assert lp.get("cmodes") is not None and lp["cmodes"].shape == (2, 1)
    # auto-detects modes and draws the correlated catalog
    z = _z(lp, range(2000))
    assert np.corrcoef(z[:, 0], z[:, 1])[0, 1] > 0.6
