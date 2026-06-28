"""Registry + the two completion modes (restore_missing, resample_redshift).

Exercises the dataset-agnostic seam on tiny synthetic catalogs (no survey data
needed): the registry round-trips a spec; ``synth_resample_targets`` shapes a
resample target set; ``build_package`` runs both modes and ``draw`` reconstructs a
realization with the right provenance and redshift support; the ``photoz=None``
path yields a field-only posterior.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from echoes.surveys import (DatasetSpec, CompletionTargets, register, get, available,
                            synth_resample_targets, ResampleTargets)
from echoes.posterior import build_package, draw, write_package, load_package
from echoes.completion import PROV


# ---- tiny synthetic catalogs -------------------------------------------------

class _Cat:
    """Minimal SurveyCatalog-ish object for the unit tests."""
    def __init__(self, ra, dec, z, z_err=None, wsys=None, mags=None):
        self.ra_data = ra; self.dec_data = dec; self.z_data = z
        self.z_data_err = z_err
        self.w_sys_data = wsys
        self.mags_data = mags
        self.nside = 64


class _Targets:
    """Minimal restore_missing CompletionTargets (DESI-like: no colours)."""
    def __init__(self, ra, dec, host, kind):
        self.ra = ra; self.dec = dec
        self.colors = np.empty((len(ra), 0)); self.mags = None
        self.host_index = np.asarray(host); self.miss_kind = np.asarray(kind)

    @property
    def N(self):
        return len(self.ra)


def _sky(n, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(150, 160, n), rng.uniform(-2, 2, n)


# ---- registry ----------------------------------------------------------------

def test_registry_roundtrip():
    spec = DatasetSpec(name="_unit_test_ds", load_catalog=lambda **k: None,
                       completion_mode="resample_redshift", product_prefix="ut")
    register(spec)
    assert "_unit_test_ds" in available()
    assert get("_unit_test_ds") is spec
    assert spec.prefix == "ut"
    with pytest.raises(KeyError):
        get("_does_not_exist")


def test_resample_targets_shape():
    ra, dec = _sky(50, 1)
    z = np.random.default_rng(2).uniform(0.5, 3.0, 50)
    zerr = np.full(50, 0.05)
    cat = _Cat(ra, dec, z, z_err=zerr)
    tg = synth_resample_targets(cat)
    assert isinstance(tg, ResampleTargets) and isinstance(tg, CompletionTargets)
    assert tg.N == 50 and tg.colors.shape == (50, 0) and tg.mags is None
    assert np.array_equal(tg.host_index, np.arange(50))
    assert (tg.miss_kind == "photoz").all()
    assert np.array_equal(tg.photoz_mu, z) and np.array_equal(tg.photoz_sigma, zerr)
    # missing z_data_err is an explicit error
    with pytest.raises(ValueError):
        synth_resample_targets(_Cat(ra, dec, z))


# ---- resample_redshift mode --------------------------------------------------

def test_resample_mode_endtoend():
    n = 400
    ra, dec = _sky(n, 7)
    rng = np.random.default_rng(7)
    z_true = rng.uniform(0.8, 2.5, n)
    z_obs = z_true + rng.normal(0, 0.05, n)            # photo-z point estimate
    zerr = np.full(n, 0.05)
    cat = _Cat(ra, dec, z_obs, z_err=zerr)
    tg = synth_resample_targets(cat)

    pkg = build_package(cat, tg, photoz=None, mode="resample_redshift", copula=False)
    assert pkg["package_mode"] == "resample_redshift"
    assert pkg["n_obs"] == 0 and pkg["n_miss"] == n            # all objects resampled, no fixed backbone
    assert pkg["obs_z"].size == 0

    out = draw(pkg, seed=0, systot=False)
    assert out["N"] == n
    assert (out["prov"] == PROV["photoz"]).all()               # every object: photo-z resampled
    assert np.isfinite(out["z"]).all()
    assert (out["z"] >= pkg["zmin"] - 1e-3).all() and (out["z"] <= pkg["zmax"] + 1e-3).all()
    # positions are FIXED (data), only z varies between seeds
    out2 = draw(pkg, seed=1, systot=False)
    assert np.array_equal(out["ra"], out2["ra"]) and np.array_equal(out["dec"], out2["dec"])
    assert not np.allclose(out["z"], out2["z"])                # redshifts genuinely resampled

    # round-trips through the released .npz unchanged
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ut_resample.npz")
        write_package(pkg, p)
        pk2 = load_package(p)
        assert pk2["package_mode"] == "resample_redshift" and pk2["n_obs"] == 0
        o2 = draw(pk2, seed=0, systot=False)
        assert o2["N"] == n and (o2["prov"] == PROV["photoz"]).all()


def test_resample_field_sharpens_toward_structure():
    """The local-density prior should pull a broad photo-z toward real structure:
    plant a tight redshift overdensity on a patch of sky and check resampled z there
    concentrate more than the raw photo-z width."""
    rng = np.random.default_rng(11)
    n = 600
    ra, dec = _sky(n, 11)
    # a compact sky cluster all at z~1.5; the rest spread out
    in_clump = (ra < 151.5) & (dec < -1.0)
    z_obs = rng.uniform(0.8, 2.5, n)
    z_obs[in_clump] = 1.5 + rng.normal(0, 0.02, in_clump.sum())
    zerr = np.full(n, 0.08)                                    # broad photo-z
    cat = _Cat(ra, dec, z_obs, z_err=zerr)
    tg = synth_resample_targets(cat)
    pkg = build_package(cat, tg, photoz=None, mode="resample_redshift",
                        anchor_frac=1.0, copula=False)          # all objects are anchors here
    # posterior median spread within the clump should be far below the 0.08 photo-z width
    q = np.asarray(pkg["qlev"]); iv = np.asarray(pkg["invcdf"])
    zmed = np.array([np.interp(0.5, q, iv[i]) for i in range(len(iv))])
    assert np.std(zmed[in_clump]) < 0.05


# ---- restore_missing with no colours (DESI-like) -----------------------------

def test_restore_missing_no_photoz_field_only():
    n_obs = 800
    ra_o, dec_o = _sky(n_obs, 3)
    rng = np.random.default_rng(3)
    z_o = rng.uniform(0.8, 2.1, n_obs)
    cat = _Cat(ra_o, dec_o, z_o, wsys=np.ones(n_obs))
    # restore ~100 coverage-missing objects, each a clone of a random host
    host = rng.integers(0, n_obs, 100)
    tg = _Targets(ra_o[host] + 1e-3, dec_o[host] + 1e-3, host, np.full(100, "coverage"))

    pkg = build_package(cat, tg, photoz=None, mode="restore_missing", copula=False)
    assert pkg["package_mode"] == "restore_missing"
    assert pkg["n_obs"] == n_obs and pkg["n_miss"] == 100

    out = draw(pkg, seed=0, systot=False)
    assert out["N"] == n_obs + 100
    assert (out["prov"][:n_obs] == PROV["observed"]).all()
    assert (out["prov"][n_obs:] == PROV["coverage"]).all()      # DESI coverage provenance
    assert np.isfinite(out["z"]).all()
    # observed redshifts are the fixed backbone
    assert np.allclose(out["z"][:n_obs], z_o.astype(np.float32), atol=1e-3)
