"""Derived rest-frame absolute magnitudes + stellar masses (kcorrect) are sane and
deterministic, and are robust to a non-finite (u-band) entry."""
import numpy as np
import pytest

kcorrect = pytest.importorskip("kcorrect")          # optional dependency
from echoes.derived import derive_properties, add_derived


def _cmass_like(n=60, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.45, 0.65, n)
    # rough CMASS ugriz: red, i ~ 17.5-19.9
    i = rng.uniform(17.5, 19.9, n)
    mags = np.column_stack([i + 2.5, i + 1.5, i + 0.55, i, i - 0.35])   # u,g,r,i,z
    mags[0, 0] = np.nan                                                # a bad u-band
    return mags.astype(np.float32), z


def test_absmag_and_mass_in_cmass_range():
    mags, z = _cmass_like()
    d = derive_properties(mags, z)
    assert d["absmag"].shape == (len(z), 5)
    assert -25 < np.nanmedian(d["absmag"][:, 3]) < -22          # M_i of CMASS LRGs
    assert 10.5 < np.nanmedian(d["logmass"]) < 12.0            # log10 M*/Msun
    assert np.isfinite(d["logmass"]).mean() > 0.9


def test_robust_to_bad_uband():
    mags, z = _cmass_like()
    d = derive_properties(mags, z)
    assert np.isfinite(d["logmass"][0]) and np.isfinite(d["absmag"][0, 3])   # bad u didn't break row 0


def test_deterministic_and_add_derived():
    mags, z = _cmass_like()
    a = derive_properties(mags, z)["logmass"]
    b = derive_properties(mags, z)["logmass"]
    np.testing.assert_array_equal(a, b)
    cat = {"mags": mags, "z": z}
    add_derived(cat)
    assert "absmag" in cat and "logmass" in cat and cat["absmag"].shape == (len(z), 5)
