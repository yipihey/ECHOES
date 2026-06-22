"""Completed catalogs carry real ugriz mags + colors; the package round-trips them
(per-band NaN preserved), systot extras inherit the source photometry, the reproducer
is bit-for-bit, and legacy packages without photometry still draw."""
import sys, os
import numpy as np
import pytest
from echoes.posterior import draw, write_package, load_package

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_release"))
import draw_samples as DS


def _toy(n_obs=6, n_miss=4, with_mags=True):
    nq = 9
    qlev = np.linspace(0.0, 1.0, nq)
    invcdf = np.repeat(np.linspace(0.45, 0.65, nq)[None, :], n_miss, axis=0)
    nb = n_obs + n_miss
    pkg = {"n_obs": n_obs, "n_miss": n_miss, "zmin": 0.4, "zmax": 0.7,
           "qlev": qlev, "jitter": 0.0,
           "obs_z": np.linspace(0.45, 0.65, n_obs).astype(np.float32),
           "base_ra": np.linspace(10.0, 20.0, nb).astype(np.float32),
           "base_dec": np.zeros(nb, np.float32),
           "base_wsys": np.full(nb, 1.6, np.float32),          # >1 → systot extras exist
           "base_prov": np.array([0] * n_obs + [1] * n_miss, np.int8),
           "invcdf": invcdf}
    if with_mags:
        rng = np.random.default_rng(1)
        m = rng.uniform(17.0, 21.0, (nb, 5)).astype(np.float32)
        m[0, 0] = np.nan                                       # a bad u-band (sentinel path)
        pkg["base_mags"] = m
    return pkg


def test_draw_emits_consistent_colors():
    pkg = _toy()
    c = draw(pkg, seed=0)
    assert {"mags", "colors", "colors_finite"} <= set(c)
    assert c["mags"].shape == (c["N"], 5) and c["colors"].shape == (c["N"], 4)
    np.testing.assert_allclose(c["colors"], c["mags"][:, :-1] - c["mags"][:, 1:],
                               rtol=0, atol=0, equal_nan=True)


def test_systot_extras_inherit_source_photometry():
    pkg = _toy()
    nb = pkg["n_obs"] + pkg["n_miss"]
    c = draw(pkg, seed=0)
    assert c["N"] > nb                                          # some systot extras added
    # base is emitted first, then the extras (each a copy of a base row)
    np.testing.assert_array_equal(c["mags"][:nb], pkg["base_mags"])
    extras = c["mags"][nb:]
    base_rows = {tuple(np.where(np.isnan(r), -999, r)) for r in pkg["base_mags"]}
    for r in extras:                                            # every extra equals some base row
        assert tuple(np.where(np.isnan(r), -999, r)) in base_rows


def test_write_load_roundtrip_preserves_mags_and_nan(tmp_path):
    pkg = _toy()
    p = tmp_path / "toy_mags.npz"
    write_package(pkg, str(p))
    lp = load_package(str(p))
    fin = np.isfinite(pkg["base_mags"])
    np.testing.assert_allclose(lp["base_mags"][fin], pkg["base_mags"][fin], atol=1e-3)
    assert np.isnan(lp["base_mags"][0, 0])                      # u sentinel → NaN preserved
    assert int(np.isnan(lp["base_mags"]).sum()) == int((~fin).sum())


def test_reproducer_lockstep_with_photometry(tmp_path):
    p = tmp_path / "toy_mags.npz"
    write_package(_toy(), str(p))
    a = draw(load_package(str(p)), seed=3)
    b = DS.draw(DS.load_package(str(p)), seed=3)
    for k in ("ra", "dec", "z", "prov", "mags", "colors"):
        np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(b[k]))   # NaN positions identical too


def test_legacy_package_without_photometry_draws(tmp_path):
    p = tmp_path / "toy_nomags.npz"
    write_package(_toy(with_mags=False), str(p))
    lp = load_package(str(p))
    assert lp["base_mags"] is None
    c = draw(lp, seed=0)
    assert "mags" not in c and c["N"] > 0
