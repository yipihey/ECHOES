"""The released posterior draws the documented BOSS CMASS-South catalogs."""
import os
import numpy as np
import pytest
from echoes import load_package, draw

PKG = os.path.join(os.path.dirname(__file__), "..", "data_release",
                   "cmass_south_posterior.npz")
pytestmark = pytest.mark.skipif(not os.path.exists(PKG), reason="posterior package not present")


def test_seed0_census():
    cat = draw(load_package(PKG), seed=0)
    counts = dict(zip(*[a.tolist() for a in np.unique(cat["prov"], return_counts=True)]))
    assert counts[0] == 109636          # observed
    assert counts[1] == 5272            # fiber-collided
    assert counts[2] == 1505            # redshift-failure
    assert counts[3] == 3472            # imaging-systematic analog (RNG-stream dependent)
    assert cat["N"] == 119885


def test_schema_and_determinism():
    pkg = load_package(PKG)
    a = draw(pkg, seed=7)
    b = draw(pkg, seed=7)
    for k in ("ra", "dec", "z", "prov"):
        assert k in a and len(a[k]) == a["N"]
        np.testing.assert_array_equal(a[k], b[k])   # same seed -> identical


def test_observed_fixed_missing_varies():
    pkg = load_package(PKG)
    n_obs = pkg["n_obs"]
    a, b = draw(pkg, seed=0), draw(pkg, seed=1)
    np.testing.assert_array_equal(a["z"][:n_obs], b["z"][:n_obs])        # observed shared
    assert not np.array_equal(a["z"][n_obs:n_obs + pkg["n_miss"]],
                              b["z"][n_obs:n_obs + pkg["n_miss"]])        # missing differ


def test_cli_npz_output(tmp_path):
    from echoes.cli import draw_main

    out = tmp_path / "catalog_0.npz"
    draw_main(["--package", PKG, "--seed", "0", "--out", str(out)])
    d = np.load(out)
    # photometry columns present iff the package carries base_mags (the released one does)
    assert {"ra", "dec", "z", "prov"} <= set(d.files)
    if "mags" in d.files:
        assert d["mags"].shape == (119885, 5) and d["colors"].shape == (119885, 4)
    assert len(d["ra"]) == 119885
