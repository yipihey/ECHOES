"""Every ECHOES module imports cleanly (without the optional graphgp/jax)."""
import importlib
import pytest

MODULES = [
    "echoes", "echoes.completion", "echoes.photoz", "echoes.posterior",
    "echoes.graphgp_field", "echoes.clustering", "echoes.ls_corrfunc",
    "echoes.field_kernel", "echoes.randoms", "echoes.geometry", "echoes.distance",
    "echoes.perf", "echoes.inpaint", "echoes.selection_coupling",
    "echoes.mock_systematics", "echoes.cli",
    "echoes.surveys.base", "echoes.surveys.sdss_io", "echoes.surveys.boss",
    "echoes.surveys.boss_targets",
    # experimental kNN2D engine (Yuan-Abel-Wechsler) — needs numba + healpy
    "echoes.knn", "echoes.knn.cdf", "echoes.knn.derived", "echoes.knn.analytic_rr",
    "echoes.knn2d_field",
]

@pytest.mark.parametrize("mod", MODULES)
def test_import(mod):
    importlib.import_module(mod)
