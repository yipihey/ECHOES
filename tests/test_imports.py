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
    # field-level reconstruction (needs jax/graphgp for the conditional solve)
    "echoes.pit", "echoes.selection_model", "echoes.field_posterior", "echoes.fieldpost",
]

@pytest.mark.parametrize("mod", MODULES)
def test_import(mod):
    importlib.import_module(mod)
