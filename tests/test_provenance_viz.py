"""Provenance grouping + visualizer colour helpers."""
import numpy as np

from echoes.completion import PROV, PROV_GROUP, PROV_COLOR
from tools.viz_provenance import prov_rgba, prov_k3d_colors, _group_of


def test_every_prov_code_has_group_and_colour():
    for code in PROV.values():
        assert code in PROV_GROUP
        assert code in PROV_COLOR


def test_groups_separate_inpaint_from_completed_and_by_kind():
    # the three origins the user must be able to separate
    assert PROV_GROUP[PROV["observed"]] == "observed"
    assert PROV_GROUP[PROV["collided"]].startswith("completed:fiber-collision")
    assert PROV_GROUP[PROV["zhost"]].startswith("completed:fiber-collision")
    assert PROV_GROUP[PROV["zfail"]].startswith("completed:redshift-failure")
    assert PROV_GROUP[PROV["systot"]].startswith("inpainted")
    assert PROV_GROUP[PROV["inpaint"]].startswith("inpainted")
    # completed != inpainted, and the two completed kinds differ
    assert PROV_GROUP[PROV["collided"]] != PROV_GROUP[PROV["zfail"]]
    assert PROV_GROUP[PROV["collided"]] != PROV_GROUP[PROV["systot"]]


def test_prov_rgba_and_k3d_colours_match_mapping():
    codes = [PROV["observed"], PROV["collided"], PROV["zfail"], PROV["systot"]]
    prov = np.array(codes)
    rgba = prov_rgba(prov, alpha=0.7)
    assert rgba.shape == (4, 4) and np.allclose(rgba[:, 3], 0.7)
    # rgba reproduces the canonical PROV_COLOR hex for every code (palette-agnostic)
    for row, code in enumerate(codes):
        h = PROV_COLOR[code].lstrip("#")
        want = np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0
        assert np.allclose(rgba[row, :3], want, atol=1 / 255)
    k = prov_k3d_colors(prov)
    assert k.dtype == np.uint32
    # packed 0xRRGGBB equals the hex literal for each code
    for j, code in enumerate(codes):
        assert int(k[j]) == int(PROV_COLOR[code].lstrip("#"), 16)


def test_group_of_vectorised():
    prov = np.array([0, 1, 2, 3])
    g = _group_of(prov)
    assert len(g) == 4 and g[0] == "observed"
