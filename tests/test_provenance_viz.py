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
    prov = np.array([PROV["observed"], PROV["collided"], PROV["zfail"], PROV["systot"]])
    rgba = prov_rgba(prov, alpha=0.7)
    assert rgba.shape == (4, 4) and np.allclose(rgba[:, 3], 0.7)
    # collided is orange-ish (R>G>B); observed teal (G,B > R)
    assert rgba[1, 0] > rgba[1, 2]
    assert rgba[0, 1] > rgba[0, 0] and rgba[0, 2] > rgba[0, 0]
    k = prov_k3d_colors(prov)
    assert k.dtype == np.uint32
    # packed 0xRRGGBB equals the hex literal for collided
    assert int(k[1]) == int(PROV_COLOR[PROV["collided"]].lstrip("#"), 16)


def test_group_of_vectorised():
    prov = np.array([0, 1, 2, 3])
    g = _group_of(prov)
    assert len(g) == 4 and g[0] == "observed"
