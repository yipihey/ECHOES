"""Provenance grouping + visualizer colour helpers."""
import numpy as np

from echoes.completion import (PROV, PROV_GROUP, PROV_COLOR, prov_registry,
                               group_registry)
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


def test_registries_are_single_source_of_truth():
    # every code has full metadata; the viewer manifest builds straight from this
    reg = prov_registry()
    for code in PROV.values():
        assert set(reg[code]) >= {"short_label", "label", "description", "color", "group"}
        assert reg[code]["color"] == PROV_COLOR[code]
        assert reg[code]["group"] == PROV_GROUP[code]
    # group registry: fiber-collision merges collided + zhost; every code covered once
    grp = group_registry()
    assert PROV["zhost"] in grp["completed:fiber-collision"]["codes"]
    assert PROV["collided"] in grp["completed:fiber-collision"]["codes"]
    covered = [c for meta in grp.values() for c in meta["codes"]]
    assert sorted(covered) == sorted(PROV.values())          # partition, no gaps/dupes
    # group colour matches its representative code's colour (kept palette)
    assert grp["completed:fiber-collision"]["color"] == PROV_COLOR[PROV["collided"]]
