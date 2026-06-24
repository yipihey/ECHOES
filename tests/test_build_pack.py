"""Extension-pack builder + manifest integrity (GPU-free).

Unit-tests the reusable pack-io + layer builder on synthetic data, and — if the committed
local-2mpp pack is present — verifies its on-disk integrity (every chunk's size + sha256), the
core-tier byte budget that structurally guarantees fast startup, and the registry shape.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
from _pack_io import write_array, write_json   # noqa: E402
import build_pack   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "docs" / "packs" / "local-2mpp"
REGISTRY = ROOT / "docs" / "packs" / "packs.json"
CORE_BUDGET = 1_500_000          # the <~2s-startup contract, enforced structurally


def test_write_array_roundtrip_and_descriptor(tmp_path):
    arr = np.arange(12, dtype=np.float32).reshape(4, 3)
    d = write_array(tmp_path, "sub/x.f32.bin", arr, "<f4")
    p = tmp_path / "sub" / "x.f32.bin"
    assert p.stat().st_size == d["bytes"] == 48 and d["count"] == 12 and d["shape"] == [4, 3]
    assert hashlib.sha256(p.read_bytes()).hexdigest() == d["sha256"]
    assert np.array_equal(np.frombuffer(p.read_bytes(), np.float32).reshape(4, 3), arr)


def test_point_layer_schema(tmp_path):
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(100, 3)).astype(np.float32) * 50.0
    val = rng.uniform(8, 12, 100).astype(np.float32)
    L = build_pack._point_layer(tmp_path, "obs", "core", 0, "#9aa0a6", xyz, val)
    assert L["tier"] == "core" and L["count"] == 100 and set(L["columns"]) == {"xyz", "value"}
    assert len(L["bbox"]) == 6 and L["bbox"][0] <= L["bbox"][3]      # min <= max per axis
    assert (tmp_path / L["columns"]["xyz"]["file"]).exists()


@pytest.mark.skipif(not PACK.exists(), reason="local-2mpp pack not built")
def test_committed_pack_integrity():
    m = json.loads((PACK / "pack_manifest.json").read_text())
    assert m["schema_version"] == "echoes.pack.v1" and m["pack"]["id"] == "local-2mpp"
    tier_names = set(m["tiers"])
    core_bytes = 0
    for L in m["layers"]:
        assert L["tier"] in tier_names                              # every tier is declared
        for c in L["columns"].values():
            p = PACK / c["file"]
            assert p.stat().st_size == c["bytes"], f"size drift: {c['file']}"
            assert hashlib.sha256(p.read_bytes()).hexdigest() == c["sha256"], f"sha drift: {c['file']}"
            if L["tier"] == "core":
                core_bytes += c["bytes"]
    assert core_bytes < CORE_BUDGET, f"core {core_bytes} exceeds fast-start budget {CORE_BUDGET}"
    # the cartesian/true-3D coordinate contract the viewer keys on
    assert m["pack"]["coordinate"] == "cartesian_mpc"


@pytest.mark.skipif(not REGISTRY.exists(), reason="registry not built")
def test_registry_shape_and_resolution():
    r = json.loads(REGISTRY.read_text())
    assert r["schema_version"] == "echoes.packs.v1"
    ids = [p["id"] for p in r["packs"]]
    assert "boss-cmass" in ids and len(ids) == len(set(ids))          # unique ids, BOSS present
    for p in r["packs"]:
        assert {"id", "title", "version", "kind", "manifest_url", "requires"} <= set(p)
        # manifest_url resolves (relative to packs.json's dir)
        assert (REGISTRY.parent / p["manifest_url"]).resolve().exists(), p["manifest_url"]
        for req in p["requires"]:                                    # requires reference real packs
            assert req in ids
