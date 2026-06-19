import json
from pathlib import Path

import numpy as np
import pytest

from pipeline.build_viewer_bundle import DEFAULT_PACKAGE, DEFAULT_SOURCE, build_viewer_bundle


pytestmark = pytest.mark.skipif(not DEFAULT_PACKAGE.exists(), reason="posterior package not present")


def test_viewer_bundle_manifest_and_seed0(tmp_path):
    out = tmp_path / "visualizer"
    manifest_path = build_viewer_bundle(
        package=DEFAULT_PACKAGE,
        source=DEFAULT_SOURCE,
        out=out,
        seeds=[0],
    )
    manifest = json.loads(manifest_path.read_text())

    assert manifest["schema_version"] == "echoes.viewer.v1"
    assert manifest["counts"] == {"observed": 109636, "missing": 6777, "base": 116413}
    assert {c["id"] for c in manifest["columns"]} >= {
        "ra", "dec", "z", "weight_systot", "provenance", "source",
    }
    assert manifest["enriched_bundle"]["supported"] is True

    method = manifest["methods"][0]
    assert method["id"] == "knn-field"
    realization = method["realizations"][0]
    assert realization["id"] == "seed-0000"
    assert realization["total_count"] == 119923
    assert realization["provenance_counts"] == {
        "0": 109636,
        "1": 5272,
        "2": 1505,
        "3": 3510,
    }

    for group in (manifest["base"]["columns"], realization["chunks"]):
        for desc in group.values():
            path = manifest_path.parent / desc["file"]
            assert path.exists(), desc["file"]
            assert path.stat().st_size == desc["bytes"]

    assert (out / "index.html").exists()
    assert (out / "app.js").exists()
    assert (out / "styles.css").exists()


def test_viewer_bundle_accepts_enriched_columns(tmp_path):
    enriched = tmp_path / "enriched.npz"
    np.savez(
        enriched,
        WEIGHT_CP=np.ones(109636, dtype=np.float32),
        SECTOR=np.zeros(109636, dtype=np.uint8),
    )
    out = tmp_path / "visualizer_enriched"
    manifest_path = build_viewer_bundle(
        package=DEFAULT_PACKAGE,
        source=DEFAULT_SOURCE,
        out=out,
        seeds=[0],
        enriched_npz=enriched,
    )
    manifest = json.loads(manifest_path.read_text())
    column_ids = {c["id"] for c in manifest["columns"]}
    assert {"weight_cp", "sector"} <= column_ids
    assert manifest["enriched_bundle"]["columns_added"] == ["sector", "weight_cp"]
    assert (manifest_path.parent / manifest["base"]["columns"]["weight_cp"]["file"]).exists()
