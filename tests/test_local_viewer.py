"""Geometry helpers for the textured local viewers (GPU-free; pure math)."""
import numpy as np

# a synthetic atlas manifest: 4096px sheets, 128px tiles, 2px gutter -> 32x32 = 1024 tiles/sheet
M = {"tile_px": 128, "gutter_px": 2, "sheet_px": 4096, "tiles_per_row": 32, "tiles_per_sheet": 1024}


def test_tile_uv_within_tile_bounds_and_ordered():
    from pipeline.build_local_viewer import _tile_uv
    for idx in (0, 1, 31, 32, 1023, 1024, 2000):    # spans sheet 0, row/col edges, sheet 1
        u0, v0, u1, v1 = _tile_uv(idx, M)
        assert 0.0 <= u0 < u1 <= 1.0 and 0.0 <= v0 < v1 <= 1.0
        # the rect is one inner tile wide (tile - 2*gutter) / sheet
        inner = (M["tile_px"] - 2 * M["gutter_px"]) / M["sheet_px"]
        assert abs((u1 - u0) - inner) < 1e-9 and abs((v1 - v0) - inner) < 1e-9
    # tile 0 starts at the gutter, tile 1 is one tile to the right
    assert _tile_uv(1, M)[0] > _tile_uv(0, M)[0]
    # tile 32 (next row) wraps to col 0 but a row down
    assert abs(_tile_uv(32, M)[0] - _tile_uv(0, M)[0]) < 1e-9 and _tile_uv(32, M)[1] > _tile_uv(0, M)[1]


def test_fork_tile_rects_match_scalar_helper():
    from pipeline.build_local_viewer import _tile_uv
    from pipeline.build_local_viewer_fork import _tile_rects
    idx = np.array([0, 5, 33, 1024, 2047])
    rects = _tile_rects(idx, M)
    assert rects.shape == (5, 4)
    for k, i in enumerate(idx):
        assert np.allclose(rects[k], _tile_uv(int(i), M), atol=1e-6)


def test_tangent_basis_orthonormal_and_perp_to_radial():
    from pipeline.build_local_viewer import _tangent_basis
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(50, 3)) * 100.0
    xyz[0] = [0, 0, 120.0]                            # a near-pole sightline (degenerate east)
    n, east, north = _tangent_basis(xyz)
    for a in (n, east, north):
        assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-6)
    assert np.all(np.abs(np.einsum("ij,ij->i", n, east)) < 1e-6)
    assert np.all(np.abs(np.einsum("ij,ij->i", n, north)) < 1e-6)
    assert np.all(np.abs(np.einsum("ij,ij->i", east, north)) < 1e-6)
    # n is the unit radial direction
    assert np.allclose(n, xyz / np.linalg.norm(xyz, axis=1, keepdims=True), atol=1e-6)
