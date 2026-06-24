"""Anisotropic-covariance parity: GraphGP.jl (Julia bridge) vs the Python fork's K(Δspatial, Δz).

The BOSS field is anisotropic in observed coordinates — the correlation depends separately on angular
separation Δθ and redshift separation Δz. The fork (``~/Projects/graphgp/graphgp/aniso.py``) supports
this; GraphGP.jl now does too. This gate proves the Julia engine reproduces the fork's anisotropic
``generate`` element-for-element, so BOSS can swap to the Julia backend with NO method change.

The test grid has DISTINCT Δspatial vs Δz structure (different Matérn lengths per axis), so an
accidental transpose of the 2-D grid, or a collapse to an isotropic 4-D Euclidean kernel, produces an
order-1 error — the failure modes this is built to catch.

Requires the FORK graphgp (with ``build_anisotropic_covariance``) + the Julia toolchain; skipped else.
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import pytest

_FORK = os.path.expanduser("~/Projects/graphgp")
if os.path.isdir(_FORK):
    sys.path.insert(0, _FORK)

gp = pytest.importorskip("graphgp")
pytest.importorskip("jax")
if not hasattr(gp, "build_anisotropic_covariance"):
    pytest.skip("fork graphgp (anisotropic) not importable", allow_module_level=True)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from echoes import graphgp_julia as ggj
from echoes import graphgp_backend as gb

pytestmark = pytest.mark.skipif(not (os.path.exists(ggj.JULIA) and os.path.exists(ggj.DRIVER)),
                                reason="Julia GraphGP backend not available")


def _matern1(d, ell):
    u = np.sqrt(3.0) * np.asarray(d, np.float64) / ell
    return (1.0 + u) * np.exp(-u)


def _aniso_cov(alpha=2.0, jitter=1e-3):
    """A genuinely anisotropic K(Δspatial, Δz): a tensor Matérn with a SHORT angular length and a
    LONG redshift length (so the two axes are not interchangeable)."""
    sb = np.concatenate([[0.0], np.geomspace(1e-4, 0.4, 63)]).astype(np.float64)   # chord (Δθ) bins
    zb = np.concatenate([[0.0], np.geomspace(1e-4, 0.10, 47)]).astype(np.float64)  # Δz bins
    grid = np.outer(_matern1(sb, 0.02), _matern1(zb, 0.05))                        # (n_s, n_z)
    return gp.build_anisotropic_covariance(jnp.asarray(sb), jnp.asarray(zb), jnp.asarray(grid),
                                           float(alpha), jitter=jitter)


def test_gate_aniso_generate_parity():
    rng = np.random.default_rng(0)
    n, n0, k, alpha = 2000, 64, 16, 2.0
    # embed (n̂, α·z): random sky directions + CMASS-like redshifts
    v = rng.standard_normal((n, 3)); nhat = v / np.linalg.norm(v, axis=1, keepdims=True)
    z = rng.uniform(0.45, 0.60, n)
    points = np.hstack([nhat, (alpha * z)[:, None]])                # (n,4) embedded
    cov = _aniso_cov(alpha=alpha)

    work = tempfile.mkdtemp(prefix="echoes_aniso_")
    try:
        in_npz = os.path.join(work, "g.npz")
        graph, scale = gb.build_graph_npz(points, n0, k, cov, in_npz)   # dumps coords + aniso grid
        d = dict(np.load(in_npz))
        assert "aniso_grid" in d and "aniso_alpha" in d, "aniso fields not carried into the NPZ"

        # JAX reference on the SAME lattice geometry + the SAME (f32-cast) grid the bridge dumped, so
        # the only thing under test is the numerics + the 2-D lookup, not f32 rounding.
        coords = d["coords"].astype(np.float64)
        ref_pts = jnp.asarray(scale * coords)
        ref_graph = gp.Graph(points=ref_pts, neighbors=jnp.asarray(d["neighbors"].astype(np.int32)),
                             offsets=tuple(int(x) for x in d["offsets"]),
                             indices=jnp.asarray(d["indices"].astype(np.int64)))
        cov_ref = gp.build_anisotropic_covariance(
            jnp.asarray(d["aniso_spatial_bins"].astype(np.float64)),
            jnp.asarray(d["aniso_z_bins"].astype(np.float64)),
            jnp.asarray(d["aniso_grid"].astype(np.float64)),       # f32-cast grid -> f64 (jitter in)
            float(d["aniso_alpha"]), jitter=0.0)
        xi = rng.standard_normal(n)
        jax_out = np.asarray(gp.generate(ref_graph, cov_ref, jnp.asarray(xi)), np.float64)

        jl_out = gb.generate_field(points, cov, xi, n0=n0, k=k, backend="julia",
                                   graph_npz=in_npz, dtype="f64")
        amax = float(np.max(np.abs(jax_out - jl_out)))
        rel = amax / (float(np.max(np.abs(jax_out))) + 1e-30)
        print(f"\nANISO GATE  max|jax-julia|={amax:.3e}  rel={rel:.3e}")
        assert not np.any(np.isnan(jl_out))
        assert np.allclose(jax_out, jl_out, atol=1e-9, rtol=1e-7), \
            f"anisotropic generate parity failed (max abs {amax:.3e}); transpose/iso-collapse bug?"
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    test_gate_aniso_generate_parity()
    print("ANISO PARITY PASSED")
