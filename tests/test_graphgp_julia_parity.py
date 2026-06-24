"""Parity gates for the Julia GraphGP backend bridge (``echoes/graphgp_julia.py``).

The Julia rewrite is an independent reimplementation of the GraphGP forward/inverse maps. The one
thing the bench harness historically dropped — and the thing that silently scrambles a field if it is
wrong — is the ``graph.indices`` permutation (build/tree order ↔ original point order). These gates
exercise a *non-trivial* permutation and check the bridge end to end:

  * **Gate 0** — Julia ``generate`` equals the JAX reference ``graphgp.generate`` element-for-element
    (f64, ``atol≈1e-10``) in **original point order**. A dropped/mis-applied ``indices`` produces an
    order-1 scramble, so this is the load-bearing correctness gate.
  * **Gate 1** — Julia ``generate_inv ∘ generate ≈ identity`` on a real graph (original order),
    confirming the inverse map and the output scatter the driver applies.

Requires the Julia toolchain + the path-dependent ``graphgp-julia`` checkout; skipped otherwise.
"""
import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

pytest.importorskip("graphgp")
pytest.importorskip("jax")

import jax

# f64 numerics for the parity comparison; build_graph_npz transiently disables x64 for the
# (precision-insensitive) k-d tree build, which otherwise hits an lax.cond f32/f64 branch mismatch.
jax.config.update("jax_enable_x64", True)

from echoes import graphgp_julia as ggj  # noqa: E402

# Skip cleanly when the Julia side is not installed on this machine.
_HAVE_JULIA = os.path.exists(ggj.JULIA) and os.path.exists(ggj.DRIVER)
pytestmark = pytest.mark.skipif(not _HAVE_JULIA,
                                reason="Julia GraphGP backend (julia + run_graphgp.jl) not available")


def _strexp_kernel(r0=2.5, alpha=1.5, amp=1.0, rmax=18.0, n=600):
    """Stretched-exponential ξ(r) — the ECHOES ``tabulate_kernel`` form. alpha<2 keeps the binned
    covariance well-conditioned; inflate k(0) slightly so the dense first block is PD."""
    r = np.linspace(0.0, rmax, n)
    k = amp * np.exp(-((r / r0) ** alpha))
    k = k.copy()
    k[0] *= 1.0 + 1e-6
    return r.astype(np.float64), k.astype(np.float64)


def _build(points, n0, k, cov, work):
    """Build the bridge NPZ and return (in_npz, dumped-dict)."""
    in_npz = os.path.join(work, "graph.npz")
    graph, scale = ggj.build_graph_npz(points, n0, k, cov[0], cov[1], in_npz)
    assert graph.indices is not None, "test needs a non-trivial permutation to exercise indices"
    d = dict(np.load(in_npz))
    assert "indices" in d
    perm = d["indices"]
    assert not np.array_equal(perm, np.arange(len(perm))), "permutation is trivial (identity)"
    return in_npz, d, scale


def _jax_reference(d, scale, xi):
    """JAX ``graphgp.generate`` on a graph reconstructed from the SAME integer lattice the Julia side
    consumes (f64 ``scale·coords``, no origin — distances are origin-invariant), so the only thing
    under test is the numerics + the ``indices`` carry, not f32 geometry rounding."""
    import jax.numpy as jnp
    import graphgp as gp

    coords = d["coords"].astype(np.float64)
    points = jnp.asarray(scale * coords)                       # f64, matches Julia geometry exactly
    neighbors = jnp.asarray(d["neighbors"].astype(np.int32))
    offsets = tuple(int(x) for x in d["offsets"])
    indices = jnp.asarray(d["indices"].astype(np.int64))
    graph = gp.Graph(points=points, neighbors=neighbors, offsets=offsets, indices=indices)
    cov = (jnp.asarray(d["cov_bins32"].astype(np.float64)),
           jnp.asarray(d["cov_vals32"].astype(np.float64)))
    return np.asarray(gp.generate(graph, cov, jnp.asarray(xi)), np.float64)


def test_gate0_generate_parity_original_order():
    """Gate 0: Julia generate == JAX generate, element-wise, in ORIGINAL order."""
    rng = np.random.default_rng(0)
    n, ndim, n0, k = 2000, 3, 64, 16
    points = rng.uniform(0.0, 40.0, (n, ndim))
    cov = _strexp_kernel()
    xi = rng.standard_normal(n)

    work = tempfile.mkdtemp(prefix="echoes_ggp_parity_")
    try:
        in_npz, d, scale = _build(points, n0, k, cov, work)
        jax_out = _jax_reference(d, scale, xi)
        jl = ggj.run_graphgp(points, n0, k, cov[0], cov[1], ops=("generate",), xi=xi,
                             dtype="f64", _graph_npz=in_npz, work_dir=work)
        jl_out = np.asarray(jl["generate"])[0]                 # (1, N) -> (N,), original order

        assert jl_out.shape == jax_out.shape
        assert not np.any(np.isnan(jl_out)), "Julia generate returned NaN"
        amax = float(np.max(np.abs(jax_out - jl_out)))
        rel = amax / float(np.max(np.abs(jax_out)) + 1e-30)
        print(f"\nGATE 0  max|jax-julia|={amax:.3e}  rel={rel:.3e}")
        assert np.allclose(jax_out, jl_out, atol=1e-9, rtol=1e-7), \
            f"generate parity failed (max abs {amax:.3e}); a dropped `indices` would scramble this"
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_gate1_generate_inverse_roundtrip():
    """Gate 1: generate_inv(generate(xi)) ≈ xi on a real graph (original order)."""
    rng = np.random.default_rng(1)
    n, ndim, n0, k = 1500, 3, 48, 16
    points = rng.uniform(0.0, 30.0, (n, ndim))
    cov = _strexp_kernel()
    xi = rng.standard_normal(n)

    work = tempfile.mkdtemp(prefix="echoes_ggp_rt_")
    try:
        in_npz, d, scale = _build(points, n0, k, cov, work)
        fwd = ggj.run_graphgp(points, n0, k, cov[0], cov[1], ops=("generate",), xi=xi,
                              dtype="f64", _graph_npz=in_npz, work_dir=work)
        values = np.asarray(fwd["generate"])[0]                # original order
        inv = ggj.run_graphgp(points, n0, k, cov[0], cov[1], ops=("generate_inv",), values=values,
                              dtype="f64", _graph_npz=in_npz, work_dir=work)
        xi_rec = np.asarray(inv["generate_inv"]).ravel()       # original order

        assert xi_rec.shape == xi.shape
        amax = float(np.max(np.abs(xi_rec - xi)))
        print(f"\nGATE 1  max|xi - inv(gen(xi))|={amax:.3e}")
        assert np.allclose(xi_rec, xi, atol=1e-6, rtol=1e-5), \
            f"roundtrip failed (max abs {amax:.3e})"
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_gate2_backend_field_equivalence():
    """Gate 2: the ``graphgp_backend.generate_field`` switch yields the SAME field for the same xi
    under either backend (deterministic; the field is L·xi). Validates the P1 heavy-Vecchia drop-in.
    Tolerance is loose (f32 GPU/quantization, not f64 parity) but far below the order-1 difference a
    wrong graph/permutation would cause."""
    import jax.numpy as jnp
    import graphgp as gp
    from echoes import graphgp_backend as gb

    rng = np.random.default_rng(2)
    n, ndim, n0, k = 1800, 3, 48, 16
    # mildly clustered points (two blobs) so the graph is non-uniform like a real field
    a = rng.normal([5, 5, 5], 3.0, (n // 2, ndim))
    b = rng.normal([25, 20, 15], 4.0, (n - n // 2, ndim))
    points = np.vstack([a, b])
    cov = _strexp_kernel()
    xi = rng.standard_normal((n, 2))                            # batch of 2 fields

    work = tempfile.mkdtemp(prefix="echoes_ggp_gate2_")
    try:
        # share ONE graph so the only thing under test is the numerics, not graph-build determinism
        in_npz, d, scale = _build(points, n0, k, cov, work)
        ref_graph = gp.Graph(points=jnp.asarray(scale * d["coords"].astype(np.float64)),
                             neighbors=jnp.asarray(d["neighbors"].astype(np.int32)),
                             offsets=tuple(int(x) for x in d["offsets"]),
                             indices=jnp.asarray(d["indices"].astype(np.int64)))
        f_jax = gb.generate_field(points, cov, xi, n0=n0, k=k, backend="jax", graph=ref_graph)
        f_jul = gb.generate_field(points, cov, xi, n0=n0, k=k, backend="julia", device="cpu",
                                  graph_npz=in_npz, dtype="f64")
        assert f_jax.shape == f_jul.shape == (2, n)
        rel = float(np.linalg.norm(f_jax - f_jul) / (np.linalg.norm(f_jax) + 1e-30))
        print(f"\nGATE 2  L2-rel(jax,julia) field ={rel:.3e}")
        # ~1e-7: the bridge stores the kernel TABLE as f32 (cov_vals32); the numerics are f64-identical
        # (Gate 0 = 5e-15). Far below the O(1) error a wrong graph/permutation would produce.
        assert rel < 1e-5, f"backend field mismatch (L2-rel {rel:.3e}); a wrong graph would be O(1)"
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    test_gate0_generate_parity_original_order()
    test_gate1_generate_inverse_roundtrip()
    test_gate2_backend_field_equivalence()
    print("ALL GATES PASSED")
