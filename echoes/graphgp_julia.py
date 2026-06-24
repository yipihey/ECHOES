"""Bridge to the Julia GraphGP backend (``~/Projects/graphgp-julia/julia/GraphGP``).

The Julia rewrite reaches GPU parity-to-ahead of the hand-written CUDA reference, never OOMs (no
``(M,K+1,K+1)`` tensor → runs where pure-JAX dies), and uniquely provides analytic gradients. We reach
it over the **proven NPZ subprocess bridge** (Julia owns the GPU in its own process, so it never
fights JAX's CUDA on the shared A6000) — see ``julia/GraphGP/bench/compare/run_graphgp.jl``.

``run_graphgp`` builds the Vecchia graph in Python (the validated ``graphgp.build_graph``), dumps it —
**including ``graph.indices``**, the permutation the bench harness drops — calls the Julia driver, and
reads results back **in original point order**. The driver gathers ``xi`` by ``indices`` so a given
original-order ``xi`` produces the same field as Python ``graphgp.generate`` (element-wise parity gate).
"""
from __future__ import annotations

import contextlib
import json
import os
import subprocess
import tempfile

import numpy as np


@contextlib.contextmanager
def _x64_disabled():
    """Force ``jax_enable_x64`` off for the duration. The k-d tree build (``graphgp.tree``) has an
    ``lax.cond`` whose branches disagree on f32/f64 when x64 is enabled — and graph *topology* is
    precision-insensitive (integer lattice coords) — so we always build under f32. Restored on exit
    so a caller running the f64 numerics (the parity gate) is unaffected."""
    import jax

    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", False)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)

# locations (overridable via env for other machines)
JULIA = os.environ.get("ECHOES_JULIA", os.path.expanduser("~/.juliaup/bin/julia"))
GRAPHGP_JL = os.environ.get("ECHOES_GRAPHGP_JL",
                            os.path.expanduser("~/Projects/graphgp-julia/julia/GraphGP"))
DRIVER = os.path.join(GRAPHGP_JL, "bench", "compare", "run_graphgp.jl")
BENCH_PROJ = os.path.join(GRAPHGP_JL, "bench")
_LMAX = (1 << 21) - 1                                   # 21-bit/axis lattice (graphgp convention)


def _quantize(points):
    """Quantize float positions onto the 21-bit lattice graphGP consumes (origin offset cancels in
    distances). Returns ``(points_q, origin, scale)``."""
    points = np.ascontiguousarray(points, np.float64)
    origin = points.min(axis=0)
    scale = float((points.max(axis=0) - origin).max()) / _LMAX or 1.0
    coords0 = np.clip(np.rint((points - origin) / scale), 0, _LMAX).astype(np.uint32)
    return origin + scale * coords0.astype(np.float64), origin, scale


def build_graph_npz(points, n0, k, cov_bins, cov_vals, npz_path, *, cuda=False, aniso=None):
    """Build the Vecchia graph with Python ``graphgp`` and dump the bridge NPZ (with ``indices``).

    ``aniso`` (optional) carries an anisotropic kernel ``K(Δspatial, Δz)`` as a dict
    ``{spatial_bins, z_bins, grid (n_s,n_z), alpha}`` (the grid already nugget-inflated). The graph
    topology is identical — built on the same embedded ``(n̂, α·z)`` points — only the covariance the
    Julia driver assembles differs; ``cov_bins/cov_vals`` are placeholders then."""
    import jax.numpy as jnp
    import graphgp as gp

    points_q, origin, scale = _quantize(points)
    with _x64_disabled():
        try:
            graph = gp.build_graph(jnp.asarray(points_q, jnp.float32), n0=n0, k=k, cuda=cuda)
        except Exception:
            graph = gp.build_graph(jnp.asarray(points_q, jnp.float32), n0=n0, k=k)
    coords = np.rint((np.asarray(graph.points, np.float64) - origin) / scale).astype(np.uint32)
    idx = None if graph.indices is None else np.asarray(graph.indices, np.int64)
    extra = {}
    if aniso is not None:
        extra = dict(
            aniso_spatial_bins=np.asarray(aniso["spatial_bins"], np.float32),
            aniso_z_bins=np.asarray(aniso["z_bins"], np.float32),
            aniso_grid=np.asarray(aniso["grid"], np.float32),          # (n_s, n_z), jitter already in
            aniso_alpha=np.float64(aniso["alpha"]),
        )
    np.savez(npz_path,
             coords=coords, neighbors=np.asarray(graph.neighbors, np.int32),
             offsets=np.asarray(graph.offsets, np.int64), n0=np.int64(n0), scale=np.float64(scale),
             cov_bins32=np.asarray(cov_bins, np.float32), cov_vals32=np.asarray(cov_vals, np.float32),
             **({"indices": idx} if idx is not None else {}), **extra)
    return graph, scale


def dump_build_npz(points, n0, k, cov_bins, cov_vals, npz_path, *, aniso=None):
    """Dump RAW points for the build-in-Julia path: GraphGP.jl ``build_graph_ka`` does the k-d tree
    build + neighbor query + depth order + quantize on the backend (GPU), so the WHOLE field pipeline
    (build + generate) runs in one Julia process — no Python ``gp.build_graph``."""
    extra = {}
    if aniso is not None:
        extra = dict(aniso_spatial_bins=np.asarray(aniso["spatial_bins"], np.float32),
                     aniso_z_bins=np.asarray(aniso["z_bins"], np.float32),
                     aniso_grid=np.asarray(aniso["grid"], np.float32),
                     aniso_alpha=np.float64(aniso["alpha"]))
    np.savez(npz_path, build_points=np.asarray(points, np.float32), n0=np.int64(n0), k=np.int64(k),
             cov_bins32=np.asarray(cov_bins, np.float32), cov_vals32=np.asarray(cov_vals, np.float32),
             **extra)


def run_graphgp(points, n0, k, cov_bins, cov_vals, *, ops=("generate",), xi=None, values=None,
                device="cpu", dtype="f32", julia_threads=8, work_dir=None, _graph_npz=None,
                aniso=None, build_in_julia=False):
    """Run GraphGP.jl ops on a graph of ``points`` (original order). ``ops`` ⊆
    {generate, generate_inv, logdet, grad}. ``xi`` (N,) or (N,n_samples) and ``values`` (N,) are in
    ORIGINAL order. Returns a dict with the requested keys (generate → (n_samples,N) original order)."""
    assert device in ("cpu", "gpu") and dtype in ("f32", "f64")
    work = work_dir or tempfile.mkdtemp(prefix="echoes_ggp_")
    in_npz = os.path.join(work, "in.npz"); out_npz = os.path.join(work, "out.npz")

    if build_in_julia:
        dump_build_npz(points, n0, k, cov_bins, cov_vals, in_npz, aniso=aniso)
    elif _graph_npz is None:
        build_graph_npz(points, n0, k, cov_bins, cov_vals, in_npz, aniso=aniso)
    else:
        # reuse a prebuilt graph NPZ (amortise the build across calls), append xi/values
        d = dict(np.load(_graph_npz))
        np.savez(in_npz, **d)
    # append xi / values to the graph NPZ, in the RUN dtype (f32 halves the on-disk + RAM footprint
    # of a large (n_cand, n_samples) white-noise batch; the Julia driver reads them as T anyway).
    _xdt = np.float32 if dtype == "f32" else np.float64
    base = dict(np.load(in_npz))
    if xi is not None:
        base["xi"] = np.asarray(xi, _xdt)
    if values is not None:
        base["values"] = np.asarray(values, _xdt)
    np.savez(in_npz, **base)

    env = dict(os.environ)
    cmd = [JULIA, "-t", str(julia_threads), "--project=" + BENCH_PROJ, DRIVER,
           in_npz, out_npz, device, ",".join(ops), dtype]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if res.returncode != 0 or not os.path.exists(out_npz):
        raise RuntimeError(f"run_graphgp.jl failed (rc={res.returncode}):\n{res.stderr[-2000:]}")
    out = {kk: np.asarray(v) for kk, v in np.load(out_npz).items()}
    if "logdet" in out:
        out["logdet"] = float(out["logdet"])
    return out
