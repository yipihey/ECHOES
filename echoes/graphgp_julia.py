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
import shutil
import subprocess
import tempfile

import numpy as np


def _scratch_base():
    """Base dir for the bridge's temp NPZ dumps. A large ensemble dumps a (n_cand, n_samples)
    white-noise array (multi-GB) that overflows the small /tmp, so prefer a fast/large scratch:
    ``$ECHOES_SCRATCH`` if set, else the nvme zfs pool if present, else None (tempfile default)."""
    d = os.environ.get("ECHOES_SCRATCH")
    if d:
        os.makedirs(d, exist_ok=True)
        return d
    for c in ("/zpool/nvme/data/tabel/echoes_scratch",):
        if os.path.isdir(c):
            return c
    return None


def _default_julia_threads():
    """Resolve the Julia worker thread count when the caller passes ``julia_threads=None``:
    ``$ECHOES_JULIA_THREADS`` (set per-worker by :func:`echoes.parallel.numa_map` to the NUMA
    node's core count) if present, else 8. This lets a NUMA-pinned ensemble worker spin its Julia
    subprocess at exactly its node's width — and keep it on-node — without threading the count
    through every call site. A single saturated process should stay ≲ one NUMA domain (~8); the
    win on this 16×8 box is many on-node processes, not one 128-thread pool."""
    try:
        return max(1, int(os.environ.get("ECHOES_JULIA_THREADS", "8")))
    except ValueError:
        return 8


@contextlib.contextmanager
def _x64_disabled():
    """Force ``jax_enable_x64`` off for the duration. The k-d tree build (``graphgp.tree``) has an
    ``lax.cond`` whose branches disagree on f32/f64 when x64 is enabled — and graph *topology* is
    precision-insensitive (integer lattice coords) — so we always build under f32. Restored on exit
    so a caller running the f64 numerics (the parity gate) is unaffected."""
    # vestigial no-op: graph build is now JAX-free (GraphGP.jl), so there is no x64 flag to toggle.
    yield

# locations (overridable via env for other machines)
JULIA = os.environ.get("ECHOES_JULIA", os.path.expanduser("~/.juliaup/bin/julia"))


def _find_graphgp_jl():
    """Locate the GraphGP.jl repo (only its `bench` Project.toml is needed — it path-deps the
    package). Auto-detected so the upstream repo can be renamed/restructured without breaking us:
    ``$ECHOES_GRAPHGP_JL`` if set, else the standalone ``~/Projects/GraphGP.jl``, else the legacy
    nested ``~/Projects/graphgp-julia/julia/GraphGP``."""
    cands = [os.environ.get("ECHOES_GRAPHGP_JL")]
    cands += [os.path.expanduser(p) for p in
              ("~/Projects/GraphGP.jl", "~/Projects/graphgp-julia/julia/GraphGP")]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "bench", "Project.toml")):
            return c
    return os.path.expanduser("~/Projects/GraphGP.jl")


GRAPHGP_JL = _find_graphgp_jl()
BENCH_PROJ = os.path.join(GRAPHGP_JL, "bench")          # the Julia project (path-deps the package)
# The driver SCRIPTS are vendored INSIDE ECHOES (echoes/jl/) so the upstream repo's renames/resets
# can't delete them out from under a running pipeline; only the GraphGP *package* comes from the repo.
DRIVER = os.path.join(os.path.dirname(__file__), "jl", "run_graphgp.jl")
BUILD_DRIVER = os.path.join(os.path.dirname(__file__), "jl", "run_build_graph.jl")
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
    """Build the Vecchia graph with **GraphGP.jl** (byte-identical to ``gp.build_graph`` — verified
    coords/neighbours/offsets/indices to 0/N) and dump the bridge NPZ (with ``indices``). JAX-free:
    the graph is constructed by ``run_build_graph.jl``; ``points`` are pre-quantised to the 21-bit
    lattice (as f32) so the Julia ``build_graph`` reproduces the Python build exactly.

    ``aniso`` (optional) carries an anisotropic kernel ``K(Δspatial, Δz)`` as a dict
    ``{spatial_bins, z_bins, grid (n_s,n_z), alpha}`` (the grid already nugget-inflated). The graph
    topology is identical — built on the same embedded ``(n̂, α·z)`` points — only the covariance the
    Julia driver assembles differs; ``cov_bins/cov_vals`` are placeholders then. (``cuda`` is accepted
    for back-compat and ignored.)"""
    points_q, _origin, scale = _quantize(points)
    extra = {}
    if aniso is not None:
        extra = dict(
            aniso_spatial_bins=np.asarray(aniso["spatial_bins"], np.float32),
            aniso_z_bins=np.asarray(aniso["z_bins"], np.float32),
            aniso_grid=np.asarray(aniso["grid"], np.float32),          # (n_s, n_z), jitter already in
            aniso_alpha=np.float64(aniso["alpha"]),
        )
    bin_npz = str(npz_path) + ".buildin.npz"
    np.savez(bin_npz, build_points=points_q.astype(np.float32), n0=np.int64(n0), k=np.int64(k),
             cov_bins32=np.asarray(cov_bins, np.float32), cov_vals32=np.asarray(cov_vals, np.float32),
             **extra)
    try:
        res = subprocess.run([JULIA, "--project=" + BENCH_PROJ, BUILD_DRIVER, bin_npz, str(npz_path)],
                             capture_output=True, text=True)
    finally:
        if os.path.exists(bin_npz):
            os.remove(bin_npz)
    if res.returncode != 0 or not os.path.exists(npz_path):
        raise RuntimeError(f"run_build_graph.jl failed (rc={res.returncode}):\n{res.stderr[-2000:]}")
    g = _GraphInfo(npz_path)
    return g, g.scale          # the graph's OWN scale (consistent with the written coords)


class _GraphInfo:
    """Lightweight view of the just-built graph NPZ (preserves the ``.indices``/``.neighbors``/
    ``.offsets``/``.scale`` attributes the previous ``gp.build_graph`` return exposed)."""
    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.coords = d["coords"]; self.neighbors = d["neighbors"]; self.offsets = d["offsets"]
        self.scale = float(d["scale"])
        self.indices = d["indices"] if "indices" in d.files else None


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
                device="cpu", dtype="f32", julia_threads=None, work_dir=None, _graph_npz=None,
                aniso=None, build_in_julia=False):
    """Run GraphGP.jl ops on a graph of ``points`` (original order). ``ops`` ⊆
    {generate, generate_inv, logdet, grad}. ``xi`` (N,) or (N,n_samples) and ``values`` (N,) are in
    ORIGINAL order. Returns a dict with the requested keys (generate → (n_samples,N) original order).
    ``julia_threads=None`` resolves to ``$ECHOES_JULIA_THREADS`` (set by the NUMA ensemble runner)
    or 8."""
    assert device in ("cpu", "gpu") and dtype in ("f32", "f64")
    if julia_threads is None:
        julia_threads = _default_julia_threads()
    _auto = work_dir is None                              # auto-created dirs get cleaned up below
    work = work_dir or tempfile.mkdtemp(prefix="echoes_ggp_", dir=_scratch_base())
    in_npz = os.path.join(work, "in.npz"); out_npz = os.path.join(work, "out.npz")
    try:
        return _run_graphgp_inner(points, n0, k, cov_bins, cov_vals, ops, xi, values, device, dtype,
                                  julia_threads, _graph_npz, aniso, build_in_julia, in_npz, out_npz)
    finally:
        if _auto:
            shutil.rmtree(work, ignore_errors=True)       # free the multi-GB dump after each call


def _run_graphgp_inner(points, n0, k, cov_bins, cov_vals, ops, xi, values, device, dtype,
                       julia_threads, _graph_npz, aniso, build_in_julia, in_npz, out_npz):
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
    with np.load(out_npz) as _npz:                        # close before the caller's rmtree
        out = {kk: np.asarray(v) for kk, v in _npz.items()}
    if "logdet" in out:
        out["logdet"] = float(out["logdet"])
    return out
