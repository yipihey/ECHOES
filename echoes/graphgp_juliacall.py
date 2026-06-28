"""In-process GraphGP backend via juliacall — the persistent-worker upgrade to the
NPZ subprocess bridge (``graphgp_julia``).

The subprocess bridge pays a Julia cold-start (~30–90 s) and writes the graph + a
(possibly multi-GB) white-noise batch to disk on every call. juliacall keeps a warm
Julia in the SAME Python process: GraphGP's byte-identical ``build_graph`` + the
analytic ``generate``/``generate_inv``/``generate_logdet``/``generate_logdet_grad_vals``
run in-process, arrays passed in memory — no cold-start, no disk round-trip. Ideal for
the many-small-call paths (kernel-MLE objective/grad, per-sightline posteriors).

**CPU-only by design.** Julia-CUDA and JAX-cuda12 cannot co-reside on the one A6000;
the subprocess bridge exists precisely so Julia can own the GPU in its own process.
This in-process backend therefore runs the GraphGP ops on the CPU (where in-process
wins most — eliminating cold-start/disk for the small-call loops); GPU field generation
stays on the subprocess bridge (``device="gpu"``). JAX must be on CPU in this process
(the pipelines already use ``JAX_PLATFORMS=cpu``).

Validated (N=20k, k=30, CPU): in-process field matches the subprocess bridge to **2.1e-6**
(f32, same graph via the shared ``_quantize``); **warm call 0.8 s vs 13 s subprocess cold-start
(~16×)**; first call ~35 s (one-time PythonCall provision + Julia start, amortized).

**Launch recipe + gotchas (saved to memory).** Three environment requirements, none of which
the module can self-apply (they must hold at Python launch):

  LD_PRELOAD=<julia>/lib/julia/libstdc++.so.6 \
  PYTHONPATH=/home/tabel/Projects/ECHOES \
  python your_script.py

1. juliacall/PythonCall requires Julia **1.10–1.11**, NOT the juliaup default 1.12.6 (its
   OpenSSL_jll pin is unsatisfiable on 1.12). We point ``PYTHON_JULIAPKG_EXE`` at the real
   1.11.9 binary (the juliaup ``~/.juliaup/bin/julia`` is a *shim* with no ``sys.so``).
2. **LD_PRELOAD Julia's bundled libstdc++** (``LIBSTDCXX`` below) — the CentOS-8 system
   ``/lib64/libstdc++.so.6`` lacks ``GLIBCXX_3.4.26`` that Julia 1.11 needs; the subprocess
   bridge avoids this (standalone Julia), in-process does not.
3. Use a lightweight GraphGP-only project (``ECHOES_JULIACALL_ENV``, no CUDA → fast precompile).

Override paths via ``$ECHOES_JULIACALL_EXE`` / ``$ECHOES_JULIACALL_ENV``.
"""
from __future__ import annotations

import os

import numpy as np

_JULIA_1_11 = "/home/tabel/.julia/juliaup/julia-1.11.9+0.x64.linux.gnu/bin/julia"
EXE = os.environ.get("ECHOES_JULIACALL_EXE", _JULIA_1_11)
# Julia's bundled libstdc++ (newer than CentOS-8 system /lib64) — LD_PRELOAD this before Python
# starts so juliacall can dlopen libjulia into the process (see _main / module docstring).
LIBSTDCXX = os.path.join(os.path.dirname(os.path.dirname(EXE)), "lib", "julia", "libstdc++.so.6")
ENVDIR = os.environ.get("ECHOES_JULIACALL_ENV",
                        os.path.expanduser("~/.venv/k3d/echoes_ggp_cpu"))

_jl = None          # cached juliacall Main with GraphGP loaded (warm worker)


def available() -> bool:
    """True iff the juliacall package is installed and the 1.11 Julia + GraphGP env exist.

    Uses ``find_spec`` (does NOT import juliacall) so it is safe to call without the
    libstdc++ ``LD_PRELOAD`` — importing juliacall would trigger Julia's C-level init,
    which aborts uncatchably if the preload is missing (see module docstring)."""
    import importlib.util
    if importlib.util.find_spec("juliacall") is None:
        return False
    return os.path.isfile(EXE) and os.path.isfile(os.path.join(ENVDIR, "Project.toml"))


def _main():
    """Start (once) and return the warm juliacall Main with GraphGP loaded."""
    global _jl
    if _jl is not None:
        return _jl
    os.environ.setdefault("PYTHON_JULIAPKG_EXE", EXE)
    os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")     # keep JAX off the GPU (no CUDA clash)
    # The env is already resolved/precompiled — skip registry fetches on start (otherwise juliapkg
    # tries to update a private registry and blocks on an SSH credential prompt in non-interactive runs).
    os.environ.setdefault("JULIA_PKG_OFFLINE", "true")
    try:
        from juliacall import Main as jl
    except Exception as e:                            # the classic old-libstdc++ failure
        if "GLIBCXX" in str(e) or "libstdc++" in str(e):
            raise RuntimeError(
                "juliacall could not load Julia: the system libstdc++ is older than Julia's. "
                "Launch Python with Julia's bundled libstdc++ preloaded, e.g.\n"
                f"  LD_PRELOAD={LIBSTDCXX} python ...\n"
                "(the subprocess `backend='julia'` needs no preload — it runs Julia standalone.)"
            ) from e
        raise
    # Activate the pre-built GraphGP env (created once by the setup; see module docstring). Do NOT
    # Pkg.instantiate here — it takes the Pkg lock on every start and a stale lock (from a killed
    # process) would DEADLOCK the warm worker. `using GraphGP` errors clearly if the env is missing.
    jl.seval(f'import Pkg; Pkg.activate(raw"{ENVDIR}"; io=devnull)')
    jl.seval("using GraphGP")
    _jl = jl
    return jl


def _np_dtype(dtype):
    return np.float32 if dtype == "f32" else np.float64


def run_graphgp_inproc(points, n0, k, cov_bins, cov_vals, *, ops=("generate",),
                       xi=None, values=None, dtype="f32"):
    """In-process analog of ``graphgp_julia.run_graphgp`` (CPU). ``points`` (N,D) and
    ``xi`` (N,) or (N,S) / ``values`` (N,) are ORIGINAL order; results come back in
    ORIGINAL order (GraphGP's ``generate`` gathers/scatters via ``prob.indices``)."""
    jl = _main()
    T = _np_dtype(dtype)
    # build_graph requires Float64 POINTS (topology only — quantized to the integer lattice
    # internally); the KERNEL (bins,vals) dtype sets the numeric element type of the problem (T).
    # Pre-quantize with the SAME 21-bit snap the subprocess bridge uses (graphgp_julia._quantize) so
    # `jl.build_graph` (byte-identical to gp.build_graph) yields the identical graph → the in-process
    # field matches the subprocess field for the same xi (not merely statistically equivalent).
    from .graphgp_julia import _quantize
    pts_q, _, _ = _quantize(np.asarray(points, np.float64))
    pts = np.asfortranarray(pts_q)
    bins = np.ascontiguousarray(np.asarray(cov_bins, T))
    vals = np.ascontiguousarray(np.asarray(cov_vals, T))
    prob = jl.build_graph(jl.Matrix(pts), int(n0), int(k), jl.Vector(bins), jl.Vector(vals))
    out = {}
    if "generate" in ops:
        x = np.asarray(xi, T)
        x2 = x[:, None] if x.ndim == 1 else x
        cols = [np.asarray(jl.generate(prob, jl.Vector(np.ascontiguousarray(x2[:, s]))))
                for s in range(x2.shape[1])]
        out["generate"] = np.stack(cols, axis=0)          # (S, N) original order
    if "generate_inv" in ops:
        out["generate_inv"] = np.asarray(jl.generate_inv(prob, jl.Vector(np.asarray(values, T))))
    if "logdet" in ops:
        out["logdet"] = float(jl.generate_logdet(prob))
    if "grad" in ops:
        out["grad_logdet_vals"] = np.asarray(jl.generate_logdet_grad_vals(prob), np.float64)
    return out


# kernel-MLE NLML + ∂/∂vals, defined once in Julia: rebuild the problem with new `vals` on the FIXED
# graph (cheap — no tree rebuild), return NamedTuple(f = ½logdet + ½yᵀK⁻¹y, dvals = ∂f/∂vals).
_MLE_FG_SRC = """
function _echoes_mle_fg(prob, vals, y)
    p = GraphGPProblem(prob.coords, prob.neighbors, prob.offsets, prob.n0, prob.scale,
                       prob.bins, vals, prob.indices)
    ld = generate_logdet(p)
    loss, dvals_loss = generate_inv_loss_grad_vals(p, y)      # (½ yᵀK⁻¹y, ∂/∂vals)
    dvals = 0.5 .* generate_logdet_grad_vals(p) .+ dvals_loss
    return (f = 0.5 * ld + loss, dvals = dvals)
end
"""


class KernelMLEProblem:
    """Warm in-process NLML objective/gradient for the GraphGP kernel-MLE.

    Builds the Vecchia graph ONCE (in-process, byte-identical to the subprocess via the
    shared ``_quantize``), then ``fg(vals)`` evaluates the negative log marginal
    likelihood and its ∂/∂vals on the fixed graph — the warm-call loop that the
    Python-driven optimizer (``kernel_mle.fit_kernel_mle(backend="juliacall")``) iterates.
    F64 (gradient accuracy). Many fits in one Python session share this warm Julia."""

    def __init__(self, points, y, *, n0, k, bins):
        jl = _main()
        from .graphgp_julia import _quantize
        pts_q, _, _ = _quantize(np.asarray(points, np.float64))
        bins = np.ascontiguousarray(np.asarray(bins, np.float64))
        vals0 = np.ones_like(bins)                            # placeholder; fg() supplies real vals
        self._jl = jl
        self._prob = jl.build_graph(jl.Matrix(np.asfortranarray(pts_q)), int(n0), int(k),
                                    jl.Vector(bins), jl.Vector(vals0))
        self._y = jl.Vector(np.ascontiguousarray(np.asarray(y, np.float64)))
        jl.seval(_MLE_FG_SRC)

    def fg(self, vals):
        """Return ``(f, dvals)`` for kernel values ``vals`` (length nbins) — NLML + ∂/∂vals."""
        res = self._jl._echoes_mle_fg(self._prob,
                                      self._jl.Vector(np.ascontiguousarray(np.asarray(vals, np.float64))),
                                      self._y)
        return float(res.f), np.asarray(res.dvals, np.float64)
