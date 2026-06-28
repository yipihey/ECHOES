"""Backend-switchable GraphGP field generation — the one seam through which heavy Vecchia field
realizations in ECHOES flow, so a single ``backend=`` kwarg routes between (the pure-JAX
``graphgp`` engine has been retired):

  ``"julia"`` (default) the GraphGP.jl rewrite over the NPZ subprocess bridge (``graphgp_julia``) —
              never materializes the ``(M,k+1,k+1)`` conditional tensor, GPU parity-to-ahead of the
              CUDA ref, and the only path that also returns analytic gradients. Owns the GPU in its
              own process.
  ``"juliacall"`` the SAME GraphGP.jl but called IN-PROCESS via juliacall (``graphgp_juliacall``) —
              warm Julia, no subprocess cold-start, no NPZ disk round-trip. **CPU-only**, and
              isotropic-only for now. Best for the many-small-call paths (kernel-MLE objective/grad).

The subprocess Julia path owns the GPU in its own process. All paths consume the SAME tabulated
``(cov_bins, cov_vals)`` kernel and the same ``(points, n0, k)`` Vecchia spec, and return the field
in ORIGINAL point order.

Use ``"julia"`` for BATCH GPU generation (one call, many samples). Use ``"juliacall"`` for warm,
CPU, many-small-call loops where the subprocess cold-start would dominate.
"""
from __future__ import annotations

import os

import numpy as np

DEFAULT_BACKEND = os.environ.get("ECHOES_GRAPHGP_BACKEND", "julia")


def _as_2d(xi):
    """(N,) -> (N,1); (N,S) unchanged. Returns (xi2d, was_1d)."""
    xi = np.asarray(xi)
    if xi.ndim == 1:
        return xi[:, None], True
    return xi, False


def _is_aniso(cov):
    """True for an ``AnisotropicCovariance`` (the fork's K(Δspatial,Δz)); False for a (bins,vals)
    tuple. Duck-typed so we don't hard-depend on the fork being importable."""
    return all(hasattr(cov, a) for a in ("grid", "spatial_bins", "z_bins", "alpha"))


def _aniso_dict(cov):
    return dict(spatial_bins=np.asarray(cov.spatial_bins, np.float64),
                z_bins=np.asarray(cov.z_bins, np.float64),
                grid=np.asarray(cov.grid, np.float64),         # (n_s,n_z), nugget already applied
                alpha=float(cov.alpha))


def generate_field(points, cov, xi, *, n0, k, backend=None, device="cpu",
                   graph=None, graph_npz=None, dtype="f32", build_in_julia=False):
    """Draw GP field(s) ``L·xi`` at ``points`` with the chosen backend.

    Parameters
    ----------
    points : (N, D) array          Vecchia point set (original order).
    cov    : (cov_bins, cov_vals)  tabulated covariance (numpy arrays).
    xi     : (N,) or (N, S)        white-noise input(s), original order.
    n0, k  : Vecchia dense-block size and neighbor count.
    backend: "julia" | "juliacall" (default ``DEFAULT_BACKEND``).
    device : "cpu" | "gpu"         Julia backend only.
    graph  : unused (retained for API compatibility; JAX engine retired).
    graph_npz : prebuilt bridge NPZ path to reuse (julia backend; amortizes the build).

    Returns
    -------
    (S, N) array of fields in ORIGINAL point order, or (N,) if ``xi`` was 1-D.
    """
    backend = backend or DEFAULT_BACKEND
    aniso = _aniso_dict(cov) if _is_aniso(cov) else None
    if aniso is None:
        cov_bins = np.asarray(cov[0], np.float64)
        cov_vals = np.asarray(cov[1], np.float64)
    else:                                                     # placeholders; driver uses the aniso grid
        cov_bins = aniso["spatial_bins"]
        cov_vals = np.zeros_like(cov_bins)
    xi2d, was_1d = _as_2d(xi)
    N, S = xi2d.shape

    if backend == "julia":
        from . import graphgp_julia as ggj
        res = ggj.run_graphgp(np.asarray(points), n0, k, cov_bins, cov_vals,
                              ops=("generate",), xi=xi2d, device=device, dtype=dtype,
                              _graph_npz=graph_npz, aniso=aniso, build_in_julia=build_in_julia)
        out = np.asarray(res["generate"], np.float64)         # (S, N) original order
        return out[0] if was_1d else out

    if backend == "juliacall":
        if aniso is not None:
            raise NotImplementedError(
                "juliacall backend is isotropic-only; use backend='julia' for anisotropic covariance")
        from . import graphgp_juliacall as ggc
        res = ggc.run_graphgp_inproc(np.asarray(points), n0, k, cov_bins, cov_vals,
                                     ops=("generate",), xi=xi2d, dtype=dtype)
        out = np.asarray(res["generate"], np.float64)         # (S, N) original order
        return out[0] if was_1d else out

    raise ValueError(f"unknown backend {backend!r} (expected 'julia' or 'juliacall')")


def build_graph_npz(points, n0, k, cov, npz_path, *, cuda_build=False):
    """Build the bridge NPZ once (the julia backend reuses it via ``graph_npz=``). ``cov`` is a
    ``(bins, vals)`` tuple or an ``AnisotropicCovariance``."""
    from . import graphgp_julia as ggj
    if _is_aniso(cov):
        aniso = _aniso_dict(cov)
        return ggj.build_graph_npz(np.asarray(points), n0, k, aniso["spatial_bins"],
                                   np.zeros_like(aniso["spatial_bins"]), npz_path,
                                   cuda=cuda_build, aniso=aniso)
    cov_bins = np.asarray(cov[0], np.float64)
    cov_vals = np.asarray(cov[1], np.float64)
    return ggj.build_graph_npz(np.asarray(points), n0, k, cov_bins, cov_vals, npz_path,
                               cuda=cuda_build)
