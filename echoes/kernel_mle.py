"""Marginal-likelihood hyperparameter inference for the GraphGP kernel, via GraphGP.jl's analytic
gradients (the capability the JAX CUDA extension lacks). Replaces the least-squares ``field_kernel.
fit_kernel`` — which only matches the binned ξ(r) — with a proper maximum-likelihood fit of the
stretched-exponential kernel ``A·exp(-(r/r0)^α)`` to an observed field ``y``:

    θ̂ = argmin_θ  ½ logdet K(θ) + ½ yᵀ K(θ)⁻¹ y         (negative log marginal likelihood)

The whole L-BFGS loop runs inside ONE Julia process (``run_kernel_mle.jl``), so the many small
objective/gradient evaluations never pay the subprocess cold-start — the lightweight stand-in for the
persistent-worker bridge (plan P3). Gradients are exact (``generate_logdet_grad_vals`` +
``generate_inv_loss_grad_vals`` + ``hyperparam_grad``), verified against finite differences by the
``gradcheck`` mode.
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np

from . import graphgp_julia as ggj

_DRIVER = os.path.join(ggj.GRAPHGP_JL, "bench", "compare", "run_kernel_mle.jl")


def make_cov_bins(r_min, r_max, n_bins):
    """Log-spaced covariance grid with 0.0 prepended — matches GraphGP ``make_cov_bins`` /
    ``graphgp.extras.make_cov_bins`` (length ``n_bins + 1``)."""
    grid = np.logspace(np.log10(r_min), np.log10(r_max), int(n_bins))
    return np.concatenate([[0.0], grid]).astype(np.float64)


def strexp_vals(bins, theta, jitter=1e-3):
    """Stretched-exp ``A·exp(-(r/r0)^α)`` on ``bins`` from ``θ=[logA, log r0, α]``; vals[0] inflated."""
    logA, logr0, alpha = theta
    A, r0 = np.exp(logA), np.exp(logr0)
    vals = A * np.exp(-((bins / r0) ** alpha))
    vals = vals.copy()
    vals[0] *= 1.0 + jitter
    return vals.astype(np.float64)


def _prepare_npz(points, y, theta0, *, n0, k, bins, work):
    """Build the Vecchia graph NPZ (incl. indices) on the fixed ``bins`` grid and append y/theta0."""
    in_npz = os.path.join(work, "mle.npz")
    vals0 = strexp_vals(bins, theta0)
    ggj.build_graph_npz(np.asarray(points), n0, k, bins, vals0, in_npz)
    base = dict(np.load(in_npz))
    base["y"] = np.asarray(y, np.float64)
    base["theta0"] = np.asarray(theta0, np.float64)
    np.savez(in_npz, **base)
    return in_npz


def _run(in_npz, out_npz, mode, julia_threads):
    cmd = [ggj.JULIA, "-t", str(julia_threads), "--project=" + ggj.BENCH_PROJ, _DRIVER,
           in_npz, out_npz, mode]
    res = subprocess.run(cmd, env=dict(os.environ), capture_output=True, text=True)
    if res.returncode != 0 or not os.path.exists(out_npz):
        raise RuntimeError(f"run_kernel_mle.jl ({mode}) failed (rc={res.returncode}):\n{res.stderr[-2000:]}")
    return {kk: np.asarray(v) for kk, v in np.load(out_npz).items()}


def fit_kernel_mle(points, y, theta0, *, n0=256, k=30, r_min=None, r_max=None, n_bins=200,
                   julia_threads=8, work_dir=None):
    """Maximum-likelihood fit of the stretched-exp kernel to field ``y`` at ``points``.

    Parameters
    ----------
    points : (N, D)        point set.
    y      : (N,)          observed field values (original order).
    theta0 : [logA, logr0, alpha]  initial hyperparameters.
    n0, k  : Vecchia dense-block size / neighbor count.
    r_min, r_max, n_bins : kernel grid (defaults span the point separations).

    Returns
    -------
    dict with ``theta_hat`` (=[logA, logr0, alpha]), ``A``/``r0``/``alpha`` (natural units),
    ``cov`` (the fitted ``(bins, vals)`` tuple), ``nlml``, ``nlml0``, ``gnorm``, ``niter``.
    """
    points = np.asarray(points, np.float64)
    if r_min is None or r_max is None:
        span = float((points.max(0) - points.min(0)).max())
        r_min = r_min or max(span * 1e-3, 1e-3)
        r_max = r_max or 0.5 * span
    bins = make_cov_bins(r_min, r_max, n_bins)
    work = work_dir or tempfile.mkdtemp(prefix="echoes_mle_")
    in_npz = _prepare_npz(points, y, theta0, n0=n0, k=k, bins=bins, work=work)
    out = _run(in_npz, os.path.join(work, "out.npz"), "fit", julia_threads)
    th = np.asarray(out["theta_hat"], np.float64).ravel()
    return {
        "theta_hat": th, "A": float(np.exp(th[0])), "r0": float(np.exp(th[1])),
        "alpha": float(th[2]), "cov": (bins, strexp_vals(bins, th)),
        "nlml": float(out["nlml"]), "nlml0": float(out["nlml0"]),
        "gnorm": float(out["gnorm"]), "niter": int(out["niter"]),
    }


def gradcheck_kernel_mle(points, y, theta0, *, n0=256, k=30, r_min=None, r_max=None, n_bins=200,
                         julia_threads=8, work_dir=None):
    """Analytic vs central-difference NLML gradient at ``theta0`` (the gradient-correctness gate).
    Returns dict with ``g_analytic``, ``g_fd``, ``rel`` (max abs rel difference), ``f``."""
    points = np.asarray(points, np.float64)
    if r_min is None or r_max is None:
        span = float((points.max(0) - points.min(0)).max())
        r_min = r_min or max(span * 1e-3, 1e-3)
        r_max = r_max or 0.5 * span
    bins = make_cov_bins(r_min, r_max, n_bins)
    work = work_dir or tempfile.mkdtemp(prefix="echoes_mlegc_")
    in_npz = _prepare_npz(points, y, theta0, n0=n0, k=k, bins=bins, work=work)
    out = _run(in_npz, os.path.join(work, "gc.npz"), "gradcheck", julia_threads)
    return {"g_analytic": np.asarray(out["g_analytic"]).ravel(),
            "g_fd": np.asarray(out["g_fd"]).ravel(),
            "rel": float(out["rel"]), "f": float(out["f"])}
