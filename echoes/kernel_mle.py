"""Marginal-likelihood hyperparameter inference for the GraphGP kernel, via GraphGP.jl's analytic
gradients (the capability the JAX CUDA extension lacks). Replaces the least-squares ``field_kernel.
fit_kernel`` — which only matches the binned ξ(r) — with a proper maximum-likelihood fit of the
stretched-exponential kernel ``A·exp(-(r/r0)^α)`` to an observed field ``y``:

    θ̂ = argmin_θ  ½ logdet K(θ) + ½ yᵀ K(θ)⁻¹ y         (negative log marginal likelihood)

Two backends (``fit_kernel_mle(..., backend=)``), same NLML optimum:

- ``"subprocess"`` (default): the whole L-BFGS loop runs inside ONE Julia process
  (``run_kernel_mle.jl``) — one cold-start per fit, then the many objective/gradient evals are
  in-Julia. Self-contained (standalone Julia, no preload).
- ``"juliacall"``: the optimizer stays in Python (scipy L-BFGS-B) and each NLML + ∂/∂θ eval runs
  IN-PROCESS via a warm juliacall worker (``graphgp_juliacall.KernelMLEProblem``); the kernel
  derivative ∂vals/∂θ is the analytic ``_strexp_jac``, chained to GraphGP's ∂/∂vals adjoint. The
  warm Julia is reused across ALL fits in the session, so per-fit cold-start vanishes (validated:
  gradcheck 6e-10 vs FD; θ̂ matches the subprocess to 3e-8; warm fit ~5 s vs ~15 s subprocess).
  Needs the juliacall launch env (Julia 1.11 + libstdc++ LD_PRELOAD — see ``graphgp_juliacall``).

Gradients are exact either way (``generate_logdet_grad_vals`` + ``generate_inv_loss_grad_vals`` +,
for the subprocess, ``hyperparam_grad``), verified against finite differences by ``gradcheck``.
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np

from . import graphgp_julia as ggj

_DRIVER = os.path.join(os.path.dirname(ggj.DRIVER), "run_kernel_mle.jl")   # vendored in echoes/jl/


def make_cov_bins(r_min, r_max, n_bins):
    """Log-spaced covariance grid with 0.0 prepended — matches GraphGP ``make_cov_bins`` /
    ``graphgp.extras.make_cov_bins`` (length ``n_bins + 1``)."""
    grid = np.logspace(np.log10(r_min), np.log10(r_max), int(n_bins))
    return np.concatenate([[0.0], grid]).astype(np.float64)


JIT = 1e-3                          # vals[0] inflation (matches run_kernel_mle.jl make_kernel)


def strexp_vals(bins, theta, jitter=JIT):
    """Stretched-exp ``A·exp(-(r/r0)^α)`` on ``bins`` from ``θ=[logA, log r0, α]``; vals[0] inflated."""
    logA, logr0, alpha = theta
    A, r0 = np.exp(logA), np.exp(logr0)
    vals = A * np.exp(-((bins / r0) ** alpha))
    vals = vals.copy()
    vals[0] *= 1.0 + jitter
    return vals.astype(np.float64)


def _strexp_jac(bins, theta, jitter=JIT):
    """Analytic ∂vals/∂θ (nbins, 3) for the stretched-exp kernel — the chain rule that turns
    ∂NLML/∂vals (GraphGP's analytic adjoint) into ∂NLML/∂θ. bins[0]=0 ⇒ vals[0]=A·(1+jit), so its
    r0/α derivatives vanish (the formulas below give 0 there automatically)."""
    bins = np.asarray(bins, np.float64)
    vals = strexp_vals(bins, theta, jitter)
    alpha = theta[2]; r0 = np.exp(theta[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = bins / r0
        u = np.where(bins > 0, ratio ** alpha, 0.0)               # (r/r0)^α
        lnr = np.where(bins > 0, np.log(ratio), 0.0)
    J = np.empty((len(bins), 3), np.float64)
    J[:, 0] = vals                                                # ∂/∂logA
    J[:, 1] = vals * alpha * u                                    # ∂/∂log r0
    J[:, 2] = -vals * u * lnr                                     # ∂/∂α
    return J


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


def _grid(points, r_min, r_max, n_bins):
    if r_min is None or r_max is None:
        span = float((points.max(0) - points.min(0)).max())
        r_min = r_min or max(span * 1e-3, 1e-3)
        r_max = r_max or 0.5 * span
    return make_cov_bins(r_min, r_max, n_bins)


def fit_kernel_mle(points, y, theta0, *, n0=256, k=30, r_min=None, r_max=None, n_bins=200,
                   backend="subprocess", julia_threads=None, work_dir=None):
    """Maximum-likelihood fit of the stretched-exp kernel to field ``y`` at ``points``.

    Parameters
    ----------
    points : (N, D)        point set.
    y      : (N,)          observed field values (original order).
    theta0 : [logA, logr0, alpha]  initial hyperparameters.
    n0, k  : Vecchia dense-block size / neighbor count.
    r_min, r_max, n_bins : kernel grid (defaults span the point separations).
    backend : ``"subprocess"`` (default) runs the whole L-BFGS in ONE Julia process
        (``run_kernel_mle.jl``) — one cold-start per fit. ``"juliacall"`` keeps the optimizer in
        Python (scipy L-BFGS-B) and evaluates the NLML + analytic gradient IN-PROCESS via juliacall
        (``graphgp_juliacall.KernelMLEProblem``) — the warm Julia is reused across ALL fits in the
        session, so the per-fit cold-start vanishes (see graphgp_juliacall for the LD_PRELOAD /
        Julia-1.11 launch requirements). Same NLML optimum either way.

    Returns
    -------
    dict with ``theta_hat`` (=[logA, logr0, alpha]), ``A``/``r0``/``alpha`` (natural units),
    ``cov`` (the fitted ``(bins, vals)`` tuple), ``nlml``, ``nlml0``, ``gnorm``, ``niter``.
    """
    points = np.asarray(points, np.float64)
    if julia_threads is None:
        julia_threads = ggj._default_julia_threads()
    bins = _grid(points, r_min, r_max, n_bins)
    if backend == "juliacall":
        return _fit_kernel_mle_juliacall(points, y, theta0, n0=n0, k=k, bins=bins)

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


def _fit_kernel_mle_juliacall(points, y, theta0, *, n0, k, bins):
    """Python-driven (scipy L-BFGS-B) MLE with warm in-process GraphGP evals (juliacall)."""
    from scipy.optimize import minimize
    from .graphgp_juliacall import KernelMLEProblem
    prob = KernelMLEProblem(points, y, n0=n0, k=k, bins=bins)
    theta0 = np.asarray(theta0, np.float64)

    def obj_grad(theta):
        # non-PD kernels (e.g. r0 → huge ⇒ singular dense block) throw inside Julia; return +inf so
        # the line search backtracks out of the infeasible region (mirrors run_kernel_mle.jl fg_safe).
        try:
            f, dvals = prob.fg(strexp_vals(bins, theta))
        except Exception:
            return np.inf, np.zeros(3)
        if not np.isfinite(f):
            return np.inf, np.zeros(3)
        g = _strexp_jac(bins, theta).T @ dvals                   # chain ∂/∂vals → ∂/∂θ
        return float(f), np.asarray(g, np.float64)

    f0 = obj_grad(theta0)[0]
    res = minimize(obj_grad, theta0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 200, "gtol": 1e-6})
    th = np.asarray(res.x, np.float64)
    return {
        "theta_hat": th, "A": float(np.exp(th[0])), "r0": float(np.exp(th[1])),
        "alpha": float(th[2]), "cov": (bins, strexp_vals(bins, th)),
        "nlml": float(res.fun), "nlml0": float(f0),
        "gnorm": float(np.linalg.norm(res.jac)), "niter": int(res.nit),
    }


def gradcheck_kernel_mle(points, y, theta0, *, n0=256, k=30, r_min=None, r_max=None, n_bins=200,
                         backend="subprocess", julia_threads=None, work_dir=None):
    """Analytic vs central-difference NLML gradient at ``theta0`` (the gradient-correctness gate).
    Returns dict with ``g_analytic``, ``g_fd``, ``rel`` (max abs rel difference), ``f``. The
    ``backend="juliacall"`` path additionally exercises the Python ∂vals/∂θ chain (``_strexp_jac``)."""
    points = np.asarray(points, np.float64)
    if julia_threads is None:
        julia_threads = ggj._default_julia_threads()
    bins = _grid(points, r_min, r_max, n_bins)
    theta0 = np.asarray(theta0, np.float64)
    if backend == "juliacall":
        from .graphgp_juliacall import KernelMLEProblem
        prob = KernelMLEProblem(points, y, n0=n0, k=k, bins=bins)
        f0, dvals = prob.fg(strexp_vals(bins, theta0))
        g = _strexp_jac(bins, theta0).T @ dvals
        h = 1e-5; gfd = np.empty(3)
        for i in range(3):
            tp = theta0.copy(); tp[i] += h; tm = theta0.copy(); tm[i] -= h
            gfd[i] = (prob.fg(strexp_vals(bins, tp))[0] - prob.fg(strexp_vals(bins, tm))[0]) / (2 * h)
        rel = float(np.max(np.abs(g - gfd)) / (np.max(np.abs(gfd)) + 1e-30))
        return {"g_analytic": g, "g_fd": gfd, "rel": rel, "f": float(f0)}

    work = work_dir or tempfile.mkdtemp(prefix="echoes_mlegc_")
    in_npz = _prepare_npz(points, y, theta0, n0=n0, k=k, bins=bins, work=work)
    out = _run(in_npz, os.path.join(work, "gc.npz"), "gradcheck", julia_threads)
    return {"g_analytic": np.asarray(out["g_analytic"]).ravel(),
            "g_fd": np.asarray(out["g_fd"]).ravel(),
            "rel": float(out["rel"]), "f": float(out["f"])}
