"""Conditional Gaussian-process field posterior — the real Matheron solve.

This is the field-level reconstruction primitive the ECHOES "graphGP" engine was
supposed to provide but never did: a *proper* conditional GP posterior of the
latent field given noisy observations, rather than a local-density estimate.

Given observations ``y_D`` at data points ``x_D`` with per-point noise variance
``N_D``, and a stationary kernel ``K`` tabulated from the measured ξ(r) (the
graphGP ``(cov_bins, cov_vals)`` convention), the posterior of the field at
prediction points ``x_*`` is the textbook GP regression

    mean_* = K_{*D} (K_{DD} + N)^{-1} y_D
    cov_** = K_** - K_{*D} (K_{DD} + N)^{-1} K_{D*},

and posterior *samples* follow Matheron's pathwise-conditioning rule

    f^post_* = f^prior_* + K_{*D} (K_{DD}+N)^{-1} (y_D + ε_D - f^prior_D),
    (f^prior_*, f^prior_D) ~ GP(0, K),   ε_D ~ N(0, N),

which gives a draw from the exact posterior with one prior draw + one solve.

Two backends:
- ``dense`` (this module): exact via ``graphgp.compute_cov_matrix`` + Cholesky,
  for moderate ``N_D`` (≲ a few ×10³). Used to validate the math and to drive the
  completion on subsampled / tiled problems.
- ``vecchia`` (``field_posterior_vecchia``, to follow): the scalable graphGP
  version — prior draws via ``graphgp.generate`` on a Vecchia graph and
  ``(K_DD+N)^{-1}`` by matrix-free conjugate gradient — validated to match
  ``dense`` on a small problem, then scaled to the full BOSS catalog.

The kernel is the graphGP ``(cov_bins, cov_vals)`` tuple from
:func:`echoes.field_kernel.tabulate_kernel`; ``compute_cov_matrix`` evaluates the
exact cross-covariance for arbitrary point sets, so the same code conditions the
field at observed galaxy positions and predicts it along missing sightlines.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _as_cov(cov):
    """Coerce a ``(cov_bins, cov_vals)`` tuple to jax arrays (graphgp requires
    jax.Array elements); pass any non-tuple covariance object through untouched."""
    import jax.numpy as jnp
    if isinstance(cov, (tuple, list)) and len(cov) == 2:
        return (jnp.asarray(cov[0], dtype=jnp.float64), jnp.asarray(cov[1], dtype=jnp.float64))
    return cov


def _cov_matrix(cov, A, B):
    """Exact kernel cross-covariance K(A, B), returned as a writable numpy array.

    For the tabulated isotropic ``(cov_bins, cov_vals)`` kernel this is a pure
    numpy distance-lookup (``np.interp``, identical to graphgp's ``cov_lookup``)
    — much faster than dispatching jax per call inside the per-sightline loop.
    Non-tuple covariances (e.g. ``AnisotropicCovariance``) fall back to graphgp.
    """
    A = np.ascontiguousarray(A, np.float64)
    B = np.ascontiguousarray(B, np.float64)
    if isinstance(cov, (tuple, list)) and len(cov) == 2:
        bins = np.asarray(cov[0], np.float64)
        vals = np.asarray(cov[1], np.float64)
        # pairwise Euclidean distances via cdist (C, no Python temporary) — ~3x
        # faster than the broadcasting `A[:,None,:]-B[None,:,:]`, which dominated
        # the per-sightline GP solve (the inner kernel of the field/generative
        # engines' M-sightline loop). Mathematically identical.
        from scipy.spatial.distance import cdist
        d = cdist(A, B)
        return np.interp(d.ravel(), bins, vals).reshape(d.shape)
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from graphgp import compute_cov_matrix
    return np.array(compute_cov_matrix(_as_cov(cov), jnp.asarray(A), jnp.asarray(B)),
                    dtype=np.float64)


def _k0(cov) -> float:
    """Zero-lag kernel variance K(0) (stationary): cov_vals[0]."""
    return float(np.asarray(cov[1])[0])


def _chol_factor(A):
    from scipy.linalg import cho_factor
    return cho_factor(A, lower=True, check_finite=False)


def _chol_solve(c, b):
    from scipy.linalg import cho_solve
    return cho_solve(c, b, check_finite=False)


def gp_posterior_dense(
    x_data: np.ndarray,
    y_data: np.ndarray,
    noise_var,
    x_pred: np.ndarray,
    cov,
    *,
    jitter: float = 1e-6,
    return_var: bool = True,
):
    """Posterior mean (and marginal variance) of the GP field at ``x_pred``.

    ``noise_var`` is a scalar or ``(N_D,)`` per-point observation variance.
    Returns ``mean`` (and ``var`` if ``return_var``), shape ``(N_*,)``.
    """
    x_data = np.ascontiguousarray(x_data, np.float64)
    x_pred = np.ascontiguousarray(x_pred, np.float64)
    y_data = np.ascontiguousarray(y_data, np.float64)
    nd = len(x_data)
    nv = np.broadcast_to(np.asarray(noise_var, np.float64), (nd,))
    A = _cov_matrix(cov, x_data, x_data)
    A[np.diag_indices(nd)] += nv + jitter
    c = _chol_factor(A)
    alpha = _chol_solve(c, y_data)                       # (K_DD+N)^-1 y_D
    K_sD = _cov_matrix(cov, x_pred, x_data)              # (N_*, N_D)
    mean = K_sD @ alpha
    if not return_var:
        return mean
    # var_* = K(0) - sum_i (L^-1 K_Ds)_i^2  (marginal)
    from scipy.linalg import solve_triangular
    L = c[0]
    v = solve_triangular(L, K_sD.T, lower=True, check_finite=False)   # (N_D, N_*)
    var = _k0(cov) - np.einsum("ij,ij->j", v, v)
    return mean, np.maximum(var, 0.0)


def gp_sample_dense(
    x_data: np.ndarray,
    y_data: np.ndarray,
    noise_var,
    x_pred: np.ndarray,
    cov,
    *,
    n_samples: int = 1,
    seed: int = 0,
    jitter: float = 1e-6,
):
    """Exact posterior samples of the field at ``x_pred`` by Matheron's rule.

    Returns ``(n_samples, N_*)``. Each sample is a draw from the GP posterior
    (correct mean, variance, AND cross-point correlations) — the property the
    local-density engines lack.
    """
    rng = np.random.default_rng(seed)
    x_data = np.ascontiguousarray(x_data, np.float64)
    x_pred = np.ascontiguousarray(x_pred, np.float64)
    y_data = np.ascontiguousarray(y_data, np.float64)
    nd, ns = len(x_data), len(x_pred)
    nv = np.broadcast_to(np.asarray(noise_var, np.float64), (nd,))

    A = _cov_matrix(cov, x_data, x_data)
    A[np.diag_indices(nd)] += nv + jitter
    cA = _chol_factor(A)
    K_sD = _cov_matrix(cov, x_pred, x_data)

    # joint prior covariance over [x_*; x_D]; one dense Cholesky reused per draw.
    Xall = np.concatenate([x_pred, x_data], axis=0)
    K_all = _cov_matrix(cov, Xall, Xall)
    K_all[np.diag_indices(len(Xall))] += jitter
    L_all = np.linalg.cholesky(K_all)

    out = np.empty((n_samples, ns), np.float64)
    sig = np.sqrt(nv)
    for s in range(n_samples):
        f_prior = L_all @ rng.standard_normal(len(Xall))
        f_star, f_D = f_prior[:ns], f_prior[ns:]
        resid = y_data + sig * rng.standard_normal(nd) - f_D
        out[s] = f_star + K_sD @ _chol_solve(cA, resid)
    return out


def conditional_overdensity_los(
    x_obs: np.ndarray,
    nbar_obs,
    x_pred: np.ndarray,
    cov,
    *,
    n_samples: int = 0,
    seed: int = 0,
    jitter: float = 1e-6,
):
    """Posterior ``1+δ`` at prediction points, conditioned on observed galaxy
    positions via the Gaussian-linearized point (Poisson) model.

    Each observed galaxy is a linearized observation of the overdensity field at
    its position, ``y_i = 1/n̄_i − 1`` with variance ``N_i = 1/n̄_i`` (the
    log-Gaussian-Cox-process linearization), where ``n̄_i`` is the local mean
    galaxy density (galaxies per unit volume in the same coordinates as ``cov``).
    The GP posterior of δ at ``x_pred`` then carries the field's correlation
    structure and a *calibrated* uncertainty — the properties a local-density
    estimate lacks — and extends into data-poor stretches via the kernel rather
    than reverting blindly to the mean.

    (A sharp, skewed lognormal contrast is obtained downstream by the rank-
    preserving ``DensityTransform`` — ``generative.build_generative_model(lognormal=True)``
    — which maps this Gaussian posterior to ``1+δ=exp(g)``. A *native* log-Gaussian
    conditioning is not used here: the delta-function observation ``y=1/n̄`` is a
    linear-δ quantity that exponentiates catastrophically, so a native log field
    would need a binned-count LGCP Laplace solve.)

    Returns ``(opd_mean, opd_var)`` (both ``(N_*,)``), and, if ``n_samples>0``,
    also ``opd_samples`` ``(n_samples, N_*)`` (Matheron draws). All ``1+δ`` are
    floored at 0.
    """
    nbar = np.broadcast_to(np.asarray(nbar_obs, np.float64), (len(x_obs),))
    y = 1.0 / nbar - 1.0
    nv = 1.0 / nbar
    mean, var = gp_posterior_dense(x_obs, y, nv, x_pred, cov, jitter=jitter)
    opd_mean = np.clip(1.0 + mean, 0.0, None)
    if n_samples <= 0:
        return opd_mean, var
    S = gp_sample_dense(x_obs, y, nv, x_pred, cov, n_samples=n_samples, seed=seed,
                        jitter=jitter)
    return opd_mean, var, np.clip(1.0 + S, 0.0, None)
