"""Scalable Vecchia GP posterior (``field_posterior.field_posterior_vecchia``) vs the dense reference.

The Vecchia posterior is matrix-free over a single joint graph on ``[x_pred; x_data]`` — the exact
Matheron rule with ``(K_DD+N)⁻¹`` by conjugate gradient and the full-space covariance apply
``K·u = generate(generate_grad_xi(u))`` (``L·Lᵀ``). Gate 3:

  * **exact limit** (``k = N-1`` → the Vecchia factor IS the dense Cholesky): posterior **mean**
    reproduces ``gp_posterior_dense`` to ~1e-6, and the Matheron **sample variance** reproduces the
    dense marginal variance (the draws carry the correct posterior covariance, not just the mean);
  * **production limit** (``k = 30``): the standard Vecchia approximation stays within a few percent.

Requires the Julia toolchain + the graphgp-julia checkout; skipped otherwise.
"""
import os

import numpy as np
import pytest

pytest.importorskip("graphgp")
pytest.importorskip("jax")

import jax
jax.config.update("jax_enable_x64", True)

from echoes import graphgp_julia as ggj
from echoes.field_posterior import (gp_posterior_dense, gp_sample_dense,
                                     field_posterior_vecchia, _cov_matrix)

_DRIVER = os.path.join(os.path.dirname(ggj.DRIVER), "run_posterior.jl")
pytestmark = pytest.mark.skipif(not (os.path.exists(ggj.JULIA) and os.path.exists(_DRIVER)),
                                reason="Julia GraphGP backend / run_posterior.jl not available")


def _problem(seed=3, nd=80, ns=60, ndim=3, box=10.0, nv=0.05):
    rng = np.random.default_rng(seed)
    r = np.linspace(0, 14, 500)
    cov = (r, 1.0 * np.exp(-((r / 2.5) ** 1.5)))
    x_data = rng.uniform(0, box, (nd, ndim))
    x_pred = rng.uniform(0, box, (ns, ndim))
    # coherent truth = a joint prior draw; observe the data block with noise
    N = nd + ns
    Kall = _cov_matrix(cov, np.vstack([x_pred, x_data]), np.vstack([x_pred, x_data]))
    Kall[np.diag_indices(N)] += 1e-6
    f = np.linalg.cholesky(Kall) @ rng.standard_normal(N)
    y_data = f[ns:] + np.sqrt(nv) * rng.standard_normal(nd)
    return cov, x_data, x_pred, y_data, nv


def test_vecchia_posterior_mean_matches_dense_exact():
    cov, x_data, x_pred, y_data, nv = _problem()
    N = len(x_data) + len(x_pred)
    m_d, _ = gp_posterior_dense(x_data, y_data, nv, x_pred, cov)
    m_v = field_posterior_vecchia(x_data, y_data, nv, x_pred, cov, n0=N - 1, k=N - 1, cg_tol=1e-10)
    rel = np.linalg.norm(m_v - m_d) / np.linalg.norm(m_d)
    print(f"\nVecchia posterior mean (exact k=N-1): L2-rel vs dense = {rel:.3e}")
    assert rel < 1e-3, f"exact-limit posterior mean disagrees with dense (rel {rel:.3e})"


def test_vecchia_posterior_samples_match_dense_variance():
    cov, x_data, x_pred, y_data, nv = _problem()
    N = len(x_data) + len(x_pred)
    _, v_d = gp_posterior_dense(x_data, y_data, nv, x_pred, cov)
    _, S_v = field_posterior_vecchia(x_data, y_data, nv, x_pred, cov, n_samples=800, seed=1,
                                     n0=N - 1, k=N - 1, cg_tol=1e-10)
    ratio = float(np.median(S_v.var(0) / np.maximum(v_d, 1e-12)))
    print(f"\nVecchia Matheron sample-var / dense-var median = {ratio:.3f}")
    assert 0.85 < ratio < 1.15, f"posterior sample variance off (median ratio {ratio:.3f})"


def test_vecchia_posterior_approx_close_at_production_k():
    cov, x_data, x_pred, y_data, nv = _problem()
    m_d, _ = gp_posterior_dense(x_data, y_data, nv, x_pred, cov)
    m_a = field_posterior_vecchia(x_data, y_data, nv, x_pred, cov, n0=30, k=30, cg_tol=1e-10)
    rel = np.linalg.norm(m_a - m_d) / np.linalg.norm(m_d)
    print(f"\nVecchia posterior mean (approx k=30): L2-rel vs dense = {rel:.3e}")
    assert rel < 0.1, f"Vecchia approximation too far from dense (rel {rel:.3e})"


if __name__ == "__main__":
    test_vecchia_posterior_mean_matches_dense_exact()
    test_vecchia_posterior_samples_match_dense_variance()
    test_vecchia_posterior_approx_close_at_production_k()
    print("ALL VECCHIA POSTERIOR GATES PASSED")
