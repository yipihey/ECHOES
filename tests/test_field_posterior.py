"""Correctness of the dense conditional-GP field posterior (the Matheron solve).

Self-consistency checks (no external GP library needed):
  * noiseless limit: posterior mean interpolates the data, variance → 0 at data;
  * Matheron sample mean/variance match the closed-form posterior mean/variance;
  * conditioning reduces the variance below the prior K(0) near data.
"""
import numpy as np
import pytest

pytest.importorskip("graphgp")
pytest.importorskip("jax")

from echoes.field_posterior import gp_posterior_dense, gp_sample_dense, _cov_matrix, _k0


def _strexp_kernel(r0=2.0, alpha=1.5, amp=1.0, rmax=14.0, n=500):
    # stretched exponential (the ECHOES tabulate_kernel form); alpha<2 keeps the
    # covariance well-conditioned, unlike a squared-exponential.
    r = np.linspace(0.0, rmax, n)
    k = amp * np.exp(-((r / r0) ** alpha))
    return (r, k)


def _toy(seed=0, nd=60, ns=80, ndim=2, box=10.0):
    rng = np.random.default_rng(seed)
    cov = _strexp_kernel()
    x_data = rng.uniform(0, box, (nd, ndim))
    x_pred = rng.uniform(0, box, (ns, ndim))
    # true field = a joint prior draw over all points
    Xall = np.concatenate([x_data, x_pred], 0)
    K = _cov_matrix(cov, Xall, Xall) + 1e-6 * np.eye(len(Xall))
    f_true = np.linalg.cholesky(K) @ rng.standard_normal(len(Xall))
    return cov, x_data, x_pred, f_true[:nd], f_true[nd:], rng


def test_noiseless_interpolation_and_zero_variance():
    cov, x_data, x_pred, f_data, f_pred, rng = _toy(seed=1)
    # predict AT the data points with tiny noise -> interpolate, variance ~ 0
    mean, var = gp_posterior_dense(x_data, f_data, 1e-8, x_data, cov, jitter=1e-8)
    assert np.allclose(mean, f_data, atol=1e-3)
    assert np.max(var) < 1e-3 * _k0(cov)


def test_conditioning_reduces_variance():
    cov, x_data, x_pred, f_data, f_pred, rng = _toy(seed=2)
    noise = 0.05 * _k0(cov)
    _, var = gp_posterior_dense(x_data, f_data, noise, x_pred, cov)
    # posterior variance is below the prior variance K(0) everywhere
    assert np.all(var <= _k0(cov) + 1e-9)
    # and strictly below for prediction points that have a near data neighbour
    from scipy.spatial import cKDTree
    dnn, _ = cKDTree(x_data).query(x_pred)
    near = dnn < 1.0
    assert near.any() and np.median(var[near]) < 0.8 * _k0(cov)


def test_matheron_samples_match_closed_form():
    cov, x_data, x_pred, f_data, f_pred, rng = _toy(seed=3, nd=50, ns=40)
    noise = 0.1 * _k0(cov)
    mean, var = gp_posterior_dense(x_data, f_data, noise, x_pred, cov)
    S = gp_sample_dense(x_data, f_data, noise, x_pred, cov, n_samples=4000, seed=7)
    samp_mean = S.mean(0)
    samp_var = S.var(0)
    # Monte-Carlo: mean within a few sigma/sqrt(N); variance within ~10%
    se = np.sqrt(var / S.shape[0])
    assert np.max(np.abs(samp_mean - mean) / (se + 1e-12)) < 5.0
    ok = var > 1e-6 * _k0(cov)
    assert np.median(np.abs(samp_var[ok] / var[ok] - 1.0)) < 0.12


def test_recovers_truth_near_data():
    cov, x_data, x_pred, f_data, f_pred, rng = _toy(seed=4, nd=120, ns=120)
    noise = 0.02 * _k0(cov)
    y = f_data + np.sqrt(noise) * rng.standard_normal(len(f_data))
    mean, var = gp_posterior_dense(x_data, y, noise, x_pred, cov)
    from scipy.spatial import cKDTree
    dnn, _ = cKDTree(x_data).query(x_pred)
    near = dnn < 0.6
    # where data is dense, the posterior mean tracks the true field
    assert near.sum() > 10
    rms_near = np.sqrt(np.mean((mean[near] - f_pred[near]) ** 2))
    assert rms_near < 0.4 * np.sqrt(_k0(cov))
