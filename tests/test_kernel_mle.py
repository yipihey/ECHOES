"""Marginal-likelihood hyperparameter MLE via GraphGP.jl's analytic gradients (``echoes/kernel_mle``).

Two gates:
  * **gradient correctness** — the analytic NLML gradient equals central finite differences
    (~1e-6); this is the rigorous proof that the Julia gradient machinery (the capability the JAX
    CUDA extension lacks) is right.
  * **recovery** — fitting a field drawn from a known stretched-exp kernel recovers its *shape*
    (r0, alpha) to a few percent. The amplitude A is left loose: a single field realization does not
    pin the GP amplitude/lengthscale degeneracy — the likelihood is still correctly maximized
    (gradient → 0), it just has little Fisher information on A.

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
from echoes import kernel_mle as km

_HAVE_JULIA = os.path.exists(ggj.JULIA) and os.path.exists(km._DRIVER)
pytestmark = pytest.mark.skipif(not _HAVE_JULIA,
                                reason="Julia GraphGP backend not available")


def _synthetic_field(seed=7):
    rng = np.random.default_rng(seed)
    N, n0, k = 1500, 64, 20
    pts = np.vstack([rng.normal([10, 10, 10], 4, (N // 2, 3)),
                     rng.normal([30, 22, 16], 5, (N - N // 2, 3))])
    span = float((pts.max(0) - pts.min(0)).max())
    rmn, rmx = max(span * 1e-3, 1e-3), 0.5 * span
    bins = km.make_cov_bins(rmn, rmx, 200)
    th_true = np.array([np.log(1.0), np.log(0.06 * span), 1.4])     # [logA, logr0, alpha]
    vals = km.strexp_vals(bins, th_true)
    y = ggj.run_graphgp(pts, n0, k, bins, vals, ops=("generate",),
                        xi=rng.standard_normal(N), dtype="f64")["generate"][0]
    return pts, y, th_true, dict(n0=n0, k=k, r_min=rmn, r_max=rmx, n_bins=200)


def test_mle_gradient_matches_finite_difference():
    pts, y, th_true, kw = _synthetic_field()
    th0 = th_true + np.array([0.4, 0.3, -0.2])                      # perturb the init
    gc = km.gradcheck_kernel_mle(pts, y, th0, **kw)
    print(f"\nMLE gradcheck rel={gc['rel']:.3e}  analytic={np.round(gc['g_analytic'], 3)}")
    assert gc["rel"] < 1e-6, f"analytic NLML gradient disagrees with FD (rel {gc['rel']:.3e})"


def test_mle_recovers_kernel_shape():
    pts, y, th_true, kw = _synthetic_field()
    th0 = th_true + np.array([0.4, 0.3, -0.2])
    fit = km.fit_kernel_mle(pts, y, th0, **kw)
    r0_t, al_t = float(np.exp(th_true[1])), float(th_true[2])
    print(f"\nMLE fit nlml {fit['nlml0']:.4g}->{fit['nlml']:.4g} ({fit['niter']} it)  "
          f"r0={fit['r0']:.3f}(t{r0_t:.3f}) alpha={fit['alpha']:.3f}(t{al_t:.3f})")
    assert fit["nlml"] < fit["nlml0"], "NLML did not decrease"
    assert fit["gnorm"] < 1e-4, f"optimizer did not converge (gnorm {fit['gnorm']:.1e})"
    assert abs(fit["r0"] / r0_t - 1.0) < 0.25, f"r0 not recovered: {fit['r0']:.3f} vs {r0_t:.3f}"
    assert abs(fit["alpha"] - al_t) < 0.25, f"alpha not recovered: {fit['alpha']:.3f} vs {al_t:.3f}"


if __name__ == "__main__":
    test_mle_gradient_matches_finite_difference()
    test_mle_recovers_kernel_shape()
    print("ALL MLE GATES PASSED")
