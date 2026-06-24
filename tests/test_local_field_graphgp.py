"""Local-volume GraphGP density field (``echoes/local_field_graphgp``) — a ``GriddedFieldContext``-
compatible ``1+δ`` cube, the GraphGP alternative to the Manticore/CF4 reconstructions.

Statistical gates (a grid GP draw is validated by its statistics, not bit-reproducibility — on a
regular grid, neighbour ties make different builds give distinct but equivalent draws):
  * **prior** lognormal Cox cube — strictly positive, spatial mean ≈ 1, marginal variance ≈
    ``exp(σ²)−1``, and a spatial autocorrelation that decays with lag (the kernel imprinted it);
  * **posterior_sample** cube conditioned on synthetic galaxies — finite and strictly positive.

Uses the Julia backend (the dense first-block Cholesky runs inside Julia, avoiding the host BLAS),
so it needs the graphgp-julia checkout; skipped otherwise.
"""
import os

import numpy as np
import pytest

pytest.importorskip("graphgp")
pytest.importorskip("jax")

import jax
jax.config.update("jax_enable_x64", True)

from echoes import graphgp_julia as ggj
from echoes.local_field_graphgp import build_local_gp_field
from echoes.field_grid import GriddedFieldContext

pytestmark = pytest.mark.skipif(not (os.path.exists(ggj.JULIA) and os.path.exists(ggj.DRIVER)),
                                reason="Julia GraphGP backend not available")


def _kernel(sigma2=1.0, r0=12.0, alpha=1.5, rmax=60.0, n=400):
    r = np.linspace(0.0, rmax, n)
    return (r, sigma2 * np.exp(-((r / r0) ** alpha)))


def test_prior_lognormal_cube_statistics():
    box, n, s2 = 200.0, 24, 1.0
    cov = _kernel(sigma2=s2)
    fc = build_local_gp_field(box, n, cov, mode="prior", backend="julia", n0=200, k=30, seed=5)

    assert isinstance(fc, GriddedFieldContext) and fc.delta.shape == (n, n, n)
    d = fc.delta
    assert np.isfinite(d).all() and (d > 0).all(), "lognormal 1+δ must be finite and positive"
    print(f"\nlocal prior cube: mean={d.mean():.3f} var={d.var():.3f} "
          f"(lognormal target var=exp(σ²)−1={np.exp(s2)-1:.3f})")
    assert abs(d.mean() - 1.0) < 0.15, f"spatial mean of 1+δ should be ≈1, got {d.mean():.3f}"
    assert 0.5 * (np.exp(s2) - 1) < d.var() < 1.5 * (np.exp(s2) - 1), "marginal variance off"

    # spatial autocorrelation of the Gaussian field g decays with lag (the kernel is imprinted)
    g = np.log(d) + 0.5 * s2
    g -= g.mean()
    c0 = float(np.mean(g * g))
    c1 = float(np.mean(g[:-1] * g[1:]))                      # lag-1 along axis 0
    cL = float(np.mean(g[:-n // 2] * g[n // 2:]))           # lag-n/2
    print(f"  autocorr g: c1/c0={c1/c0:.3f}  c(n/2)/c0={cL/c0:.3f}")
    assert c1 / c0 > 0.3, "neighbouring voxels should be correlated (smooth field)"
    assert c1 / c0 > cL / c0, "correlation should decay with separation"


def test_posterior_sample_cube_conditioned_on_galaxies():
    box, n = 200.0, 20
    cov = _kernel()
    rng = np.random.default_rng(1)
    gal = rng.uniform(-box / 3, box / 3, (300, 3))
    nbar = np.full(300, 0.5)
    fc = build_local_gp_field(box, n, cov, mode="posterior_sample", points_data=gal, nbar_data=nbar,
                              backend="julia", n0=128, k=30, seed=2)
    assert isinstance(fc, GriddedFieldContext) and fc.delta.shape == (n, n, n)
    assert np.isfinite(fc.delta).all() and (fc.delta > 0).all()
    print(f"\nlocal posterior cube: mean={fc.delta.mean():.3f} "
          f"min={fc.delta.min():.3f} max={fc.delta.max():.3f}")


if __name__ == "__main__":
    test_prior_lognormal_cube_statistics()
    test_posterior_sample_cube_conditioned_on_galaxies()
    print("ALL LOCAL FIELD GATES PASSED")
