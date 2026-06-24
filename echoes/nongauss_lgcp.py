"""Non-Gaussian LGCP Gaussianization — the kernel re-derivation that lets the LGCP use the
MEASURED density transform ``T(g)`` (echoes.density_transform) instead of the bare lognormal,
WITHOUT breaking the two-point match. This is the core of covariance next-step #4 (which absorbs
the dead #1: cf-escalation can't kill multi-occupancy because the σ²=4.14 lognormal tail is
intrinsic; the measured T, with the data's CiC skew≈3.1 instead of the lognormal's ≈500, bounds
the peak intensity and the multi-occupancy collapses — see DIAGNOSTICS.md).

The construction (standard Gaussianized-field / "lognormal-generalisation"):

  The LGCP draws a Gaussian field ``f`` (UNIT variance here) with correlation ``ρ(r)`` and forms the
  intensity ``1+δ = T(f)``. For a transform ``T`` of a unit normal, the intensity correlation is
      ξ_T(ρ) = ⟨T(g_i) T(g_j)⟩ − ⟨T⟩²            (g_i,g_j ~ bivariate-normal, corr ρ)
  a monotone function of ρ. So to reproduce a TARGET intensity correlation ξ_in(r) (the
  window-deconvolved measurement) we set the Gaussian correlation to
      ρ(r) = ξ_T⁻¹( ξ_in(r) )
  and build the kernel from ρ (unit diagonal). Generation then applies ``opd = T(f)`` directly (no
  exp(f−σ²/2)). For the lognormal ``T(g)=exp(σ_g g − σ_g²/2)`` this reduces to the exact closed form
  ``ξ_T(ρ)=exp(σ_g² ρ)−1`` ⇒ ``ρ=log(1+ξ)/σ_g²`` — the relation kernel_from_K2d already hard-codes.
  Replacing that one inversion with ``ξ_T⁻¹`` for the measured T is the whole change; the 2-pt is
  preserved BY CONSTRUCTION, only the 1-point PDF (and the induced higher-order structure) changes.

``ξ_T(ρ)`` is computed by Gauss--Hermite quadrature of the bivariate-normal expectation; ``ξ_T⁻¹`` by
monotone interpolation on a ρ-grid. Pure-numpy, no GPU, cheap (a few thousand T evals).
"""
from __future__ import annotations

import numpy as np

from .density_transform import DensityTransform


def _gh(n):
    """Gauss--Hermite nodes/weights for ∫ e^{-x²} f(x) dx, rescaled to the N(0,1) measure
    (nodes ·√2, weights /√π) so ``Σ w_i f(x_i) ≈ E_{x~N(0,1)}[f(x)]``."""
    x, w = np.polynomial.hermite_e.hermegauss(n)   # probabilists' Hermite: weight e^{-x²/2}
    return x, w / np.sqrt(2.0 * np.pi)


def xi_of_rho(dt: DensityTransform, rho, *, n_gh: int = 48):
    """Intensity correlation ``ξ_T(ρ) = ⟨T(g_i)T(g_j)⟩ − ⟨T⟩²`` for a unit-normal pair of
    correlation ``ρ``. Vectorised over ``rho`` (array in [-1,1]).

    Bivariate expectation via the conditional decomposition ``g2 = ρ g1 + √(1−ρ²) z`` with g1,z iid
    N(0,1) on a Gauss--Hermite product grid. ``⟨T⟩`` is evaluated on the same 1-D grid (so the mean
    subtraction is consistent and ξ_T(0)=0 exactly)."""
    rho = np.asarray(rho, float)
    x, w = _gh(n_gh)                               # 1-D N(0,1) quadrature
    Tx = dt.T(x)                                   # T on the grid
    mean_T = float(w @ Tx)                         # ⟨T⟩
    # E[T(g1)T(g2)] = Σ_i w_i T(x_i) Σ_j w_j T(ρ x_i + √(1−ρ²) x_j)
    out = np.empty(rho.shape, float)
    flat = rho.ravel()
    res = np.empty(flat.size, float)
    for k, r in enumerate(flat):
        s = np.sqrt(max(1.0 - r * r, 0.0))
        # T at the conditional nodes: shape (n_gh grid for g1, n_gh grid for z)
        g2 = r * x[:, None] + s * x[None, :]
        Tg2 = dt.T(g2)                             # (n,n)
        inner = Tg2 @ w                            # Σ_j w_j T(...)  -> (n,)
        res[k] = (w * Tx) @ inner                  # Σ_i w_i T(x_i) inner_i
    out = res.reshape(rho.shape)
    return out - mean_T * mean_T


def rho_of_xi(dt: DensityTransform, xi_target, *, n_rho: int = 257, n_gh: int = 48,
              rho_max: float = 0.9995):
    """Invert ``ξ_T`` for the Gaussian correlation ``ρ`` reproducing ``ξ_target`` (the measured
    intensity correlation). Builds ξ_T on a ρ-grid once, then monotone-interpolates.

    ξ_T is monotone increasing in ρ (positive map), with ξ_T(0)=0. Targets above the model's reach
    ``ξ_T(ρ_max)`` clip to ρ_max; targets below ξ_T(−...) clip to the grid floor. Returns ρ same
    shape as ``xi_target``."""
    xi_target = np.asarray(xi_target, float)
    rho_grid = np.linspace(-rho_max, rho_max, n_rho)
    xi_grid = xi_of_rho(dt, rho_grid, n_gh=n_gh)
    xi_grid = np.maximum.accumulate(xi_grid)       # enforce monotone (guards quadrature wiggles)
    # np.interp needs increasing xp; xi_grid is increasing in rho
    rho = np.interp(xi_target, xi_grid, rho_grid, left=rho_grid[0], right=rho_grid[-1])
    return rho


def sigma_from_transform(dt: DensityTransform):
    """Marginal variance of ``1+δ=T(g)`` implied by the transform (for bookkeeping / σ² reporting)."""
    x, w = _gh(96)
    Tx = dt.T(x)
    m = float(w @ Tx)
    return float(w @ (Tx * Tx) - m * m)


if __name__ == "__main__":     # self-test: recover the lognormal closed form ξ_T(ρ)=exp(σ²ρ)−1
    from .density_transform import DensityTransform
    for sg in (0.5, 1.0, 1.5):
        dt = DensityTransform(kind="lognormal", sigma_g=sg, delta0=0.0)
        rho = np.array([-0.3, 0.0, 0.25, 0.5, 0.8, 0.95])
        num = xi_of_rho(dt, rho)
        exact = np.exp(sg * sg * rho) - 1.0
        err = np.max(np.abs(num - exact))
        # round-trip: rho_of_xi(xi_of_rho(rho)) ≈ rho
        rt = rho_of_xi(dt, num)
        rterr = np.max(np.abs(rt - rho))
        print(f"sigma_g={sg}:  max|ξ_num-ξ_exact|={err:.2e}   max|ρ roundtrip|={rterr:.2e}   "
              f"var(T)={sigma_from_transform(dt):.4f} (exact {np.exp(sg*sg)-1:.4f})")
