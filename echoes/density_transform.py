"""Measured monotonic density transform ``1+δ = T(g)`` — the purely data-driven
non-Gaussian field engine (Tier A).

GraphGP / fieldpost draw a *Gaussian* overdensity field ``g`` with the data's
measured two-point structure: maximum-entropy given ξ(r), hence the right power
spectrum but the WRONG one-point PDF (symmetric, no skew, no empty voids, no
filament tail). This module measures the data's own one-point overdensity PDF
(counts-in-cells) and builds the monotonic map ``T`` that carries a standard
Gaussian to that PDF, so applying ``T`` to the GraphGP draw reshapes the marginal
to match the data while **leaving the rank ordering — hence the calibrated
posterior coverage — untouched** (a monotonic map is PIT/rank invariant).

Two maps, both differentiable (``np.interp`` ↔ ``jnp.interp``):

* ``empirical`` — ``T = F_data⁻¹ ∘ Φ`` (rank/histogram match). Reproduces the
  measured PDF *exactly*; the non-parametric default for the decision probe.
* ``lognormal`` — shifted-lognormal ``1+δ = (1+δ₀)·exp(σ g − σ²/2) − δ₀`` with
  ``(δ₀, σ)`` moment-matched to the measured (variance, skew), mean fixed to 1.
  A smooth 2-parameter parametric alternative (Coles & Jones 1991).

What this CAN do: fix the 1-pt PDF and the higher-order moments that 1-pt × 2-pt
induce — most of what kNN-CDF / counts-in-cells measure. What it CANNOT do:
manufacture genuine filament *phase coherence* (a phase correlation from
non-linear collapse, not a 1- or 2-pt property). That ceiling is the Tier-A → Tier-B
(disco-dj) escalation boundary; see ``snug-sleeping-micali`` plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def _ndtr(x):
    """Standard-normal CDF Φ(x) (numpy; ``scipy.special.ndtr`` if available)."""
    try:
        from scipy.special import ndtr
        return ndtr(x)
    except Exception:                                            # pragma: no cover
        from math import erf
        return 0.5 * (1.0 + np.vectorize(erf)(np.asarray(x) / np.sqrt(2.0)))


def _ndtri(p):
    """Inverse standard-normal CDF Φ⁻¹(p)."""
    from scipy.special import ndtri
    return ndtri(np.clip(p, 1e-12, 1.0 - 1e-12))


@dataclass
class DensityTransform:
    """A measured monotonic map ``T: g(standard-normal) → 1+δ`` and its inverse.

    ``apply_to_field`` is the production entry point: it gaussianises a GraphGP /
    fieldpost ``1+δ`` field against a reference (prior) mean+std and pushes it
    through ``T``. With ``kind='identity'`` it is a no-op (the Stage-1 skeleton
    that must reproduce ``fieldpost`` byte-for-byte).
    """

    kind: str                                  # 'empirical' | 'lognormal' | 'identity'
    scale: float = 8.0                         # CiC smoothing radius R [Mpc/h]
    mean_opd: float = 1.0                       # measured mean of (1+δ)
    var_opd: float = 0.0                        # measured variance of (1+δ)
    skew_opd: float = 0.0                       # measured skew of (1+δ)
    # lognormal params
    sigma_g: float = 0.0
    delta0: float = 0.0                         # shift (shifted-lognormal)
    # empirical params: monotone table keyed on the standard-normal variate g
    # (knots uniform in g, where the map is evaluated → tail resolved exactly)
    g_grid: Optional[np.ndarray] = field(default=None, repr=False)
    val_grid: Optional[np.ndarray] = field(default=None, repr=False)

    # ---- forward / inverse point maps -----------------------------------
    def T(self, g):
        """Standard-normal variate ``g`` → ``1+δ`` (>= 0, monotone in g)."""
        g = np.asarray(g, float)
        if self.kind == "identity":
            return g
        if self.kind == "lognormal":
            opd = (1.0 + self.delta0) * np.exp(self.sigma_g * g - 0.5 * self.sigma_g ** 2) - self.delta0
            return np.clip(opd, 0.0, None)
        # empirical: T = F_data^{-1} ∘ Φ, tabulated on g-knots
        return np.interp(g, self.g_grid, self.val_grid)

    def T_inv(self, opd):
        """``1+δ`` → standard-normal variate (for calibration round-trip checks)."""
        opd = np.asarray(opd, float)
        if self.kind == "identity":
            return opd
        if self.kind == "lognormal":
            y = np.clip((opd + self.delta0) / (1.0 + self.delta0), 1e-12, None)
            return (np.log(y) + 0.5 * self.sigma_g ** 2) / max(self.sigma_g, 1e-12)
        return np.interp(opd, self.val_grid, self.g_grid)

    # ---- field application ----------------------------------------------
    def apply_to_field(self, opd, *, mu=None, sigma=None):
        """Map a GraphGP/fieldpost ``1+δ`` field through ``T``.

        The field is gaussianised to a unit normal ``g = (opd − mu)/sigma`` against
        a reference (the prior mean ``mu`` and std ``sigma``), then pushed through
        ``T``. ``mu``/``sigma`` default to the field's own moments — appropriate
        for a prior-dominated draw (Stage 0). Monotone in ``opd`` ⇒ rank-preserving
        ⇒ posterior coverage is unchanged. Returns identity if ``kind='identity'``.
        """
        opd = np.asarray(opd, float)
        if self.kind == "identity":
            return opd
        if mu is None:
            mu = float(np.mean(opd))
        if sigma is None:
            sigma = float(np.std(opd))
        if not np.isfinite(sigma) or sigma <= 1e-12:
            return opd
        g = (opd - mu) / sigma
        return self.T(g)


def _moments(x):
    x = np.asarray(x, float)
    m = x.mean()
    v = x.var()
    s = ((x - m) ** 3).mean() / max(v, 1e-12) ** 1.5
    return float(m), float(v), float(s)


def _fit_shifted_lognormal(var, skew):
    """Moment-match a unit-mean shifted-lognormal ``(δ₀, σ)`` to (var, skew).

    For ``1+δ = (1+δ₀)·exp(σg − σ²/2) − δ₀`` (mean 1):
        var  = (1+δ₀)² · (e^{σ²} − 1)
        skew = (ω + 3) · √(e^{σ²} − 1),  ω = e^{σ²}
    Solve skew → σ first (independent of δ₀), then var → δ₀. Falls back to the
    plain lognormal (δ₀ = 0) if the skew is too small / inconsistent.
    """
    var = max(float(var), 1e-8)
    skew = float(skew)
    # solve (ω+3)·√(ω−1) = skew for ω>1 by bisection (monotone in ω)
    if skew <= 1e-3:
        sigma = float(np.sqrt(np.log1p(var)))                   # plain lognormal
        return 0.0, sigma
    lo, hi = 1.0 + 1e-9, 50.0
    f = lambda w: (w + 3.0) * np.sqrt(w - 1.0) - skew
    if f(hi) < 0:                                               # skew beyond model reach
        w = hi
    else:
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if f(mid) > 0:
                hi = mid
            else:
                lo = mid
        w = 0.5 * (lo + hi)
    sigma = float(np.sqrt(np.log(w)))
    one_plus_d0 = float(np.sqrt(var / max(w - 1.0, 1e-12)))
    delta0 = max(one_plus_d0 - 1.0, -0.999)
    return delta0, sigma


def field_moments_from_counts(counts):
    """Shot-noise-free continuous-field moments ``(mean=1, var, skew)`` of ``1+δ``
    from raw cell COUNTS via factorial moments.

    Cell counts ``N ~ Poisson(λ)`` with ``λ = n̄·(1+δ)``. Factorial moments isolate
    the field: ``⟨N⟩=⟨λ⟩``, ``⟨N(N-1)⟩=⟨λ²⟩``, ``⟨N(N-1)(N-2)⟩=⟨λ³⟩`` — so the
    field moments ``⟨(1+δ)^k⟩ = ⟨N!/(N-k)!⟩/⟨N⟩^k`` carry NO Poisson contribution.
    Fitting the transform to these (rather than to ``N/⟨N⟩``, which is Poisson-
    broadened) prevents the double-counting of shot noise that otherwise over-skews
    the re-sampled field (Stage-0 finding: empirical CiC skew 3.6 vs data 3.1)."""
    N = np.asarray(counts, float)
    nb = N.mean()
    if nb <= 0:
        return 1.0, 0.0, 0.0
    m2 = (N * (N - 1)).mean() / nb ** 2                          # ⟨(1+δ)²⟩
    m3 = (N * (N - 1) * (N - 2)).mean() / nb ** 3                # ⟨(1+δ)³⟩
    var = max(m2 - 1.0, 1e-8)
    skew = (m3 - 3.0 * m2 + 2.0) / var ** 1.5
    return 1.0, float(var), float(skew)


def fit_density_transform(opd_cells, *, kind="empirical", scale=8.0, n_grid=512, counts=None):
    """Build a :class:`DensityTransform` from measured counts-in-cells overdensities.

    Parameters
    ----------
    opd_cells : array
        Measured ``1+δ`` values in fixed apertures (e.g. ``N_cell / ⟨N_cell⟩`` from
        :func:`validation.higher_order.cic` on the data, random-normalised). These
        carry the data's own (optionally systematics-weighted) one-point PDF.
    kind : {'empirical','lognormal','identity'}
    scale : float
        The CiC smoothing radius the PDF was measured at [Mpc/h] (bookkeeping).
    n_grid : int
        Quantile-table resolution for the empirical map.
    counts : array, optional
        Raw cell COUNTS (not ``1+δ``). When given with ``kind='lognormal'`` the
        transform is fit to the SHOT-NOISE-FREE field moments
        (:func:`field_moments_from_counts`) — the recommended, unbiased fit.
    """
    opd = np.asarray(opd_cells, float)
    opd = opd[np.isfinite(opd)]
    opd = np.clip(opd, 0.0, None)
    m, v, s = _moments(opd)
    dt = DensityTransform(kind=kind, scale=float(scale), mean_opd=m, var_opd=v, skew_opd=s)
    if kind == "identity":
        return dt
    if kind == "lognormal":
        if counts is not None:                                  # shot-noise-deconvolved
            _, v, s = field_moments_from_counts(counts)
            dt.var_opd, dt.skew_opd = v, s
        dt.delta0, dt.sigma_g = _fit_shifted_lognormal(v, s)
        return dt
    if kind != "empirical":
        raise ValueError(f"unknown transform kind {kind!r}")
    # empirical monotone quantile table keyed on g (knots uniform in the variate we
    # feed): val_grid[i] = data quantile at Φ(g_grid[i]). Uniform-in-g spacing puts
    # resolution where the map is evaluated, so the heavy 1+δ tail is captured (a
    # uniform-in-u grid linearly over-shoots the convex tail and inflates var/skew).
    g_grid = np.linspace(-5.0, 5.0, int(n_grid))
    val_grid = np.quantile(opd, np.clip(_ndtr(g_grid), 0.0, 1.0))
    val_grid = np.maximum.accumulate(val_grid)                  # enforce monotone
    dt.g_grid = g_grid
    dt.val_grid = val_grid
    return dt


if __name__ == "__main__":                                      # quick self-check
    rng = np.random.default_rng(0)
    # synthetic skewed target: a lognormal sample
    g0 = rng.normal(size=200_000)
    target = np.exp(0.7 * g0 - 0.5 * 0.7 ** 2)                   # mean 1, skewed
    print("target moments (mean,var,skew):", tuple(round(x, 3) for x in _moments(target)))
    for kind in ("empirical", "lognormal"):
        dt = fit_density_transform(target, kind=kind)
        out = dt.T(rng.normal(size=200_000))
        # round-trip on the unclipped support (the void floor 1+δ→0 is intentionally
        # non-invertible; check only where T(g) stays positive)
        gg = np.linspace(-2.5, 2.5, 50)
        err = float(np.max(np.abs(dt.T_inv(dt.T(gg)) - gg)))
        print(f"  {kind:9s} -> recovered:", tuple(round(x, 3) for x in _moments(out)),
              " T_inv∘T max err:", round(err, 4))
