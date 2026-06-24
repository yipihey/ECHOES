"""Local-volume GraphGP density field — a ``GriddedFieldContext``-compatible ``1+δ`` cube built by
GraphGP (Julia or JAX backend), an alternative/complement to the externally-reconstructed Manticore
/ CF4 cubes that ``local_completion`` consumes.

Two modes:
  ``"prior"``            a lognormal Cox field with the measured ξ(r) covariance: draw a Gaussian
                         GP ``g`` on the grid (``generate``) and map ``1+δ = exp(g − σ²/2)`` (mean 1).
                         Data-agnostic in value but carries the right clustering — the sharp-contrast
                         local-volume sampler.
  ``"posterior_sample"`` a conditional Matheron realization: ``g`` is a posterior draw at the grid
                         conditioned on the local galaxies (``field_posterior_vecchia``), then mapped
                         lognormally — a true-3D inpainted field that honours the data where present
                         and reverts to the prior where it is not.

Both reach the GraphGP.jl backend over the NPZ bridge (``backend="julia"``, the no-OOM path for big
grids) or stay on JAX (``backend="jax"``). The cube spans ``[-box/2, +box/2]`` per axis about the
observer, matching ``GriddedFieldContext``.
"""
from __future__ import annotations

import numpy as np

from .field_grid import GriddedFieldContext


def _grid_points(n_grid, box_mpc):
    """Observer-centred (n,n,n) voxel-centre coordinates, returned as ((n^3,3) points, axis)."""
    ax = (np.arange(n_grid) + 0.5) / n_grid * box_mpc - box_mpc / 2.0       # voxel centres
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    return pts, ax


def build_local_gp_field(
    box_mpc: float,
    n_grid: int,
    cov,
    *,
    mode: str = "prior",
    points_data: np.ndarray = None,
    nbar_data=None,
    backend: str = "julia",
    device: str = "cpu",
    n0: int = 256,
    k: int = 30,
    seed: int = 0,
    sigma2: float = None,
    jitter: float = 1e-6,
):
    """Build a ``GriddedFieldContext`` ``1+δ`` cube on an observer-centred grid.

    Parameters
    ----------
    box_mpc, n_grid : cube side (Mpc) and per-axis voxel count (cube is ``n_grid³``).
    cov : ``(cov_bins, cov_vals)`` tabulated kernel (from ``field_kernel.tabulate_kernel``).
    mode : ``"prior"`` (lognormal Cox draw) or ``"posterior_sample"`` (conditioned on the galaxies).
    points_data, nbar_data : local galaxy comoving positions ``(N_D,3)`` and per-point mean density;
        required for ``"posterior_sample"`` (linearized obs ``y=1/n̄−1``, noise ``1/n̄``).
    backend : ``"julia"`` | ``"jax"``.   sigma2 : field variance for the lognormal mean correction
        (defaults to ``cov_vals[0]=K(0)``).

    Returns
    -------
    ``GriddedFieldContext`` with ``delta`` the ``(n,n,n)`` ``1+δ`` cube, ``box_mpc`` set.
    """
    cov_bins = np.asarray(cov[0], np.float64)
    cov_vals = np.asarray(cov[1], np.float64)
    s2 = float(cov_vals[0]) if sigma2 is None else float(sigma2)
    grid_pts, _ = _grid_points(n_grid, box_mpc)
    N = len(grid_pts)

    if mode == "prior":
        from .graphgp_backend import generate_field
        rng = np.random.default_rng(seed)
        xi = rng.standard_normal(N)
        g = generate_field(grid_pts, (cov_bins, cov_vals), xi, n0=n0, k=k,
                           backend=backend, device=device)
    elif mode == "posterior_sample":
        if points_data is None or nbar_data is None:
            raise ValueError("posterior_sample mode requires points_data and nbar_data")
        from .field_posterior import field_posterior_vecchia
        nbar = np.broadcast_to(np.asarray(nbar_data, np.float64), (len(points_data),))
        y = 1.0 / nbar - 1.0
        noise = 1.0 / nbar
        _, samples = field_posterior_vecchia(points_data, y, noise, grid_pts, (cov_bins, cov_vals),
                                             n_samples=1, seed=seed, n0=n0, k=k, jitter=jitter)
        g = samples[0]
    else:
        raise ValueError(f"unknown mode {mode!r} (expected 'prior' or 'posterior_sample')")

    delta = np.exp(g - 0.5 * s2).reshape(n_grid, n_grid, n_grid)            # lognormal 1+δ, mean→1
    return GriddedFieldContext(delta=np.ascontiguousarray(delta, np.float64), box_mpc=float(box_mpc))
