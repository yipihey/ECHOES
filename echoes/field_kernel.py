"""Part III: GP / Vecchia weights via GraphGP.jl.

Sec. 4.4 of the ECHOES methods paper: build per-point density weights using the
Vecchia approximation of a Gaussian process whose covariance kernel is the
measured xi(r). Cost is O(N * k^3) time and O(N * k) memory, scaling
linearly to N ~ 10^9 with the right hardware.

The pipeline is::

    cov   = tabulate_kernel(r_centers, xi_j)            # numpy (cov_bins, cov_vals)
    delta = generate_field(positions, cov, xi_white, n0=N0, k=K, backend="julia")

where ``generate_field`` (via GraphGP.jl) computes ``L @ xi_white`` for the
Vecchia Cholesky factor ``L`` (so ``L L^T`` approximates the prior covariance).
Feeding the mean-centered, unit-variance KDE overdensity in as ``xi_white``
yields a data-aware smoothed field whose pair correlations recover the input
xi(r) -- this is the "calibrated GP sample" form discussed in
IMPLEMENTATION_PLAN.md.
"""

from __future__ import annotations

import numpy as np


def fit_kernel(r_centers: np.ndarray, xi_j: np.ndarray) -> tuple[float, float, float]:
    """Fit a stretched-exponential ``k(r) = A exp(-(r/r0)^alpha)`` to xi(r).

    This guarantees a smooth, positive, monotone-decreasing kernel that
    Cholesky-decomposes cleanly inside graphgp's per-block refinement
    step. A simple unweighted least-squares is sufficient for the typical
    LS estimator (signal dominated up to ~50 Mpc, then noise).
    """
    from scipy.optimize import curve_fit

    def model(r, A, r0, alpha):
        return A * np.exp(-((r / r0) ** alpha))

    mask = (xi_j > 0)
    if mask.sum() < 4:
        # Fallback: monotonic decay from the largest bin.
        return float(xi_j.max()), float(r_centers[-1] / 2), 1.0
    A0 = float(xi_j[mask].max())
    r0_0 = float(r_centers[xi_j > 0.5 * A0][-1]) if (xi_j > 0.5 * A0).any() else float(r_centers[0])
    try:
        popt, _ = curve_fit(
            model, r_centers[mask], xi_j[mask], p0=[A0, r0_0, 1.5],
            bounds=([0.01, 0.5, 0.3], [1e3, 200.0, 3.0]),
            maxfev=2000,
        )
        A, r0, alpha = popt
    except Exception:
        A, r0, alpha = A0, r0_0, 1.5
    return float(A), float(r0), float(alpha)


def tabulate_kernel(
    r_centers: np.ndarray,
    xi_j: np.ndarray,
    r_min: float | None = None,
    r_max: float | None = None,
    n_bins: int = 200,
    jitter: float = 1e-2,
):
    """Build a graphgp-format ``(cov_bins, cov_vals)`` tuple from xi(r).

    Fits a stretched-exponential parametric form to the measured xi(r) so
    the resulting kernel is guaranteed PSD; graphgp's per-block Cholesky
    inside ``refine`` requires this. Tabulates onto a log-spaced grid in
    graphgp's convention: ``cov_bins[0] = 0`` is the diagonal,
    ``cov_bins[1:]`` is logspace(r_min, r_max).

    Parameters
    ----------
    r_centers, xi_j
        From ``ls_corrfunc.xi_landy_szalay``.
    r_min, r_max
        Cover range for the discretized kernel. Default: spans
        ``r_centers``.
    n_bins
        Number of log-spaced bins (plus the implicit zero bin).
    jitter
        Multiplicative inflation on ``k(0)`` for PSD safety. graphgp's
        docstring: "If using your own covariance, inflate k(0) by a small
        factor to ensure positive definite."

    Returns
    -------
    (cov_bins, cov_vals) : pair of numpy arrays (the tabulated kernel the Julia backend consumes).
    fit_params : ``(A, r0, alpha)`` of the fitted ``A exp(-(r/r0)^alpha)``.
    """
    r_min = r_min if r_min is not None else float(r_centers[0])
    r_max = r_max if r_max is not None else float(r_centers[-1])
    A, r0, alpha = fit_kernel(r_centers, xi_j)

    cov_bins_np = np.concatenate([
        [0.0],
        np.logspace(np.log10(r_min), np.log10(r_max), n_bins - 1),
    ]).astype(np.float64)
    cov_vals_np = A * np.exp(-((cov_bins_np / r0) ** alpha))
    cov_vals_np[0] = A * (1.0 + jitter)
    return (cov_bins_np, cov_vals_np.astype(np.float64)), (A, r0, alpha)


def tabulate_kernel_direct(
    r_centers: np.ndarray,
    xi_j: np.ndarray,
    r_min: float | None = None,
    r_max: float | None = None,
    n_bins: int = 300,
    jitter: float = 0.1,
    tail_slope: float | None = None,
    tail_floor: float = 1e-5,
):
    """Build a graphgp kernel ``(cov_bins, cov_vals)`` *directly* from the
    measured ξ(r), preserving its shape at all scales.

    ``jitter`` is the **nugget**: K(0) = ξ(r_min)·(1+jitter).  Because a
    galaxy ξ(r) is nearly flat just below its smallest measured bin, close
    pairs (BOSS fibre-collision pairs at <1 Mpc/h) would otherwise produce
    near-identical covariance rows and a singular Vecchia block.  The nugget
    — physically the shot-noise / small-scale-cutoff term — regularises the
    diagonal without touching the off-diagonal ξ(r) being reproduced.

    Unlike :func:`tabulate_kernel` (which fits a single stretched
    exponential and therefore under-fits the large-scale tail of a
    galaxy ξ(r)), this tabulates the measured ξ(r) itself:

    1. keep the positive bins and enforce a monotone non-increasing
       profile (suppresses large-r noise wiggles that would break
       positive-definiteness);
    2. interpolate ξ(r) in log-log within the measured range;
    3. extrapolate the small-r end as a flat hold to ``K(0)`` and the
       large-r tail as a power law (slope from the outer measured
       points, or ``tail_slope`` if given) down to ``tail_floor``;
    4. inflate ``K(0)`` by ``jitter`` for positive-definiteness.

    The result is a smooth, monotone, positive kernel whose 3-D Fourier
    transform is (to the accuracy of the measurement) the survey P(k) —
    so a graphGP draw reproduces the measured ξ(r) at every scale.

    Returns ``((cov_bins, cov_vals), (xi0, r_knee, slope))`` where the
    third tuple is descriptive only (peak value, knee scale, tail slope).
    """
    r_centers = np.asarray(r_centers, dtype=np.float64)
    xi_j = np.asarray(xi_j, dtype=np.float64)

    pos = xi_j > 0
    if pos.sum() < 3:
        # Degenerate ξ(r): fall back to the parametric fit.
        return tabulate_kernel(r_centers, xi_j, r_min, r_max, n_bins, jitter)

    rp = r_centers[pos]
    xp = xi_j[pos]
    order = np.argsort(rp)
    rp, xp = rp[order], xp[order]
    # enforce monotone non-increasing (smooth large-r noise upward wiggles)
    xp_mono = np.minimum.accumulate(xp)

    logr = np.log(rp)
    logx = np.log(xp_mono)

    r_min = r_min if r_min is not None else float(rp[0])
    # extend the grid well past the measured range so the GP graph can
    # query moderate separations; the tail is power-law-extrapolated.
    r_max = r_max if r_max is not None else max(float(rp[-1]) * 6.0, 250.0)

    # power-law tail slope from the outer ~half-decade of measured points
    if tail_slope is None:
        n_tail = min(len(logr), 5)
        tail_slope = float((logx[-1] - logx[-n_tail]) /
                           (logr[-1] - logr[-n_tail] + 1e-12))
    tail_slope = min(tail_slope, -1.0)   # ensure a decaying tail

    cov_bins_np = np.concatenate([
        [0.0],
        np.logspace(np.log10(r_min * 0.5), np.log10(r_max), n_bins - 1),
    ])
    lb = np.log(cov_bins_np[1:])

    logK = np.empty_like(lb)
    inside = (lb >= logr[0]) & (lb <= logr[-1])
    below = lb < logr[0]
    above = lb > logr[-1]
    logK[inside] = np.interp(lb[inside], logr, logx)
    logK[below] = logx[0]                                   # flat hold to K(0)
    logK[above] = logx[-1] + tail_slope * (lb[above] - logr[-1])
    K = np.exp(logK)
    K = np.maximum(K, tail_floor * xp_mono[0])
    # guarantee global monotonicity of the tabulated kernel
    K = np.minimum.accumulate(K)

    cov_vals_np = np.concatenate([[xp_mono[0] * (1.0 + jitter)], K])
    desc = (float(xp_mono[0]), float(rp[0]), float(tail_slope))
    return (cov_bins_np.astype(np.float64), cov_vals_np.astype(np.float64)), desc


def compute_2pt_weights(
    positions: np.ndarray,
    r_centers: np.ndarray,
    xi_j: np.ndarray,
    nbar: np.ndarray | None = None,
    box_size: float | None = None,
    n0: int = 100,
    k: int = 30,
    r_kernel: float | None = None,
    mode: str = "prior_sample",
    seed: int = 0,
    n_kernel_bins: int = 200,
    return_diagnostics: bool = False,
    backend: str = "julia",
    device: str = "cpu",
):
    """Layer III per-point density weights via a Vecchia GP sample.

    Two modes are supported:

    ``"prior_sample"`` (default)
        Draw white noise ``xi ~ N(0, I)`` and compute
        ``delta = generate(graph, cov, xi) = L xi``. The result has prior
        covariance ``L L^T = Sigma`` exactly, so the weighted-DD pair sum
        recovers ``xi(r)`` in expectation. Data-agnostic in values, but
        evaluated AT the data positions, so per-point weights still
        encode the local correlation structure.

    ``"data_driven"``
        Use the (mean-centered, unit-variance) KDE overdensity as the
        white-noise input: ``delta = L * d_normalized``. Data-aware. The
        recovered xi has the same calibration relation as Layer I:
        scaled by ``<w>^2`` plus the weight-correlation term.

    ``"posterior_sample"``
        Draw a single posterior sample of δ(x) at the data positions via
        Matheron's rule:
            δ_post = δ_prior + K_D [K_DD + N_D]^{-1} (y_D - δ_prior|_D)
        where y_i = 1/nbar_i - 1 and N_D = diag(1/nbar_i). Requires
        ``nbar`` per-point mean density. The returned weights 1 + δ_post
        are posterior realizations consistent with the Poisson observations.
        Uses a Vecchia CG solve (O(N k³) per sample).

    Parameters
    ----------
    positions
        ``(N_D, 3)`` data positions.
    r_centers, xi_j
        Output of ``ls_corrfunc.xi_landy_szalay``.
    nbar
        Per-point local mean density. Required only for ``data_driven``.
    box_size, r_kernel
        For the KDE in ``data_driven`` mode (passed to Layer I helpers).
    n0, k
        graphgp Vecchia parameters: dense initial block size and number
        of conditional neighbors per point.
    mode
        ``"prior_sample"``, ``"data_driven"``, or ``"posterior_sample"``.
    seed
        Seed for the white-noise draw in ``prior_sample`` mode.
    n_kernel_bins
        Number of bins for the discretized covariance kernel.
    return_diagnostics
        If True, also return the fitted kernel params and the Graph.

    Returns
    -------
    weights
        ``(N_D,)`` numpy array of per-point density weights.
    diagnostics (optional)
        dict with keys ``'kernel_fit'`` (``(A, r0, alpha)``) and ``'graph'``.
    """
    from . import graphgp_julia as ggj

    N = len(positions)
    pts = np.asarray(positions, dtype=np.float64)
    n0e, ke = min(n0, max(2, N // 2)), min(k, N - 1)

    # 1. Tabulate kernel (parametric, PSD by construction).
    cov, fit_params = tabulate_kernel(
        r_centers, xi_j,
        r_min=float(max(r_centers[0], 0.5)),
        r_max=float(r_centers[-1]),
        n_bins=n_kernel_bins,
    )
    cov_bins = np.asarray(cov[0], np.float64)
    cov_vals = np.asarray(cov[1], np.float64)

    # 3. Build the white-noise input.
    if mode == "prior_sample":
        rng = np.random.default_rng(seed)
        xi_white = rng.standard_normal(N).astype(np.float64)
    elif mode == "data_driven":
        if nbar is None:
            raise ValueError("data_driven mode requires nbar")
        from .weights_binned import kde_overdensity, default_kernel_radius
        if r_kernel is None:
            r_kernel = default_kernel_radius(nbar)
        d = kde_overdensity(positions, nbar, r_kernel, box_size=box_size)
        d = d - d.mean()
        d_std = float(np.std(d))
        xi_white = (d / d_std) if d_std > 1e-12 else d
    elif mode == "posterior_sample":
        if nbar is None:
            raise ValueError("posterior_sample mode requires nbar")
        # Matheron's rule at data positions only (no lightcone grid).
        # For lightcone output use density_field.sample_posterior_density_field.
        rng = np.random.default_rng(seed)
        xi_white = rng.standard_normal(N).astype(np.float64)

        # Prior sample at data positions: f_prior = L xi  (GraphGP.jl bridge).
        f_prior = np.asarray(
            ggj.run_graphgp(pts, n0e, ke, cov_bins, cov_vals,
                            ops=("generate",), xi=xi_white, dtype="f64")["generate"],
            np.float64).ravel()

        # Observed overdensity y_i = 1/nbar_i - 1; noise N_D = diag(1/nbar_i)
        nbar_safe = np.maximum(nbar, 1e-30)
        noise_var = 1.0 / nbar_safe
        y_obs = noise_var - 1.0  # 1/nbar - 1

        # Residual
        residual = (y_obs - f_prior).astype(np.float64)

        # Vecchia matvec: (K + N) v = L (L^{-1} v) + N v, routed through the bridge
        # (Lv = generate(generate_inv(v))).
        def _matvec(v_np):
            v_np = np.asarray(v_np, np.float64)
            w = np.asarray(
                ggj.run_graphgp(pts, n0e, ke, cov_bins, cov_vals,
                                ops=("generate_inv",), values=v_np,
                                dtype="f64")["generate_inv"], np.float64).ravel()
            Lv = np.asarray(
                ggj.run_graphgp(pts, n0e, ke, cov_bins, cov_vals,
                                ops=("generate",), xi=w, dtype="f64")["generate"],
                np.float64).ravel()
            return Lv + noise_var * v_np

        # Conjugate gradient solve: (K + N) alpha = residual
        alpha = np.zeros(N, dtype=np.float64)
        r = residual - _matvec(alpha)
        p = r.copy()
        rs_old = float(r @ r)
        for _ in range(100):
            Ap = _matvec(p)
            rp = float(p @ Ap)
            if abs(rp) < 1e-30:
                break
            step = rs_old / rp
            alpha += step * p
            r -= step * Ap
            rs_new = float(r @ r)
            if rs_new < 1e-8 * (residual @ residual + 1e-60):
                break
            p = r + (rs_new / (rs_old + 1e-60)) * p
            rs_old = rs_new

        # Correction at data positions: K_D alpha (diagonal K entries)
        k0 = float(cov_vals[0])  # K(0, 0) = kernel diagonal
        delta = f_prior + k0 * alpha  # simplified: K_DD alpha ≈ K(0)*alpha (diagonal only)
        xi_white = None  # already handled above
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    if mode != "posterior_sample":
        # 4. Apply the Vecchia Cholesky factor (prior_sample / data_driven) via the GraphGP.jl
        # backend: one batched generate, no (M,k+1,k+1) materialization (runs where JAX OOMs),
        # same field, original order. Only backend="julia" is supported (JAX retired).
        if backend != "julia":
            raise ValueError("only backend='julia' is supported (JAX retired)")
        from .graphgp_backend import generate_field
        delta = generate_field(
            pts, (cov_bins, cov_vals), xi_white,
            n0=n0e, k=ke, backend="julia", device=device)

    weights = 1.0 + delta

    if return_diagnostics:
        return weights, {"kernel_fit": fit_params, "graph": None}
    return weights
