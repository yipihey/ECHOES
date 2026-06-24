"""Mock-catalog use #1 — a CLUSTERING COVARIANCE for CMASS-South from the Julia GraphGP
anisotropic-LGCP ensemble.

Pipeline (reuses the build_boss_lgcp_catalogs.py flow for the kernel):
  load_boss -> measure_K2d_data -> deconvolve_window -> kernel_from_K2d (AnisotropicCovariance, sigma2)
  -> generate_catalogs_from_kernel(n_samples=N)  [ONE graph build, N field draws]
  -> per realization measure wp(rp), xi0(s), xi2(s)  (echoes.clustering_measure, Corrfunc-backed)
  -> sample covariance across realizations (+ Hartlap-corrected inverse)
  -> convergence diagnostic  (errors vs N)
  -> DATA jackknife cross-check (angular regions)  -> compare jackknife vs mock diagonal variance.

Outputs go to data_release/boss_lgcp_julia/covariance/ (gitignored; not committed).

The graph build is AMORTIZED across the ensemble: generate_catalogs_from_kernel draws N
white-noise vectors as eps_all=(n_cand, N) and the Julia backend builds the 2.19M graph ONCE
(build_in_julia=True) and generates all N fields in a single subprocess (verified in
echoes/completion.py:generate_catalogs_from_kernel + echoes/graphgp_julia.py:run_graphgp).
So a single large-n_samples call is the cheap path; we also support --batch to stream batches
if memory at very large N is a concern (each batch REBUILDS the graph — documented limitation
below, see NOTE-AMORTIZATION).

Reduced-scale validation (machinery check, NOT production):
  JAX_PLATFORMS=cpu OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 \
    python pipeline/build_boss_covariance.py --n-cand-factor 3 --n-data-meas 20000 \
        --n-realizations 50 --device gpu

Full production (the only change is scale):
  JAX_PLATFORMS=cpu OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 \
    python pipeline/build_boss_covariance.py --n-cand-factor 20 --n-data-meas 60000 \
        --n-realizations 1000 --device gpu

NOTE-AMORTIZATION: generate_catalogs_from_kernel rebuilds the graph on EACH call (no
graph_npz reuse plumbed through it). One call with n_samples=N => one build, N draws (good).
Splitting into B batches => B builds. The Julia GPU build is ~40s, so for B<=a few the overhead
is small; for huge N a single call is strictly cheaper. We default to a single call and expose
--batch only as a memory-relief escape hatch (it pays one extra ~40s build per batch).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from echoes.surveys.boss import load_boss
from echoes.completion import (measure_K2d_data, deconvolve_window, kernel_from_K2d,
                               kernel_from_K2d_tierA, lgcp_density_transform,
                               generate_catalogs_from_kernel, fkp_weight_of_z)
from echoes.randoms import make_random_from_selection_function
from echoes.distance import comoving_distance
from echoes import clustering_measure as clm

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"
OUT = os.path.join(os.path.dirname(__file__), "..", "data_release", "boss_lgcp_julia", "covariance")


# ----------------------------------------------------------------- distances / weights
def cz_of_z(z, cosmo):
    """Comoving distance [Mpc/h] for redshifts z under the catalog fiducial cosmology."""
    import jax
    jax.config.update("jax_enable_x64", True)
    return np.asarray(comoving_distance(np.asarray(z, np.float64), cosmo))


def fkp_for(z, z_data, w_fkp_data):
    if w_fkp_data is None:
        return np.ones(len(z))
    return fkp_weight_of_z(np.asarray(z), np.asarray(z_data), np.asarray(w_fkp_data))


# --------------------------------------------------------------------------- jackknife
def kmeans_regions(ra, dec, n_regions, seed=0):
    """Assign each (ra, dec) to one of n_regions angular regions by k-means on the unit
    sphere (compact, roughly equal-area cells over the footprint). Returns integer labels."""
    from echoes.geometry import _radec_to_nhat
    rng = np.random.default_rng(seed)
    X = _radec_to_nhat(np.asarray(ra), np.asarray(dec))      # (N,3) unit vectors
    # k-means++ lite: random init from points, a few Lloyd iterations.
    cidx = rng.choice(len(X), n_regions, replace=False)
    C = X[cidx].copy()
    lab = np.zeros(len(X), dtype=int)
    for _ in range(25):
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        new = d2.argmin(1)
        if np.array_equal(new, lab) and _ > 0:
            lab = new
            break
        lab = new
        for j in range(n_regions):
            m = lab == j
            if m.any():
                c = X[m].mean(0)
                C[j] = c / (np.linalg.norm(c) + 1e-30)
    return lab


def jackknife_data(cat, rp_edges, s_edges, pimax, npibins, nmu, n_jk, nthreads, seed=0):
    """Delete-one-region jackknife of the DATA clustering. Returns dicts of mean and
    (jackknife) covariance for wp, xi0, xi2, plus the region count actually used."""
    ra = np.asarray(cat.ra_data); dec = np.asarray(cat.dec_data); z = np.asarray(cat.z_data)
    w_d = np.asarray(cat.w_data) if cat.w_data is not None else np.ones(len(ra))
    cz_d = cz_of_z(z, cat.fid_cosmo)

    # randoms for the LS estimator (shared across all jackknife deletions), labelled too
    rng = np.random.default_rng(seed)
    nr = max(4 * len(ra), 200_000)
    ra_r, dec_r, z_r = make_random_from_selection_function(
        sel_map=cat.sel_map, n_random=nr, z_data=z, nside=cat.nside, rng=rng)
    w_r = fkp_for(z_r, z, cat.w_fkp_data)
    cz_r = cz_of_z(z_r, cat.fid_cosmo)

    lab_d = kmeans_regions(ra, dec, n_jk, seed=seed)
    lab_r = kmeans_regions(ra_r, dec_r, n_jk, seed=seed)  # same init => consistent cells
    regions = np.unique(lab_d)
    n_used = len(regions)

    wp_l, xi0_l, xi2_l = [], [], []
    for j in regions:
        md = lab_d != j
        mr = lab_r != j
        _, wp = clm.measure_wp(ra[md], dec[md], cz_d[md], w_d[md],
                               ra_r[mr], dec_r[mr], cz_r[mr], w_r[mr],
                               rp_edges=rp_edges, pimax=pimax, npibins=npibins, nthreads=nthreads)
        _, xi0, xi2 = clm.measure_xi_ell(ra[md], dec[md], cz_d[md], w_d[md],
                                         ra_r[mr], dec_r[mr], cz_r[mr], w_r[mr],
                                         s_edges=s_edges, nmu=nmu, nthreads=nthreads)
        wp_l.append(wp); xi0_l.append(xi0); xi2_l.append(xi2)

    out = {}
    for name, arr in (("wp", wp_l), ("xi0", xi0_l), ("xi2", xi2_l)):
        A = np.asarray(arr)                       # (n_used, n_bins)
        mean = A.mean(0)
        # delete-one jackknife covariance: (n-1)/n * sum (x_i - xbar)(x_i - xbar)^T
        d = A - mean
        cov = (n_used - 1) / n_used * (d.T @ d)
        out[name] = {"mean": mean, "cov": cov, "diag": np.diag(cov)}
    return out, n_used


# ------------------------------------------------------------------ per-realization measure
def measure_realization(cat_ra, cat_dec, cat_z, w_d, ra_r, dec_r, cz_r, w_r, cosmo,
                        rp_edges, s_edges, pimax, npibins, nmu, nthreads, rr_cache):
    """wp/xi0/xi2 of one generated catalog (unweighted data by default; w_d optional)."""
    cz_d = cz_of_z(cat_z, cosmo)
    _, wp = clm.measure_wp(cat_ra, cat_dec, cz_d, w_d, ra_r, dec_r, cz_r, w_r,
                           rp_edges=rp_edges, pimax=pimax, npibins=npibins,
                           nthreads=nthreads, rr=rr_cache)
    _, xi0, xi2 = clm.measure_xi_ell(cat_ra, cat_dec, cz_d, w_d, ra_r, dec_r, cz_r, w_r,
                                     s_edges=s_edges, nmu=nmu, nthreads=nthreads, rr=rr_cache)
    return wp, xi0, xi2


# ---------------------------------------------------------------------- covariance helpers
def sample_cov(A):
    """Unbiased sample covariance of rows of A (n_real, n_bins) -> (n_bins, n_bins)."""
    A = np.asarray(A, np.float64)
    n = A.shape[0]
    d = A - A.mean(0)
    return (d.T @ d) / (n - 1)


def hartlap_factor(n_real, n_bins):
    """Anderson--Hartlap correction so Cinv_unbiased = h * inv(C_sample)."""
    return (n_real - n_bins - 2.0) / (n_real - 1.0)


def convergence_diag(A):
    """How a representative scalar (mean diagonal std) and one covariance element stabilise
    as the realization count grows. Returns N grid + the two tracked curves."""
    A = np.asarray(A, np.float64)
    n, nb = A.shape
    grid = np.unique(np.clip(np.linspace(5, n, 10).astype(int), 5, n))
    mean_diag_err = []
    rep_cov = []                       # covariance of the two most-variable bins
    var_all = A.var(0)
    b1, b2 = np.argsort(var_all)[-2:]
    for m in grid:
        C = sample_cov(A[:m])
        mean_diag_err.append(float(np.sqrt(np.mean(np.diag(C)))))
        rep_cov.append(float(C[b1, b2]))
    return grid, np.asarray(mean_diag_err), np.asarray(rep_cov), (int(b1), int(b2))


# ------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-realizations", type=int, default=50, help="ensemble size N")
    ap.add_argument("--n-cand-factor", type=int, default=3,
                    help="candidates = factor x N_data (20 = production)")
    ap.add_argument("--n-data-meas", type=int, default=20_000, help="data subsample for K_in")
    ap.add_argument("--n-rand-meas", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    ap.add_argument("--sampling", choices=["poisson", "bernoulli"], default="poisson")
    ap.add_argument("--transform", choices=["none", "lognormal"], default="none",
                    help="'lognormal' = Tier-A non-Gaussian intensity (#4): regularise the lognormal "
                         "skew via --skew and re-derive the kernel (kernel_from_K2d_tierA) so the 2-pt "
                         "is preserved while the multi-occupancy tail is cut. 'none' = bare lognormal.")
    ap.add_argument("--skew", type=float, default=10.0,
                    help="Tier-A intensity-PDF skew target (lognormal native ≈500-720; data CiC ≈3.1)")
    ap.add_argument("--batch", type=int, default=0,
                    help="if >0, stream the ensemble in batches of this many (each batch "
                         "REBUILDS the graph; only for memory relief at very large N)")
    ap.add_argument("--n-rand-meas-clust", type=int, default=4,
                    help="randoms-per-mock-galaxy factor for the clustering LS estimator")
    ap.add_argument("--n-jk", type=int, default=30, help="angular jackknife regions (25-50)")
    ap.add_argument("--nthreads", type=int, default=16, help="Corrfunc OpenMP threads")
    ap.add_argument("--use-mock-weights", action="store_true",
                    help="apply FKP(z) weights to mock galaxies too (default: unit data weights, "
                         "matching the unweighted Poisson mock; randoms always FKP-weighted)")
    ap.add_argument("--no-mask", action="store_true",
                    help="skip the exact mangle footprint mask (default: apply it to galaxies+randoms)")
    # binning
    ap.add_argument("--rp-min", type=float, default=0.5)
    ap.add_argument("--rp-max", type=float, default=40.0)
    ap.add_argument("--n-rp", type=int, default=12)
    ap.add_argument("--pimax", type=float, default=80.0)
    ap.add_argument("--s-min", type=float, default=1.0)
    ap.add_argument("--s-max", type=float, default=40.0)
    ap.add_argument("--n-s", type=int, default=14)
    ap.add_argument("--nmu", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    if not clm.has_corrfunc():
        print("[cov] WARNING: Corrfunc not importable — using the slow scipy cKDTree LS "
              "fallback (fine for the reduced-scale check, too slow for production).", flush=True)

    os.makedirs(OUT, exist_ok=True)
    rp_edges = np.logspace(np.log10(args.rp_min), np.log10(args.rp_max), args.n_rp + 1)
    s_edges = np.linspace(args.s_min, args.s_max, args.n_s + 1)
    pimax = args.pimax
    npibins = int(round(pimax))                       # 1 Mpc/h pi bins
    nmu = args.nmu
    rp_c = 0.5 * (rp_edges[:-1] + rp_edges[1:])
    s_c = 0.5 * (s_edges[:-1] + s_edges[1:])

    T0 = time.perf_counter(); t = {}

    # ---- load + kernel (the build_boss_lgcp_catalogs.py flow) --------------------------
    print("[cov] loading CMASS-South ...", flush=True)
    _t = time.perf_counter()
    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256)
    t["load"] = time.perf_counter() - _t
    w_comp = cat.w_sys_data * cat.w_noz_data * cat.w_cp_data
    n_cand = args.n_cand_factor * cat.N_data
    print(f"[cov] N_data={cat.N_data:,}  n_cand={n_cand:,}  N_real={args.n_realizations}  "
          f"[load {t['load']:.1f}s]", flush=True)

    te = np.concatenate([[0.0], np.geomspace(0.02, 2.5, 16)])
    ze = np.linspace(0.0, 0.03, 11)
    print("[cov] measuring K_in (window-deconvolved LS) ...", flush=True)
    _t = time.perf_counter()
    _, _, xi_w, cnt = measure_K2d_data(cat, theta_edges=te, z_edges=ze,
                                       n_data=args.n_data_meas, n_rand_factor=args.n_rand_meas,
                                       seed=0, return_counts=True)
    xi_in, ic = deconvolve_window(xi_w, cnt["rr"])
    cov_ln, sigma2_ln = kernel_from_K2d(te, ze, xi_in, alpha=args.alpha)
    dt = None
    if args.transform == "lognormal":                 # Tier-A (#4): regularised non-Gaussian intensity
        dt = lgcp_density_transform(sigma2_ln, skew=args.skew, kind="lognormal")
        cov_k, sigma2 = kernel_from_K2d_tierA(te, ze, xi_in, dt, alpha=args.alpha)
        print(f"[cov] Tier-A transform: skew={args.skew} σ_g={dt.sigma_g:.3f} δ0={dt.delta0:.2f} "
              f"var(T)={dt.var_opd:.1f}; lognormal σ²={sigma2_ln:.3f} → Tier-A kernel diag={sigma2:.3f}",
              flush=True)
    else:
        cov_k, sigma2 = cov_ln, sigma2_ln
    t["kernel"] = time.perf_counter() - _t
    print(f"[cov] K_in IC={ic:.4f}  sigma2={sigma2:.3f}  [kernel {t['kernel']:.1f}s]", flush=True)

    # ---- shared randoms for the clustering LS estimator (built once) -------------------
    print("[cov] building shared clustering randoms + RR cache ...", flush=True)
    _t = time.perf_counter()
    rng = np.random.default_rng(args.seed + 777)
    n_rand_clust = args.n_rand_meas_clust * cat.N_data
    ra_r, dec_r, z_r = make_random_from_selection_function(
        sel_map=cat.sel_map, n_random=n_rand_clust, z_data=np.asarray(cat.z_data),
        nside=cat.nside, rng=rng)
    w_r = fkp_for(z_r, cat.z_data, cat.w_fkp_data)
    # Exact mangle footprint mask (the observational layer, #2) applied consistently to randoms +
    # galaxies. The angular completeness is NOT applied as a weight: both galaxies and randoms are
    # drawn from sel_map (∝ completeness), so it cancels in Landy-Szalay -- re-weighting double-counts.
    from echoes.boss_mock_columns import load_mangle_completeness
    mask = None if args.no_mask else load_mangle_completeness()
    if mask is not None:
        kr = mask.inside(ra_r, dec_r)
        print(f"[cov] exact mangle mask: randoms kept {kr.sum():,}/{len(kr):,} ({kr.mean():.4f})", flush=True)
        ra_r, dec_r, z_r, w_r = ra_r[kr], dec_r[kr], z_r[kr], w_r[kr]
    elif not args.no_mask:
        print("[cov] WARNING: pymangle unavailable -- no mangle mask applied (run pipeline/rebuild_pymangle.sh)", flush=True)
    cz_r = cz_of_z(z_r, cat.fid_cosmo)
    rr_cache = None
    if clm.has_corrfunc():
        rr_cache = clm.build_random_pairs(ra_r, dec_r, cz_r, w_r, rp_edges=rp_edges,
                                          pimax=pimax, npibins=npibins, s_edges=s_edges,
                                          nmu=nmu, nthreads=args.nthreads)
    t["randoms"] = time.perf_counter() - _t
    print(f"[cov] {n_rand_clust:,} clustering randoms, RR cached [{t['randoms']:.1f}s]", flush=True)

    # ---- generate ensemble (graph build amortized across N) ----------------------------
    print(f"[cov] generating ensemble of {args.n_realizations} ({args.device}) ...", flush=True)
    _t = time.perf_counter()
    n_builds = 0

    def _gen(n, seed):
        nonlocal n_builds
        n_builds += 1
        return generate_catalogs_from_kernel(
            cat, cov_k, sigma2, alpha=args.alpha, n_samples=n, seed=seed,
            w_completeness=w_comp, n_cand_factor=args.n_cand_factor, sampling=args.sampling,
            transform=dt, backend="julia", device=args.device, verbose=False)

    cats = []
    if args.batch and args.batch < args.n_realizations:
        done = 0; b = 0
        while done < args.n_realizations:
            nb = min(args.batch, args.n_realizations - done)
            cats.extend(_gen(nb, args.seed + 1 + 1000 * b))   # distinct seed block per batch
            done += nb; b += 1
            print(f"[cov]   batch {b}: {done}/{args.n_realizations} (graph rebuilt this batch)",
                  flush=True)
    else:
        cats = _gen(args.n_realizations, args.seed)
    t["generate"] = time.perf_counter() - _t
    n_gal = [c["N_galaxies"] for c in cats]
    print(f"[cov] generated {len(cats)} catalogs  graph_builds={n_builds}  "
          f"N_gal mean={np.mean(n_gal):,.0f} (min={min(n_gal):,} max={max(n_gal):,})  "
          f"[generate {t['generate']:.1f}s = {t['generate']/max(len(cats),1):.2f}s/real]", flush=True)

    # ---- per-realization clustering ----------------------------------------------------
    print("[cov] measuring wp, xi0, xi2 per realization ...", flush=True)
    _t = time.perf_counter()
    WP, XI0, XI2 = [], [], []
    for i, c in enumerate(cats):
        ra_d = np.asarray(c["ra"], np.float64); dec_d = np.asarray(c["dec"], np.float64)
        z_d = np.asarray(c["z"], np.float64)
        if mask is not None:                              # same exact footprint as the randoms
            kd = mask.inside(ra_d, dec_d)
            ra_d, dec_d, z_d = ra_d[kd], dec_d[kd], z_d[kd]
        w_d = fkp_for(z_d, cat.z_data, cat.w_fkp_data) if args.use_mock_weights \
            else np.ones(len(ra_d))
        wp, xi0, xi2 = measure_realization(ra_d, dec_d, z_d, w_d, ra_r, dec_r, cz_r, w_r,
                                           cat.fid_cosmo, rp_edges, s_edges, pimax, npibins,
                                           nmu, args.nthreads, rr_cache)
        WP.append(wp); XI0.append(xi0); XI2.append(xi2)
        if (i + 1) % max(1, len(cats) // 10) == 0:
            print(f"[cov]   {i+1}/{len(cats)} measured", flush=True)
    WP = np.asarray(WP); XI0 = np.asarray(XI0); XI2 = np.asarray(XI2)
    t["measure"] = time.perf_counter() - _t
    print(f"[cov] measured {len(cats)} realizations [{t['measure']:.1f}s]", flush=True)

    # ---- covariances + Hartlap ---------------------------------------------------------
    stats = {"wp": (WP, rp_c), "xi0": (XI0, s_c), "xi2": (XI2, s_c)}
    cov_out = {}
    print("\n[cov] === sample covariance (mock ensemble) ===", flush=True)
    for name, (A, x) in stats.items():
        C = sample_cov(A)
        nb = A.shape[1]
        h = hartlap_factor(args.n_realizations, nb)
        diag = np.diag(C)
        cov_out[name] = {"mean": A.mean(0), "cov": C, "diag": diag, "hartlap": h, "x": x}
        ok_h = "OK" if h > 0 else "INVALID (need N > n_bins+2)"
        print(f"  {name:>3}: cov shape {C.shape}  n_bins={nb}  Hartlap h={h:.3f} [{ok_h}]  "
              f"mean diag-err={np.sqrt(np.mean(diag)):.3e}", flush=True)

    # ---- convergence diagnostic --------------------------------------------------------
    print("\n[cov] === convergence (sqrt(mean diag) and a representative cov element vs N) ===",
          flush=True)
    conv_out = {}
    for name, (A, _) in stats.items():
        grid, mde, repcov, (b1, b2) = convergence_diag(A)
        conv_out[name] = {"N_grid": grid, "mean_diag_err": mde, "rep_cov": repcov,
                          "rep_bins": (b1, b2)}
        rel = (mde[-1] - mde[len(mde) // 2]) / (mde[-1] + 1e-30)
        print(f"  {name:>3}: N={grid.tolist()}", flush=True)
        print(f"       sqrt(mean diag)={np.array2string(mde, precision=3, max_line_width=200)}",
              flush=True)
        print(f"       rep cov[{b1},{b2}]={np.array2string(repcov, precision=3, max_line_width=200)}"
              f"   (frac change last-half->end = {rel:+.2%})", flush=True)

    # ---- data jackknife cross-check ----------------------------------------------------
    print(f"\n[cov] data jackknife ({args.n_jk} angular regions) ...", flush=True)
    _t = time.perf_counter()
    jk, n_jk_used = jackknife_data(cat, rp_edges, s_edges, pimax, npibins, nmu,
                                   args.n_jk, args.nthreads, seed=args.seed)
    t["jackknife"] = time.perf_counter() - _t
    print(f"[cov] jackknife used {n_jk_used} regions [{t['jackknife']:.1f}s]", flush=True)

    print("\n[cov] === JACKKNIFE (data) vs MOCK diagonal sigma  [key validation] ===", flush=True)
    print("       ratio sigma_jk/sigma_mock per bin; agreement to ~tens of percent = trustworthy",
          flush=True)
    jk_cmp = {}
    for name, (A, x) in stats.items():
        sig_mock = np.sqrt(np.clip(np.diag(sample_cov(A)), 0, None))
        sig_jk = np.sqrt(np.clip(jk[name]["diag"], 0, None))
        ratio = np.where(sig_mock > 0, sig_jk / sig_mock, np.nan)
        med = float(np.nanmedian(ratio))
        jk_cmp[name] = {"sig_mock": sig_mock, "sig_jk": sig_jk, "ratio": ratio, "median_ratio": med}
        print(f"  {name:>3} (n_bins={len(x)}):  median ratio={med:.2f}  "
              f"[range {np.nanmin(ratio):.2f}-{np.nanmax(ratio):.2f}]", flush=True)
        with np.printoptions(precision=3, suppress=True, linewidth=200):
            print(f"        bin x      ={x}")
            print(f"        sigma_mock ={sig_mock}")
            print(f"        sigma_jk   ={sig_jk}")
            print(f"        ratio      ={ratio}")

    # ---- save --------------------------------------------------------------------------
    # tag carries the candidate-set count (= graph builds): batch200/N1000 has 5 sets, the
    # candidate-independence rerun (#2) has many — distinct files so the comparison is preserved.
    xfm = "lognorm" if args.transform == "none" else f"tierA-sk{args.skew:g}"
    tag = f"{args.sampling}_{xfm}_N{args.n_realizations}_cf{args.n_cand_factor}_sets{n_builds}"
    meas_path = os.path.join(OUT, f"measurements_{tag}.npz")
    np.savez_compressed(
        meas_path,
        rp=rp_c, s=s_c, WP=WP, XI0=XI0, XI2=XI2,
        n_realizations=args.n_realizations, n_cand_factor=args.n_cand_factor,
        sigma2=sigma2, N_gal=np.asarray(n_gal),
        rp_edges=rp_edges, s_edges=s_edges, pimax=pimax)
    cov_path = os.path.join(OUT, f"covariance_{tag}.npz")
    np.savez_compressed(
        cov_path,
        rp=rp_c, s=s_c,
        cov_wp=cov_out["wp"]["cov"], cov_xi0=cov_out["xi0"]["cov"], cov_xi2=cov_out["xi2"]["cov"],
        mean_wp=cov_out["wp"]["mean"], mean_xi0=cov_out["xi0"]["mean"], mean_xi2=cov_out["xi2"]["mean"],
        hartlap_wp=cov_out["wp"]["hartlap"], hartlap_xi0=cov_out["xi0"]["hartlap"],
        hartlap_xi2=cov_out["xi2"]["hartlap"],
        jk_cov_wp=jk["wp"]["cov"], jk_cov_xi0=jk["xi0"]["cov"], jk_cov_xi2=jk["xi2"]["cov"],
        jk_mean_wp=jk["wp"]["mean"], jk_mean_xi0=jk["xi0"]["mean"], jk_mean_xi2=jk["xi2"]["mean"],
        n_jk=n_jk_used,
        ratio_wp=jk_cmp["wp"]["ratio"], ratio_xi0=jk_cmp["xi0"]["ratio"], ratio_xi2=jk_cmp["xi2"]["ratio"])

    total = time.perf_counter() - T0
    print(f"\n[cov] wrote:\n   {os.path.relpath(meas_path)}\n   {os.path.relpath(cov_path)}", flush=True)
    print(f"[cov] TIMING  load={t['load']:.1f}  kernel={t['kernel']:.1f}  randoms={t['randoms']:.1f}  "
          f"generate={t['generate']:.1f}  measure={t['measure']:.1f}  jackknife={t['jackknife']:.1f}  "
          f"TOTAL={total:.1f}s", flush=True)
    print(f"[cov] AMORTIZATION: graph_builds={n_builds} for N={args.n_realizations} realizations "
          f"({'single-call, build amortized' if n_builds == 1 else f'{n_builds} builds (batched)'})",
          flush=True)


if __name__ == "__main__":
    main()
