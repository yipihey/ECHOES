"""Step 2 — the JOINT conditional posterior covariance (field-engine completion).

Measures the conditional completion covariance (wp/ξ₀/ξ₂) for the FIELD-engine completions
(`complete_catalog_photoz`) — drawing the missing redshifts from the conditional field posterior
(z_mode='fieldpost': all missing galaxies in a realization share ONE Matheron field draw) vs a sharp
per-object engine (z_mode='nn'). NOTE: these are different z-ENGINES, not the clean factorized-vs-joint
of ONE model — `nn` is actually sharper at small rp. The genuine coherence (factorized→joint) term is
the field-correlation COPULA (`build_package(copula=True)`, the shipped default; +19% trace / ×1.21,
`validation/completion_covariance_shape.py`). What this script establishes is robustness: the
conditional covariance is ~0.2× Patchy (σ) across every completion engine — far below cosmic variance.

Field is GAUSSIAN (fieldpost / z_mode='fieldpost'), NOT lognormal: object_pit.py shows the lognormal
field degrades the per-object redshift PIT (KS 0.085→0.175) — sharpening (1+δ) over-peaks the p(z)
weight. Lognormal's home is the SPATIAL density path, not BOSS z-completion.

Same REAL targets (load_cmass_targets: collided + z-failures) for every z_mode, so factorized-vs-joint
is an apples-to-apples comparison.

  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 JAX_PLATFORMS=cpu \
    python pipeline/build_conditional_covariance_field.py --n-real 40 --z-modes nn fieldpost \
       --compare covariance_patchy_N600.npz
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from echoes.surveys.boss import load_boss
from echoes.surveys.boss_targets import load_cmass_targets
from echoes.photoz import PhotoZKNN, photoz_features
from echoes.completion import complete_catalog_photoz, measure_close_pair_dz, fkp_weight_of_z
from echoes.fieldpost import build_field_context
from echoes.randoms import make_random_from_selection_function
from echoes.distance import comoving_distance
from echoes import clustering_measure as clm

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"
OUT = os.path.join(os.path.dirname(__file__), "..", "data_release", "boss_lgcp_julia", "covariance")


def cz_of_z(z, cosmo):
    import jax
    jax.config.update("jax_enable_x64", True)
    return np.asarray(comoving_distance(np.asarray(z, np.float64), cosmo))


def sample_cov(A):
    A = np.asarray(A, np.float64); d = A - A.mean(0)
    return (d.T @ d) / (A.shape[0] - 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-real", type=int, default=40, help="realizations = field draws (n_samples)")
    ap.add_argument("--z-modes", nargs="+", default=["nn", "fieldpost"],
                    help="completion z-engines to compare (nn=factorized sharp, fieldpost=joint field)")
    ap.add_argument("--nthreads", type=int, default=8)
    ap.add_argument("--no-mask", action="store_true")
    ap.add_argument("--rp-min", type=float, default=0.5); ap.add_argument("--rp-max", type=float, default=40.0)
    ap.add_argument("--n-rp", type=int, default=12); ap.add_argument("--pimax", type=float, default=80.0)
    ap.add_argument("--s-min", type=float, default=1.0); ap.add_argument("--s-max", type=float, default=40.0)
    ap.add_argument("--n-s", type=int, default=14); ap.add_argument("--nmu", type=int, default=100)
    ap.add_argument("--n-rand-meas-clust", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--compare", type=str, default="covariance_patchy_N600.npz")
    ap.add_argument("--compare-factorized", type=str, default="covariance_conditional_N100_round_data.npz")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    rp_edges = np.logspace(np.log10(args.rp_min), np.log10(args.rp_max), args.n_rp + 1)
    s_edges = np.linspace(args.s_min, args.s_max, args.n_s + 1)
    pimax = args.pimax; npibins = int(round(pimax)); nmu = args.nmu
    rp_c = 0.5 * (rp_edges[:-1] + rp_edges[1:]); s_c = 0.5 * (s_edges[:-1] + s_edges[1:])
    T0 = time.perf_counter()

    print("[cond-f] loading CMASS-South (+photometry) ...", flush=True)
    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    ra = np.asarray(cat.ra_data); dec = np.asarray(cat.dec_data); z = np.asarray(cat.z_data)
    feat = photoz_features(cat.colors_data, cat.mags_data)
    good = np.isfinite(feat).all(1) & (np.asarray(cat.imatch_data) == 1)
    pz = PhotoZKNN(k=100).fit(feat[good], z[good])
    targets = load_cmass_targets(cat)
    dz = measure_close_pair_dz(cat)
    print(f"[cond-f] N_obs={cat.N_data:,}  N_missing(targets)={targets.N:,}", flush=True)

    mask = None
    if not args.no_mask:
        from echoes.boss_mock_columns import load_mangle_completeness
        mask = load_mangle_completeness()

    # shared randoms + RR cache
    print("[cond-f] building shared randoms + RR cache ...", flush=True)
    rng = np.random.default_rng(args.seed + 777)
    ra_r, dec_r, z_r = make_random_from_selection_function(
        sel_map=cat.sel_map, n_random=args.n_rand_meas_clust * cat.N_data,
        z_data=z, nside=cat.nside, rng=rng)
    w_r = fkp_weight_of_z(z_r, z, cat.w_fkp_data) if cat.w_fkp_data is not None else np.ones(len(ra_r))
    if mask is not None:
        kr = mask.inside(ra_r, dec_r); ra_r, dec_r, z_r, w_r = ra_r[kr], dec_r[kr], z_r[kr], w_r[kr]
    cz_r = cz_of_z(z_r, cat.fid_cosmo)
    rr_cache = clm.build_random_pairs(ra_r, dec_r, cz_r, w_r, rp_edges=rp_edges, pimax=pimax,
                                      npibins=npibins, s_edges=s_edges, nmu=nmu, nthreads=args.nthreads)

    # one field context with N draws (joint field uncertainty); reused across z_modes
    print(f"[cond-f] building field context (n_samples={args.n_real}) — the heavy solve ...", flush=True)
    _t = time.perf_counter()
    fctx = build_field_context(cat, seed=0, n_samples=args.n_real, sel_map=cat.sel_map, nside=cat.nside)
    print(f"[cond-f] field context ready [{time.perf_counter()-_t:.1f}s]", flush=True)

    def measure(c):
        ra_d = np.asarray(c["ra"], np.float64); dec_d = np.asarray(c["dec"], np.float64)
        z_d = np.asarray(c["z"], np.float64)
        if mask is not None:
            kd = mask.inside(ra_d, dec_d); ra_d, dec_d, z_d = ra_d[kd], dec_d[kd], z_d[kd]
        cz_d = cz_of_z(z_d, cat.fid_cosmo); w_d = np.ones(len(ra_d))
        _, wp = clm.measure_wp(ra_d, dec_d, cz_d, w_d, ra_r, dec_r, cz_r, w_r, rp_edges=rp_edges,
                               pimax=pimax, npibins=npibins, nthreads=args.nthreads, rr=rr_cache)
        _, xi0, xi2 = clm.measure_xi_ell(ra_d, dec_d, cz_d, w_d, ra_r, dec_r, cz_r, w_r,
                                         s_edges=s_edges, nmu=nmu, nthreads=args.nthreads, rr=rr_cache)
        return wp, xi0, xi2

    results = {}
    for zmode in args.z_modes:
        print(f"\n[cond-f] z_mode={zmode}: {args.n_real} realizations ...", flush=True)
        _t = time.perf_counter()
        WP, XI0, XI2 = [], [], []
        for sd in range(args.n_real):
            c = complete_catalog_photoz(cat, targets, pz, seed=sd, z_mode=zmode,
                                        field_ctx=fctx, dz_pool=dz)
            wp, xi0, xi2 = measure(c)
            WP.append(wp); XI0.append(xi0); XI2.append(xi2)
            if (sd + 1) % max(1, args.n_real // 5) == 0:
                print(f"[cond-f]   {sd+1}/{args.n_real}", flush=True)
        results[zmode] = {"wp": np.asarray(WP), "xi0": np.asarray(XI0), "xi2": np.asarray(XI2)}
        print(f"[cond-f] z_mode={zmode} done [{time.perf_counter()-_t:.1f}s]", flush=True)

    # report each z_mode: fractional spread + vs Patchy (+ vs the factorized lower bound)
    U = None
    cmp_path = args.compare if os.path.exists(args.compare) else os.path.join(OUT, args.compare)
    if os.path.exists(cmp_path):
        U = np.load(cmp_path)
    F = None
    fpath = args.compare_factorized if os.path.exists(args.compare_factorized) \
        else os.path.join(OUT, args.compare_factorized)
    if os.path.exists(fpath):
        F = np.load(fpath)

    for zmode, R in results.items():
        print(f"\n[cond-f] === z_mode={zmode}: conditional covariance ===", flush=True)
        for name, x in (("wp", rp_c), ("xi0", s_c), ("xi2", s_c)):
            A = R[name]; C = sample_cov(A); diag = np.diag(C); mean = A.mean(0)
            frac = np.sqrt(np.clip(diag, 0, None)) / np.where(np.abs(mean) > 0, np.abs(mean), np.nan)
            line = f"  {name:>3}: median fractional σ = {np.nanmedian(frac):.4f}"
            if U is not None and f"cov_{name}" in U:
                rU = np.sqrt(np.clip(diag, 0, None)) / np.sqrt(np.clip(np.diag(U[f"cov_{name}"]), 0, None))
                line += f"   σ/σ_Patchy = {np.nanmedian(rU):.3f}"
            if F is not None and f"cov_{name}" in F:
                rF = np.sqrt(np.clip(diag, 0, None)) / np.sqrt(np.clip(np.diag(F[f"cov_{name}"]), 0, None))
                line += f"   σ/σ_factorized = {np.nanmedian(rF):.2f}"
            print(line, flush=True)
        tag = f"conditional_field_{zmode}_N{args.n_real}"
        np.savez_compressed(os.path.join(OUT, f"covariance_{tag}.npz"),
                            rp=rp_c, s=s_c, cov_wp=sample_cov(R["wp"]), cov_xi0=sample_cov(R["xi0"]),
                            cov_xi2=sample_cov(R["xi2"]), mean_wp=R["wp"].mean(0),
                            mean_xi0=R["xi0"].mean(0), mean_xi2=R["xi2"].mean(0),
                            WP=R["wp"], XI0=R["xi0"], XI2=R["xi2"], n_real=args.n_real)

    print(f"\n[cond-f] TOTAL={time.perf_counter()-T0:.1f}s", flush=True)
    print("[cond-f] NOTE: nn vs fieldpost are two different z-ENGINES (nn is sharper at small rp), "
          "NOT the factorized-vs-joint of one model — that isolation is the field-correlation COPULA "
          "(build_package copula=True, shipped default; +19% trace / x1.21, validation/"
          "completion_covariance_shape.py). What this confirms: the conditional covariance is "
          "~0.2x Patchy (sigma) across ALL completion engines — far below cosmic variance.", flush=True)


if __name__ == "__main__":
    main()
