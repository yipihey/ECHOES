"""The ACTUAL ECHOES covariance — the CONDITIONAL completion-posterior covariance.

NOT a mock ensemble. ECHOES realizations are realizations of *this* universe: every one holds the
securely-observed galaxies fixed and IDENTICAL, and differs only where the observation process left
freedom — fiber-collided / redshift-failure / imaging-systematic additions, their local Δz, and (with
count='poisson') the missing-number shot noise. So the realizations are highly correlated by
construction and the covariance of their clustering is the **observation-model-dependent** range of
wp/ξ given what we know for sure — cosmology-free, and small by construction.

This is the opposite object from a survey covariance (Patchy = scatter across *independent* universes,
cosmology+bias dependent). We report the conditional covariance and, for scale, its ratio to the
unconditional Patchy covariance — expected ≪ 1: using Patchy for the ECHOES posterior would massively
OVER-state the uncertainty, just as the lognormal mock ensemble does. (See DIAGNOSTICS.md / COVARIANCE.md.)

  JAX_PLATFORMS=cpu OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 \
    python pipeline/build_conditional_covariance.py --n-real 100 \
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
from echoes.completion import complete_catalog, measure_close_pair_dz, fkp_weight_of_z
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
    ap.add_argument("--n-real", type=int, default=100, help="conditional completion realizations")
    ap.add_argument("--count", choices=["poisson", "round"], default="poisson",
                    help="poisson = include missing-number shot noise (fuller posterior); round = "
                         "only the redshift/placement uncertainty of the additions")
    ap.add_argument("--z-assign", choices=["data", "host", "nz", "mix"], default="data")
    ap.add_argument("--nthreads", type=int, default=16)
    ap.add_argument("--no-mask", action="store_true")
    ap.add_argument("--rp-min", type=float, default=0.5); ap.add_argument("--rp-max", type=float, default=40.0)
    ap.add_argument("--n-rp", type=int, default=12); ap.add_argument("--pimax", type=float, default=80.0)
    ap.add_argument("--s-min", type=float, default=1.0); ap.add_argument("--s-max", type=float, default=40.0)
    ap.add_argument("--n-s", type=int, default=14); ap.add_argument("--nmu", type=int, default=100)
    ap.add_argument("--n-rand-meas-clust", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--compare", type=str, default="covariance_patchy_N600.npz")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    rp_edges = np.logspace(np.log10(args.rp_min), np.log10(args.rp_max), args.n_rp + 1)
    s_edges = np.linspace(args.s_min, args.s_max, args.n_s + 1)
    pimax = args.pimax; npibins = int(round(pimax)); nmu = args.nmu
    rp_c = 0.5 * (rp_edges[:-1] + rp_edges[1:]); s_c = 0.5 * (s_edges[:-1] + s_edges[1:])
    T0 = time.perf_counter(); t = {}

    print("[cond] loading CMASS-South (the fixed observed universe) ...", flush=True)
    _t = time.perf_counter()
    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256)
    dz_pool = measure_close_pair_dz(cat)                        # measured once; shared
    mask = None
    if not args.no_mask:
        from echoes.boss_mock_columns import load_mangle_completeness
        mask = load_mangle_completeness()
    t["load"] = time.perf_counter() - _t
    print(f"[cond] N_obs={cat.N_data:,}  [load {t['load']:.1f}s]", flush=True)

    # shared randoms + RR cache (FKP-weighted randoms, equal-weight data — same estimator as the
    # unconditional run so the covariances are directly comparable)
    print("[cond] building shared randoms + RR cache ...", flush=True)
    _t = time.perf_counter()
    rng = np.random.default_rng(args.seed + 777)
    ra_r, dec_r, z_r = make_random_from_selection_function(
        sel_map=cat.sel_map, n_random=args.n_rand_meas_clust * cat.N_data,
        z_data=np.asarray(cat.z_data), nside=cat.nside, rng=rng)
    w_r = fkp_weight_of_z(z_r, np.asarray(cat.z_data), cat.w_fkp_data) \
        if cat.w_fkp_data is not None else np.ones(len(ra_r))
    if mask is not None:
        kr = mask.inside(ra_r, dec_r); ra_r, dec_r, z_r, w_r = ra_r[kr], dec_r[kr], z_r[kr], w_r[kr]
    cz_r = cz_of_z(z_r, cat.fid_cosmo)
    rr_cache = clm.build_random_pairs(ra_r, dec_r, cz_r, w_r, rp_edges=rp_edges, pimax=pimax,
                                      npibins=npibins, s_edges=s_edges, nmu=nmu, nthreads=args.nthreads)
    t["randoms"] = time.perf_counter() - _t
    print(f"[cond] {len(ra_r):,} randoms, RR cached [{t['randoms']:.1f}s]", flush=True)

    # conditional completion ensemble: SAME observed galaxies, varying only the inpainted parts
    print(f"[cond] {args.n_real} conditional realizations (count={args.count}, z={args.z_assign}) ...",
          flush=True)
    _t = time.perf_counter()
    WP, XI0, XI2, NADD = [], [], [], []
    for s in range(args.n_real):
        c = complete_catalog(cat, seed=s, count=args.count, z_assign=args.z_assign, dz_pool=dz_pool)
        ra_d = np.asarray(c["ra"], np.float64); dec_d = np.asarray(c["dec"], np.float64)
        z_d = np.asarray(c["z"], np.float64)
        if mask is not None:
            kd = mask.inside(ra_d, dec_d); ra_d, dec_d, z_d = ra_d[kd], dec_d[kd], z_d[kd]
        cz_d = cz_of_z(z_d, cat.fid_cosmo); w_d = np.ones(len(ra_d))
        _, wp = clm.measure_wp(ra_d, dec_d, cz_d, w_d, ra_r, dec_r, cz_r, w_r, rp_edges=rp_edges,
                               pimax=pimax, npibins=npibins, nthreads=args.nthreads, rr=rr_cache)
        _, xi0, xi2 = clm.measure_xi_ell(ra_d, dec_d, cz_d, w_d, ra_r, dec_r, cz_r, w_r,
                                         s_edges=s_edges, nmu=nmu, nthreads=args.nthreads, rr=rr_cache)
        WP.append(wp); XI0.append(xi0); XI2.append(xi2); NADD.append(len(ra_d) - cat.N_data)
        if (s + 1) % max(1, args.n_real // 10) == 0:
            print(f"[cond]   {s+1}/{args.n_real}  (N_added~{NADD[-1]:,}, {100*NADD[-1]/cat.N_data:.1f}%)",
                  flush=True)
    WP = np.asarray(WP); XI0 = np.asarray(XI0); XI2 = np.asarray(XI2)
    t["measure"] = time.perf_counter() - _t
    print(f"[cond] measured {args.n_real} conditional realizations [{t['measure']:.1f}s]", flush=True)

    stats = {"wp": (WP, rp_c), "xi0": (XI0, s_c), "xi2": (XI2, s_c)}
    cov_out = {}
    print("\n[cond] === conditional posterior covariance (fractional spread = σ/mean) ===", flush=True)
    for name, (A, x) in stats.items():
        C = sample_cov(A); diag = np.diag(C); mean = A.mean(0)
        frac = np.sqrt(np.clip(diag, 0, None)) / np.where(np.abs(mean) > 0, np.abs(mean), np.nan)
        cov_out[name] = {"mean": mean, "cov": C, "diag": diag, "x": x, "frac": frac}
        print(f"  {name:>3}: median fractional σ = {np.nanmedian(frac):.4f}  "
              f"[{np.nanmin(frac):.4f}-{np.nanmax(frac):.4f}]", flush=True)

    # scale vs the UNCONDITIONAL Patchy covariance — the whole point: conditional ≪ unconditional
    cmp_path = args.compare if os.path.isabs(args.compare) or os.path.exists(args.compare) \
        else os.path.join(OUT, args.compare)
    if os.path.exists(cmp_path):
        U = np.load(cmp_path)
        print(f"\n[cond] === σ_conditional / σ_Patchy  (unconditional cosmic variance, {os.path.basename(cmp_path)}) ===",
              flush=True)
        print("       ≪ 1 expected: the ECHOES posterior is the observation-model range, NOT cosmic variance",
              flush=True)
        for name in ("wp", "xi0", "xi2"):
            sc = np.sqrt(np.clip(cov_out[name]["diag"], 0, None))
            su = np.sqrt(np.clip(np.diag(U[f"cov_{name}"]), 0, None))
            r = np.where(su > 0, sc / su, np.nan)
            print(f"  {name:>3}: median σ_cond/σ_Patchy = {np.nanmedian(r):.3f}  "
                  f"[{np.nanmin(r):.3f}-{np.nanmax(r):.3f}]", flush=True)

    tag = f"conditional_N{args.n_real}_{args.count}_{args.z_assign}"
    np.savez_compressed(os.path.join(OUT, f"covariance_{tag}.npz"),
                        rp=rp_c, s=s_c, cov_wp=cov_out["wp"]["cov"], cov_xi0=cov_out["xi0"]["cov"],
                        cov_xi2=cov_out["xi2"]["cov"], mean_wp=cov_out["wp"]["mean"],
                        mean_xi0=cov_out["xi0"]["mean"], mean_xi2=cov_out["xi2"]["mean"],
                        WP=WP, XI0=XI0, XI2=XI2, N_added=np.asarray(NADD), n_real=args.n_real,
                        rp_edges=rp_edges, s_edges=s_edges)
    total = time.perf_counter() - T0
    print(f"\n[cond] wrote covariance_{tag}.npz", flush=True)
    print(f"[cond] TIMING load={t['load']:.1f} randoms={t['randoms']:.1f} measure={t['measure']:.1f} "
          f"TOTAL={total:.1f}s", flush=True)


if __name__ == "__main__":
    main()
