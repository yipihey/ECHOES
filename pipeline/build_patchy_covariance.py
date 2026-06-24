"""Covariance next-step #3 — the EXTERNAL ground-truth clustering covariance from the
MultiDark-Patchy DR12 SGC mocks, measured with the SAME estimator + binning as the LGCP
ensemble (pipeline/build_boss_covariance.py) so the two covariances diff bin-for-bin.

The data jackknife structurally mis-estimates the multipole variance (over-estimates the
monopole at large scales, under-estimates the quadrupole), so it cannot by itself validate
the LGCP ξ₀/ξ₂ covariance. Patchy is the accepted external reference: ~2048 approximate
CMASS mocks with the real survey geometry and RSD. This measures wp/ξ₀/ξ₂ on N Patchy mocks
(shared randoms + RR cache, exact mangle footprint, FKP weights) and writes the sample
covariance in the build_boss_covariance schema; if an LGCP covariance file is present it
prints the per-bin σ_patchy/σ_lgcp ratio — the trust check the jackknife can't give.

  JAX_PLATFORMS=cpu OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 \
    python pipeline/build_patchy_covariance.py --n-mocks 200 --n-rand 1500000

Patchy COMPSAM columns (V6C): galaxy = RA Dec z Mstar nbar bias veto fiber ;
random = RA Dec z nbar (bias) veto fiber. CMASS cut 0.43<z<0.7; veto flag (==1) kept;
w_fkp = 1/(1+nbar·P0), P0=1e4 (matching the data convention used in load_boss).
"""
import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from echoes.surveys.boss import load_boss
from echoes.distance import comoving_distance
from echoes import clustering_measure as clm

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"
MOCKS = "data/boss/mocks"
GLOB = "Patchy-Mocks-DR12SGC-COMPSAM_V6C_*.dat"
PRAND = "Patchy-Mocks-Randoms-DR12SGC-COMPSAM_V6C_x10.dat"
OUT = os.path.join(os.path.dirname(__file__), "..", "data_release", "boss_lgcp_julia", "covariance")
P0 = 1.0e4


def _read_table(path, ncols_min):
    """Fast whitespace table read (pandas if available, else numpy)."""
    try:
        import pandas as pd
        df = pd.read_csv(path, sep=r"\s+", header=None, engine="c", dtype=np.float64)
        return df.to_numpy()
    except Exception:
        return np.loadtxt(path)


def _load_patchy_gal(path, zmin, zmax):
    a = _read_table(path, 8)
    ra, dec, z, nbar = a[:, 0], a[:, 1], a[:, 2], a[:, 4]
    veto = a[:, -2]                                    # veto flag (1 = keep)
    keep = (z > zmin) & (z < zmax) & (veto > 0)
    ra, dec, z, nbar = ra[keep], dec[keep], z[keep], nbar[keep]
    wfkp = 1.0 / (1.0 + nbar * P0)
    return ra, dec, z, wfkp


def _load_patchy_rand(path, zmin, zmax, n_keep, rng):
    a = _read_table(path, 6)
    ra, dec, z, nbar = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    veto = a[:, -2]
    keep = (z > zmin) & (z < zmax) & (veto > 0)
    ra, dec, z, nbar = ra[keep], dec[keep], z[keep], nbar[keep]
    if n_keep and n_keep < len(ra):
        sel = rng.choice(len(ra), n_keep, replace=False)
        ra, dec, z, nbar = ra[sel], dec[sel], z[sel], nbar[sel]
    wfkp = 1.0 / (1.0 + nbar * P0)
    return ra, dec, z, wfkp


def cz_of_z(z, cosmo):
    import jax
    jax.config.update("jax_enable_x64", True)
    return np.asarray(comoving_distance(np.asarray(z, np.float64), cosmo))


def sample_cov(A):
    A = np.asarray(A, np.float64)
    n = A.shape[0]
    d = A - A.mean(0)
    return (d.T @ d) / (n - 1)


def hartlap_factor(n_real, n_bins):
    return (n_real - n_bins - 2.0) / (n_real - 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-mocks", type=int, default=200)
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("--mocks-dir", type=str, default=MOCKS,
                    help="dir of Patchy *.dat mocks (default data/boss/mocks; only 10 extracted there "
                         "— point at the nvme scratch where the full 2048 are unpacked)")
    # MATCH load_boss's CMASS-South slice (z in 0.45-0.60, N≈110k) — NOT the full CMASS 0.43-0.70,
    # so the Patchy density/n(z) matches the LGCP mocks for a fair covariance comparison.
    ap.add_argument("--zmin", type=float, default=0.45)
    ap.add_argument("--zmax", type=float, default=0.60)
    ap.add_argument("--n-rand", type=int, default=1_500_000,
                    help="downsample the x10 Patchy randoms to this many (RR cost; LS unbiased)")
    ap.add_argument("--match-n", type=int, default=109_636,
                    help="uniformly downsample each mock to this many galaxies to match the data's "
                         "shot noise (COMPSAM is the complete sample, ~1.5x denser; n(z) already "
                         "matches CMASS so uniform subsampling preserves it). 0 = keep all.")
    ap.add_argument("--nthreads", type=int, default=16)
    ap.add_argument("--no-mask", action="store_true", help="skip the exact mangle footprint mask")
    # binning — DEFAULTS MUST MATCH build_boss_covariance.py for a bin-for-bin comparison
    ap.add_argument("--rp-min", type=float, default=0.5)
    ap.add_argument("--rp-max", type=float, default=40.0)
    ap.add_argument("--n-rp", type=int, default=12)
    ap.add_argument("--pimax", type=float, default=80.0)
    ap.add_argument("--s-min", type=float, default=1.0)
    ap.add_argument("--s-max", type=float, default=40.0)
    ap.add_argument("--n-s", type=int, default=14)
    ap.add_argument("--nmu", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--compare", type=str, default="",
                    help="path to an LGCP covariance_*.npz to print sigma_patchy/sigma_lgcp ratios")
    args = ap.parse_args()

    if not clm.has_corrfunc():
        print("[patchy] WARNING: Corrfunc not importable — cKDTree fallback is too slow here.", flush=True)

    os.makedirs(OUT, exist_ok=True)
    rp_edges = np.logspace(np.log10(args.rp_min), np.log10(args.rp_max), args.n_rp + 1)
    s_edges = np.linspace(args.s_min, args.s_max, args.n_s + 1)
    pimax = args.pimax
    npibins = int(round(pimax))
    nmu = args.nmu
    rp_c = 0.5 * (rp_edges[:-1] + rp_edges[1:])
    s_c = 0.5 * (s_edges[:-1] + s_edges[1:])

    T0 = time.perf_counter(); t = {}

    # fiducial cosmology (identical z->cz as the LGCP measurement) via load_boss
    print("[patchy] loading CMASS-South (fiducial cosmology + mask) ...", flush=True)
    _t = time.perf_counter()
    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256)
    cosmo = cat.fid_cosmo
    mask = None
    if not args.no_mask:
        from echoes.boss_mock_columns import load_mangle_completeness
        mask = load_mangle_completeness()
        if mask is None:
            print("[patchy] WARNING: pymangle unavailable — no mangle mask", flush=True)
    t["load"] = time.perf_counter() - _t

    # shared randoms + RR cache (built once; the dominant cost)
    print(f"[patchy] loading + downsampling randoms to {args.n_rand:,} ...", flush=True)
    _t = time.perf_counter()
    rng = np.random.default_rng(args.seed + 777)
    ra_r, dec_r, z_r, w_r = _load_patchy_rand(os.path.join(MOCKS, PRAND), args.zmin, args.zmax,
                                              args.n_rand, rng)
    if mask is not None:
        kr = mask.inside(ra_r, dec_r)
        print(f"[patchy] mangle mask: randoms kept {kr.sum():,}/{len(kr):,} ({kr.mean():.4f})", flush=True)
        ra_r, dec_r, z_r, w_r = ra_r[kr], dec_r[kr], z_r[kr], w_r[kr]
    cz_r = cz_of_z(z_r, cosmo)
    rr_cache = clm.build_random_pairs(ra_r, dec_r, cz_r, w_r, rp_edges=rp_edges, pimax=pimax,
                                      npibins=npibins, s_edges=s_edges, nmu=nmu,
                                      nthreads=args.nthreads) if clm.has_corrfunc() else None
    t["randoms"] = time.perf_counter() - _t
    print(f"[patchy] {len(ra_r):,} randoms, RR cached [{t['randoms']:.1f}s]", flush=True)

    # per-mock measurement
    files = sorted(glob.glob(os.path.join(args.mocks_dir, GLOB)))
    files = files[args.first - 1: args.first - 1 + args.n_mocks]
    print(f"[patchy] measuring {len(files)} mocks ...", flush=True)
    _t = time.perf_counter()
    WP, XI0, XI2, NG = [], [], [], []
    for i, f in enumerate(files):
        ra_d, dec_d, z_d, w_d = _load_patchy_gal(f, args.zmin, args.zmax)
        if mask is not None:
            kd = mask.inside(ra_d, dec_d)
            ra_d, dec_d, z_d, w_d = ra_d[kd], dec_d[kd], z_d[kd], w_d[kd]
        if args.match_n and len(ra_d) > args.match_n:        # shot-noise match to the data N
            rngm = np.random.default_rng(args.seed + 1000 + i)
            sel = rngm.choice(len(ra_d), args.match_n, replace=False)
            ra_d, dec_d, z_d, w_d = ra_d[sel], dec_d[sel], z_d[sel], w_d[sel]
        cz_d = cz_of_z(z_d, cosmo)
        _, wp = clm.measure_wp(ra_d, dec_d, cz_d, w_d, ra_r, dec_r, cz_r, w_r,
                               rp_edges=rp_edges, pimax=pimax, npibins=npibins,
                               nthreads=args.nthreads, rr=rr_cache)
        _, xi0, xi2 = clm.measure_xi_ell(ra_d, dec_d, cz_d, w_d, ra_r, dec_r, cz_r, w_r,
                                         s_edges=s_edges, nmu=nmu, nthreads=args.nthreads, rr=rr_cache)
        WP.append(wp); XI0.append(xi0); XI2.append(xi2); NG.append(len(ra_d))
        if (i + 1) % max(1, len(files) // 20) == 0:
            print(f"[patchy]   {i+1}/{len(files)} measured  (N_gal~{len(ra_d):,})", flush=True)
    WP = np.asarray(WP); XI0 = np.asarray(XI0); XI2 = np.asarray(XI2)
    t["measure"] = time.perf_counter() - _t
    n_real = len(files)
    print(f"[patchy] measured {n_real} mocks [{t['measure']:.1f}s]", flush=True)

    stats = {"wp": (WP, rp_c), "xi0": (XI0, s_c), "xi2": (XI2, s_c)}
    cov_out = {}
    print("\n[patchy] === sample covariance (Patchy ensemble) ===", flush=True)
    for name, (A, x) in stats.items():
        C = sample_cov(A); nb = A.shape[1]; h = hartlap_factor(n_real, nb); diag = np.diag(C)
        cov_out[name] = {"mean": A.mean(0), "cov": C, "diag": diag, "hartlap": h}
        print(f"  {name:>3}: cov {C.shape}  Hartlap h={h:.3f}  mean diag-err={np.sqrt(np.mean(diag)):.3e}",
              flush=True)

    tag = f"patchy_N{n_real}"
    cov_path = os.path.join(OUT, f"covariance_{tag}.npz")
    np.savez_compressed(cov_path, rp=rp_c, s=s_c,
                        cov_wp=cov_out["wp"]["cov"], cov_xi0=cov_out["xi0"]["cov"],
                        cov_xi2=cov_out["xi2"]["cov"], mean_wp=cov_out["wp"]["mean"],
                        mean_xi0=cov_out["xi0"]["mean"], mean_xi2=cov_out["xi2"]["mean"],
                        WP=WP, XI0=XI0, XI2=XI2, N_gal=np.asarray(NG), n_real=n_real,
                        rp_edges=rp_edges, s_edges=s_edges, pimax=pimax)

    # external trust check vs an LGCP covariance
    if args.compare and os.path.exists(args.compare):
        L = np.load(args.compare)
        print(f"\n[patchy] === sigma_patchy / sigma_lgcp  ({os.path.basename(args.compare)}) ===",
              flush=True)
        for name in ("wp", "xi0", "xi2"):
            sp = np.sqrt(np.clip(np.diag(cov_out[name]["cov"]), 0, None))
            sl = np.sqrt(np.clip(np.diag(L[f"cov_{name}"]), 0, None))
            ratio = np.where(sl > 0, sp / sl, np.nan)
            print(f"  {name:>3}: median sigma_patchy/sigma_lgcp = {np.nanmedian(ratio):.2f}  "
                  f"[range {np.nanmin(ratio):.2f}-{np.nanmax(ratio):.2f}]", flush=True)
            with np.printoptions(precision=3, suppress=True, linewidth=200):
                print(f"        ratio = {ratio}", flush=True)

    total = time.perf_counter() - T0
    print(f"\n[patchy] wrote {os.path.relpath(cov_path)}", flush=True)
    print(f"[patchy] TIMING load={t['load']:.1f} randoms={t['randoms']:.1f} "
          f"measure={t['measure']:.1f} TOTAL={total:.1f}s", flush=True)


if __name__ == "__main__":
    main()
