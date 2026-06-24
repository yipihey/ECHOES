"""Diagnostic for covariance next-step #1: find the density-matching ``n_cand_factor``.

The N=1000 covariance is over-dispersed at small rp because the σ²≈4.1 lognormal's heavy tail
drives the per-candidate Poisson rate above 1 at field peaks → multi-occupancy (multi_frac≈0.20 at
cf=20), an unphysical Δθ→0 spike. The honest fix (NOT Bernoulli, which under-clusters globally) is to
oversample the field with more candidates so p<1 even at peaks. This measures multi_frac(cf) so we can
pick the cf where it is acceptably small (target ≲0.02), and reports the ZIP32 max batch at each cf
(the generate output (batch,n_cand) must stay <4.29 GB/entry; f32 output — the run_graphgp.jl change —
doubles the ceiling vs f64).

  JAX_PLATFORMS=cpu OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 \
    python pipeline/diag_multifrac_vs_cf.py --cfs 20 40 80 160 --n-samples 2 --device gpu
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
                               generate_catalogs_from_kernel)

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"

ZIP32 = 4.0e9   # safe per-entry ceiling (UInt32 overflow at 4.294e9; leave header margin)


def zip32_max_batch(n_cand, bytes_per_elem):
    return int(ZIP32 // (n_cand * bytes_per_elem))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cfs", type=int, nargs="+", default=[20, 40, 80, 160])
    ap.add_argument("--n-samples", type=int, default=2)
    ap.add_argument("--n-data-meas", type=int, default=60_000)
    ap.add_argument("--n-rand-meas", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    print("[diag] loading CMASS-South ...", flush=True)
    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256)
    w_comp = cat.w_sys_data * cat.w_noz_data * cat.w_cp_data
    print(f"[diag] N_data={cat.N_data:,}", flush=True)

    # kernel is independent of cf — build once (same flow as build_boss_covariance.py)
    te = np.concatenate([[0.0], np.geomspace(0.02, 2.5, 16)])
    ze = np.linspace(0.0, 0.03, 11)
    print("[diag] measuring K_in (window-deconvolved LS) ...", flush=True)
    _, _, xi_w, cnt = measure_K2d_data(cat, theta_edges=te, z_edges=ze,
                                       n_data=args.n_data_meas, n_rand_factor=args.n_rand_meas,
                                       seed=0, return_counts=True)
    xi_in, ic = deconvolve_window(xi_w, cnt["rr"])
    cov_k, sigma2 = kernel_from_K2d(te, ze, xi_in, alpha=args.alpha)
    sig = np.sqrt(max(sigma2, 1e-12))
    print(f"[diag] IC={ic:.4f}  sigma2={sigma2:.3f}  sigma={sig:.3f}\n", flush=True)

    rows = []
    for cf in args.cfs:
        n_cand = cf * cat.N_data
        _t = time.perf_counter()
        out = generate_catalogs_from_kernel(
            cat, cov_k, sigma2, alpha=args.alpha, n_samples=args.n_samples, seed=args.seed,
            w_completeness=w_comp, n_cand_factor=cf, sampling="poisson",
            backend="julia", device=args.device, verbose=False)
        dt = time.perf_counter() - _t
        mf = np.array([c["multi_frac"] for c in out])
        ng = np.array([c["N_galaxies"] for c in out])
        b32, b64 = zip32_max_batch(n_cand, 4), zip32_max_batch(n_cand, 8)
        rows.append((cf, n_cand, mf.mean(), ng.mean(), b32, b64, dt))
        print(f"[diag] cf={cf:>3}  n_cand={n_cand:>12,}  multi_frac={mf.mean():.4f}"
              f"  N_gal={ng.mean():>9,.0f}  maxbatch[f32={b32:>4} f64={b64:>4}]  [{dt:.1f}s]",
              flush=True)

    print("\n[diag] === multi_frac vs n_cand_factor ===", flush=True)
    print(f"  {'cf':>4} {'n_cand':>13} {'multi_frac':>11} {'N_gal':>10} "
          f"{'maxbatch_f32':>13} {'maxbatch_f64':>13}", flush=True)
    for cf, nc, mf, ng, b32, b64, _dt in rows:
        print(f"  {cf:>4} {nc:>13,} {mf:>11.4f} {ng:>10,.0f} {b32:>13} {b64:>13}", flush=True)
    print("\n[diag] target multi_frac ≲ 0.02 (small-scale over-dispersion ~vanishes there).", flush=True)
    print("[diag] choose the smallest cf meeting target, with maxbatch_f32 ≥ desired --batch.",
          flush=True)


if __name__ == "__main__":
    main()
