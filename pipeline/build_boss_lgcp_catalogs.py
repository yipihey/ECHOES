"""BOSS CMASS-South LGCP catalogs generated on the JULIA GraphGP engine — same method as the JAX
path (anisotropic K(Δθ,Δz) from a window-deconvolved Landy–Szalay measurement), only the backend
swapped. The field is anisotropic in observed coordinates, so this needs the GraphGP.jl anisotropic
covariance port (validated in tests/test_graphgp_julia_aniso.py to 9e-15 vs the fork).

Pipeline (mirrors graphGP-cosmology/demos/validate_observed_K2d.py with echoes.completion):
  load_boss → measure_K2d_data → deconvolve_window → kernel_from_K2d (AnisotropicCovariance, σ²)
  → generate_catalogs_from_kernel(..., backend=...) → re-measure K_out and compare to K_in.

    OMP_NUM_THREADS=16 JAX_PLATFORMS=cpu python pipeline/build_boss_lgcp_catalogs.py \
        --n-data-meas 30000 --n-cand-factor 5 --n-samples 3 --backend julia
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from echoes.surveys.boss import load_boss
from echoes.completion import (measure_K2d_data, deconvolve_window, kernel_from_K2d,
                               generate_catalogs_from_kernel, measure_K2d)
from echoes.randoms import make_random_from_selection_function
from echoes.geometry import _radec_to_nhat
from echoes.completion import fkp_weight_of_z

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"
OUT = os.path.join(os.path.dirname(__file__), "..", "data_release", "boss_lgcp_julia")


def measure_Kout(cat, ra, dec, z, te, ze, n_rand_factor=3, seed=0):
    """Re-measure K(Δθ,Δz) of a generated (unweighted) catalogue with the same LS estimator."""
    rng = np.random.default_rng(seed)
    nr = n_rand_factor * len(ra)
    ra_r, dec_r, z_r = make_random_from_selection_function(
        sel_map=cat.sel_map, n_random=nr, z_data=np.asarray(cat.z_data), nside=cat.nside, rng=rng)
    w_r = fkp_weight_of_z(z_r, np.asarray(cat.z_data), cat.w_fkp_data) \
        if cat.w_fkp_data is not None else np.ones(len(ra_r))
    w_d = np.ones(len(ra))
    return measure_K2d(ra, dec, z, w_d, ra_r, dec_r, z_r, w_r,
                       theta_edges=te, z_edges=ze, return_counts=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-data-meas", type=int, default=30_000, help="data subsample for the K2d measurement")
    ap.add_argument("--n-rand-meas", type=int, default=3)
    ap.add_argument("--n-cand-factor", type=int, default=5, help="candidates = factor x N_data")
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--backend", choices=["julia", "jax"], default="julia")
    ap.add_argument("--device", choices=["cpu", "gpu"], default="gpu",
                    help="julia device; 'gpu' is the fast no-OOM path (default)")
    ap.add_argument("--compare-jax", action="store_true", help="also generate one JAX catalog to cross-check")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    print(f"[boss-lgcp] loading CMASS-South (randoms) ...", flush=True)
    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256)
    w_comp = cat.w_sys_data * cat.w_noz_data * cat.w_cp_data
    n_cand = args.n_cand_factor * cat.N_data
    print(f"[boss-lgcp] N_data={cat.N_data:,}  n_cand={n_cand:,}  backend={args.backend}", flush=True)

    te = np.concatenate([[0.0], np.geomspace(0.02, 2.5, 16)])
    ze = np.linspace(0.0, 0.03, 11)

    print("[boss-lgcp] measuring K_in (window-deconvolved LS) ...", flush=True)
    _, _, xi_w, cnt = measure_K2d_data(cat, theta_edges=te, z_edges=ze,
                                       n_data=args.n_data_meas, n_rand_factor=args.n_rand_meas,
                                       seed=0, return_counts=True)
    xi_in, ic = deconvolve_window(xi_w, cnt["rr"])
    cov, sigma2 = kernel_from_K2d(te, ze, xi_in, alpha=args.alpha)
    print(f"[boss-lgcp] K_in IC={ic:.4f}  kernel σ²={sigma2:.3f}  cov={type(cov).__name__}", flush=True)

    print(f"[boss-lgcp] generating {args.n_samples} catalogs ({args.backend}) ...", flush=True)
    cats = generate_catalogs_from_kernel(
        cat, cov, sigma2, alpha=args.alpha, n_samples=args.n_samples, seed=1,
        w_completeness=w_comp, n_cand_factor=args.n_cand_factor,
        backend=args.backend, device=args.device, verbose=True)

    # science validation: re-measured K_out of catalog 0 vs K_in, at the core (Δθ small, Δz=0)
    c0 = cats[0]
    K_out = measure_Kout(cat, c0["ra"], c0["dec"], c0["z"], te, ze, seed=7)
    core_in = float(np.log1p(xi_in[0, 0]))
    core_out = float(np.log1p(np.clip(K_out[0, 0], -0.99, None)))
    print(f"[boss-lgcp] K core  in={core_in:.3f}  out={core_out:.3f}  "
          f"(N_gal sample0={c0['N_galaxies']:,})", flush=True)

    for i, c in enumerate(cats):
        np.savez_compressed(os.path.join(OUT, f"cmass_south_lgcp_{args.backend}_{i}.npz"),
                            ra=c["ra"], dec=c["dec"], z=c["z"], N_galaxies=np.int64(c["N_galaxies"]))
    print(f"[boss-lgcp] wrote {len(cats)} catalogs to {os.path.relpath(OUT)}", flush=True)

    if args.compare_jax and args.backend == "julia":
        print("[boss-lgcp] cross-check: one JAX catalog (same seed) ...", flush=True)
        cj = generate_catalogs_from_kernel(cat, cov, sigma2, alpha=args.alpha, n_samples=1, seed=1,
                                           w_completeness=w_comp, n_cand_factor=args.n_cand_factor,
                                           backend="jax", verbose=False)[0]
        print(f"[boss-lgcp] N_gal  julia={cats[0]['N_galaxies']:,}  jax={cj['N_galaxies']:,}  "
              f"(statistical match expected; graphs differ in tie-breaking)", flush=True)


if __name__ == "__main__":
    main()
