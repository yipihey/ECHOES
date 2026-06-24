"""Covariance next-step #4 — Tier-A non-Gaussian LGCP: does the measured transform T(g) kill the
multi-occupancy (which cf-escalation #1 could not) WITHOUT breaking the two-point match?

The lognormal intensity exp(f−σ²/2) at σ²=4.144 has zero-lag variance ξ(0,0)=e^σ²−1≈62 and a FIXED
skew≈500 — the runaway tail behind multi_frac≈0.20. Tier-A keeps the same variance (so the 2-pt can be
matched) but regularises the skew, and re-derives the Gaussian kernel via ξ_T⁻¹ so the 2-pt is
preserved by construction. This sweeps the skew knob and reports, per skew:
  * multi_frac + N_gal  (did multi-occupancy drop?)
  * the re-measured K_out(Δθ,Δz=0) profile vs K_in  (was the 2-pt preserved?)

  JAX_PLATFORMS=cpu OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 \
    python pipeline/diag_tierA.py --skews 30 10 3.1 --n-samples 3 --device gpu

--construct-only skips generation (CPU sanity of the transform + kernel build).
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
                               generate_catalogs_from_kernel, measure_K2d, fkp_weight_of_z)
from echoes.randoms import make_random_from_selection_function

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"


def measure_Kout(cat, ra, dec, z, te, ze, n_rand_factor=3, seed=7):
    """Re-measure K(Δθ,Δz) of a generated (unweighted) catalog (same LS estimator as K_in)."""
    rng = np.random.default_rng(seed)
    ra_r, dec_r, z_r = make_random_from_selection_function(
        sel_map=cat.sel_map, n_random=n_rand_factor * len(ra), z_data=np.asarray(cat.z_data),
        nside=cat.nside, rng=rng)
    w_r = fkp_weight_of_z(z_r, np.asarray(cat.z_data), cat.w_fkp_data) \
        if cat.w_fkp_data is not None else np.ones(len(ra_r))
    _, _, xi = measure_K2d(ra, dec, z, np.ones(len(ra)), ra_r, dec_r, z_r, w_r,
                           theta_edges=te, z_edges=ze, return_counts=False)
    return xi


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skews", type=float, nargs="+", default=[30.0, 10.0, 3.1],
                    help="intensity-PDF skew targets (lognormal native ≈500 → multi_frac≈0.20)")
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--n-cand-factor", type=int, default=20)
    ap.add_argument("--n-data-meas", type=int, default=60_000)
    ap.add_argument("--n-rand-meas", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--construct-only", action="store_true", help="CPU sanity; skip generation")
    args = ap.parse_args()

    print("[tierA] loading CMASS-South ...", flush=True)
    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256)
    w_comp = cat.w_sys_data * cat.w_noz_data * cat.w_cp_data

    te = np.concatenate([[0.0], np.geomspace(0.02, 2.5, 16)])
    ze = np.linspace(0.0, 0.03, 11)
    print("[tierA] measuring K_in ...", flush=True)
    _, _, xi_w, cnt = measure_K2d_data(cat, theta_edges=te, z_edges=ze, n_data=args.n_data_meas,
                                       n_rand_factor=args.n_rand_meas, seed=0, return_counts=True)
    xi_in, ic = deconvolve_window(xi_w, cnt["rr"])
    cov_ln, sigma2 = kernel_from_K2d(te, ze, xi_in, alpha=args.alpha)
    var0 = float(np.expm1(sigma2))
    # lognormal native skew at this variance (ω=e^σ², plain-lognormal skew (ω+2)√(ω-1))
    w = np.exp(sigma2)
    skew_ln = (w + 2.0) * np.sqrt(w - 1.0)
    print(f"[tierA] sigma2={sigma2:.3f}  ξ(0,0)=var0={var0:.2f}  lognormal skew≈{skew_ln:.0f}\n",
          flush=True)

    # K_in(Δθ,Δz=0) profile (the 2-pt target the Tier-A field must preserve)
    kin_prof = xi_in[:, 0]
    theta_c = np.empty(len(te) - 1); theta_c[0] = 0.5 * te[1]
    theta_c[1:] = np.sqrt(te[1:-1] * te[2:])

    rows = []
    for sk in args.skews:
        dt = lgcp_density_transform(sigma2, skew=sk, kind="lognormal")
        # peak intensity at g=3,4 (the multi-occupancy driver) vs the lognormal
        Tg = dt.T(np.array([3.0, 4.0]))
        Tln = np.exp(np.array([3.0, 4.0]) * np.sqrt(sigma2) - 0.5 * sigma2)
        cov_t, s2_t = kernel_from_K2d_tierA(te, ze, xi_in, dt, alpha=args.alpha)
        print(f"[tierA] skew={sk:6.1f}: σ_g={dt.sigma_g:.3f} δ0={dt.delta0:.2f} var(T)={dt.var_opd:.1f}"
              f"  T(3σ)={Tg[0]:.1f} T(4σ)={Tg[1]:.1f}  (lognormal {Tln[0]:.0f}/{Tln[1]:.0f})"
              f"  kernel diag={s2_t:.3f}", flush=True)
        rows.append((sk, dt, cov_t, s2_t))

    if args.construct_only:
        print("\n[tierA] --construct-only: kernel + transform build OK (CPU). Stop.", flush=True)
        return

    print(f"\n[tierA] generating (lognormal baseline + {len(rows)} Tier-A skews), {args.device} ...",
          flush=True)
    # lognormal baseline
    _t = time.perf_counter()
    base = generate_catalogs_from_kernel(cat, cov_ln, sigma2, alpha=args.alpha,
                                         n_samples=args.n_samples, seed=args.seed, w_completeness=w_comp,
                                         n_cand_factor=args.n_cand_factor, sampling="poisson",
                                         backend="julia", device=args.device, verbose=False)
    mf_ln = np.mean([c["multi_frac"] for c in base]); ng_ln = np.mean([c["N_galaxies"] for c in base])
    print(f"[tierA] lognormal: multi_frac={mf_ln:.4f}  N_gal={ng_ln:,.0f}  [{time.perf_counter()-_t:.0f}s]",
          flush=True)

    for sk, dt, cov_t, s2_t in rows:
        _t = time.perf_counter()
        out = generate_catalogs_from_kernel(cat, cov_t, s2_t, alpha=args.alpha,
                                            n_samples=args.n_samples, seed=args.seed,
                                            w_completeness=w_comp, n_cand_factor=args.n_cand_factor,
                                            sampling="poisson", transform=dt, backend="julia",
                                            device=args.device, verbose=False)
        mf = np.mean([c["multi_frac"] for c in out]); ng = np.mean([c["N_galaxies"] for c in out])
        # 2-pt preservation: re-measure K_out on the first realization, compare profile to K_in
        c0 = out[0]
        xi_out = measure_Kout(cat, np.asarray(c0["ra"], float), np.asarray(c0["dec"], float),
                              np.asarray(c0["z"], float), te, ze, n_rand_factor=args.n_rand_meas)
        kout_prof = xi_out[:, 0]
        # ratio K_out/K_in over the well-measured small-θ bins
        m = (kin_prof > 0) & np.isfinite(kout_prof)
        rr = np.nanmedian(kout_prof[m] / kin_prof[m]) if m.any() else np.nan
        print(f"[tierA] skew={sk:6.1f}: multi_frac={mf:.4f}  N_gal={ng:,.0f}  "
              f"K_out/K_in(med)={rr:.2f}  [{time.perf_counter()-_t:.0f}s]", flush=True)

    print(f"\n[tierA] DONE. lognormal multi_frac={mf_ln:.4f} is the baseline; Tier-A should be lower "
          f"with K_out/K_in≈1 (2-pt preserved).", flush=True)


if __name__ == "__main__":
    main()
