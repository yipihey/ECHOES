"""Prototype: does a Gaussian COPULA over the released marginals close the 19%?

`completion_covariance_shape.py` showed the released factorized sampler
(z_i = invcdf_i(u_i), u_i IID) under-disperses the coherent large-scale completion
variance and over-disperses small-scale, relative to the joint Matheron sampler.
The factorized sampler IS a Gaussian copula with the IDENTITY correlation. Here we
swap the identity for a field-scale correlation, keeping the SAME per-object
marginals (so per-object PIT calibration is byte-identical):

    z_i = invcdf_i( Phi(g_i) ),   g ~ N(0, C),   C_ij = exp(-r_ij / r0)

C is built on the missing galaxies' 3-D positions (angular position + each object's
own marginal-median redshift as a fixed placement proxy). r_ij is comoving
separation; r0 is swept over the field scale. Only the JOINT law of the draws
changes — every marginal is the released one, so calibration is preserved by
construction. (Production would use the measured xi(r) / the conditional field
posterior covariance; the exponential kernel here is the clean PSD demonstrator.)

We compare, on the SAME fixed query/cell points as completion_covariance_shape.py:
  * factorized_pkg  — IID copula over the package marginals (the released sampler),
  * copula_pkg(r0)  — field-correlation copula over the SAME marginals,
  * joint Matheron  — loaded from output/completion_covariance_shape.npz (the target).
Metric: total completion variance (trace C) and per-bin std vs the joint target.

    OMP_NUM_THREADS=16 JAX_PLATFORMS=cpu ~/.venv/k3d/bin/python3 \
        validation/completion_copula_prototype.py [--n-real 48]
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import ndtr                      # standard-normal CDF Phi

from echoes.surveys.boss import load_boss
from echoes.photoz import PhotoZKNN, photoz_features
from echoes.completion import measure_close_pair_dz
from echoes.clustering import comoving_mpc_h
from echoes.mock_systematics import apply_survey_systematics
from echoes.posterior import build_package
# reuse the exact statistic + geometry helpers so q-points/stat match the prior run
from completion_covariance_shape import xyz, stat_vector


def z_from_pkg(pkg, u):
    """Map uniforms u -> redshifts via the package inverse-CDF (mirrors posterior.draw)."""
    qlev = np.asarray(pkg["qlev"], float); invcdf = np.asarray(pkg["invcdf"], float)
    nq = len(qlev); M = invcdf.shape[0]
    j = np.clip(np.searchsorted(qlev, u), 1, nq - 1)
    q0, q1 = qlev[j - 1], qlev[j]
    v0 = invcdf[np.arange(M), j - 1]; v1 = invcdf[np.arange(M), j]
    z = v0 + (v1 - v0) * (u - q0) / np.maximum(q1 - q0, 1e-12)
    return np.clip(z, float(pkg["zmin"]), float(pkg["zmax"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-real", type=int, default=48)
    p.add_argument("--inject-seed", type=int, default=0)
    p.add_argument("--n-query", type=int, default=20000)
    p.add_argument("--r0", type=float, nargs="+", default=[10.0, 20.0, 40.0],
                   help="copula correlation lengths [h^-1 Mpc] to sweep")
    args = p.parse_args()
    rng0 = np.random.default_rng(12345)              # MUST match completion_covariance_shape.py

    cat = load_boss(["data/boss/galaxy_DR12v5_CMASS_South.fits.gz"],
                    ["data/boss/random0_DR12v5_CMASS_South.fits.gz"],
                    sample="CMASS", nside=256, with_photometry=True)
    ra = np.asarray(cat.ra_data); dec = np.asarray(cat.dec_data); z = np.asarray(cat.z_data)
    feat = photoz_features(cat.colors_data, cat.mags_data)
    good = np.isfinite(feat).all(1) & (cat.imatch_data == 1)
    pz = PhotoZKNN(k=100).fit(feat[good], z[good])
    obs, tg, kept, true_z = apply_survey_systematics(
        ra, dec, z, cat.colors_data, cat.mags_data, np.asarray(cat.w_sys_data),
        coll_frac=0.6, zfail_frac=0.014, zfail_faint_bias=1.5, seed=args.inject_seed)
    dz = measure_close_pair_dz(obs, 62 / 3600.)
    pkg = build_package(obs, tg, pz, dz_pool=dz)
    tra = np.asarray(tg.ra); tdec = np.asarray(tg.dec); M = len(tra)
    obs_ra = np.asarray(obs.ra_data); obs_dec = np.asarray(obs.dec_data); obs_z = np.asarray(obs.z_data)
    print(f"inject-and-recover: {len(obs_ra):,} observed + {M:,} missing", flush=True)

    # identical fixed query/cell points + statistic config as the prior run
    gtruth = xyz(ra, dec, z); lo, hi = gtruth.min(0), gtruth.max(0)
    q_knn = rng0.uniform(lo, hi, size=(args.n_query, 3))
    cen_cic = rng0.uniform(lo, hi, size=(args.n_query, 3))
    ks = [1, 2, 4]; knn_radii = np.array([8.0, 16.0, 28.0]); cic_R = [12.0, 25.0]
    labels = ([f"kNN{k}@{int(r)}" for k in ks for r in knn_radii]
              + [f"CiC{int(R)}:{s}" for R in cic_R for s in ("v/m", "skew")])

    # missing-galaxy 3-D positions for the copula correlation: marginal-median redshift
    z_proxy = z_from_pkg(pkg, np.full(M, 0.5))
    Xm = xyz(tra, tdec, z_proxy)

    def ensemble_u(u_of_seed):
        S = []
        for s in range(args.n_real):
            u = u_of_seed(s)
            zc = z_from_pkg(pkg, u)
            S.append(stat_vector(xyz(np.r_[obs_ra, tra], np.r_[obs_dec, tdec], np.r_[obs_z, zc]),
                                 q_knn, cen_cic, ks, knn_radii, cic_R))
        return np.array(S)

    # IID copula over the package marginals (= the released sampler's joint law)
    print("factorized_pkg (IID copula) ...", flush=True)
    Sfac = ensemble_u(lambda s: np.random.default_rng(7000 + s).random(M))

    # field-correlation copula: one Cholesky per r0 (seed-independent), reused across seeds
    results = {}
    d = cKDTree(Xm)                                  # only used to report typical separations
    for r0 in args.r0:
        print(f"copula_pkg(r0={r0:g}) — building C ({M}x{M}) ...", flush=True)
        # exponential (Matern-1/2) correlation on comoving separation: PSD by construction
        from scipy.spatial.distance import cdist
        C = np.exp(-cdist(Xm, Xm) / r0)
        C[np.diag_indices(M)] += 1e-8
        L = np.linalg.cholesky(C)
        def u_of_seed(s, L=L):
            g = L @ np.random.default_rng(8000 + s).standard_normal(M)
            return ndtr(g)                            # Phi(g) ~ U(0,1) marginally
        print(f"  drawing {args.n_real} realizations ...", flush=True)
        results[r0] = ensemble_u(u_of_seed)

    # joint Matheron target (the expensive ensemble already computed)
    ref = np.load("output/completion_covariance_shape.npz", allow_pickle=True)
    Cj = ref["Cj"]; sj = np.sqrt(np.diag(Cj)); trj = np.trace(Cj)
    Cf = np.cov(Sfac, rowvar=False); sf = np.sqrt(np.diag(Cf)); trf = np.trace(Cf)

    print(f"\n=== closing the gap: trace(C) vs the joint Matheron target (n_real={args.n_real}) ===")
    print(f"{'sampler':22s}{'trace':>12}{'trace/joint':>13}")
    print(f"{'factorized_pkg (IID)':22s}{trf:12.4g}{trf/trj:13.2f}")
    for r0 in args.r0:
        Cc = np.cov(results[r0], rowvar=False)
        print(f"{f'copula r0={r0:g}':22s}{np.trace(Cc):12.4g}{np.trace(Cc)/trj:13.2f}")
    print(f"{'joint Matheron (target)':22s}{trj:12.4g}{1.00:13.2f}")

    # per-bin std ratio to the joint target: which bins move into [0.9,1.1]?
    print(f"\nper-bin std / joint-target std  (1.0 = matched; <1 under-disperses):")
    print(f"{'bin':14s}{'IID':>8}" + "".join(f"{'r0='+str(int(r)):>9}" for r in args.r0))
    best_r0 = args.r0[int(np.argmin([abs(np.trace(np.cov(results[r0], rowvar=False))/trj - 1)
                                     for r0 in args.r0]))]
    for i, lab in enumerate(labels):
        row = f"{lab:14s}{sf[i]/sj[i]:8.2f}"
        for r0 in args.r0:
            sc = np.sqrt(np.diag(np.cov(results[r0], rowvar=False)))[i]
            row += f"{sc/sj[i]:9.2f}"
        print(row)
    print(f"\nbest-matching r0 (trace closest to joint): {best_r0:g} h^-1 Mpc")
    print("note: this is the UNCONDITIONAL copula (prior field corr); production would use the\n"
          "measured xi(r) / the conditional field-posterior covariance, which also fixes any\n"
          "over-dispersion near observed galaxies. Marginals (hence per-object PIT) are unchanged.")

    np.savez("output/completion_copula_prototype.npz",
             labels=np.array(labels), Sfac=Sfac, r0=np.array(args.r0),
             **{f"S_r0_{int(r)}": results[r] for r in args.r0}, Cj=Cj)
    print("saved output/completion_copula_prototype.npz")


if __name__ == "__main__":
    main()
