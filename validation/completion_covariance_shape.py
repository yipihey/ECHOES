"""Does the released (factorized) sampler get the completion-covariance SHAPE right?

Referee question (the headline "posterior samples for clustering beyond two points"):
the shipped package draws each missing redshift INDEPENDENTLY from its own fixed
per-object inverse-CDF (``z_mode='field'`` / ``echoes.posterior``). The optional
Matheron engine (``z_mode='fieldpost'``) instead draws a JOINT field realization, so
objects sharing a sightline/structure co-vary. A user is told to ADD the completion
covariance to their cosmic covariance — so it is not enough that the two samplers
agree on the completion-variance SIZE (per-bin scatter); the cross-bin/cross-scale
SHAPE of that covariance must agree too, or the factorized default mis-states the
term it asks users to add.

Inject-and-recover on the REAL CMASS-South truth (fast; no Patchy pair counting):
punch fiber collisions + faint-biased z-failures into the truth, then for each engine
build N completion realizations and measure a higher-order statistic vector per
realization. We compare the completion covariance of the two engines by:

  * SIZE  — per-bin completion std (should match if both are marginally calibrated),
  * SHAPE — the correlation matrix: mean |off-diagonal corr|, the Frobenius distance
            ||R_field - R_fieldpost||, and the total variance (trace of C).

Prediction if the factorized default is deficient: a coherent (joint) field draw
shifts many objects together, so the Matheron ensemble should show LARGER off-diagonal
correlations and total variance; the factorized one decorrelates and under-disperses
(the "mildly over-confident" PIT the paper reports, made covariance-explicit).

Statistic vector (shared fixed query/cell points for both engines, every realization):
  * kNN-CDF P_{>=k}(<r) for k=1,2,4 at fixed comoving radii (angular-3D),
  * counts-in-cells var/mean and skew in fixed-R spheres.

    OMP_NUM_THREADS=16 JAX_PLATFORMS=cpu ~/.venv/k3d/bin/python3 \
        validation/completion_covariance_shape.py [--n-real 40 --inject-seed 0]
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy.spatial import cKDTree

from echoes.surveys.boss import load_boss
from echoes.photoz import PhotoZKNN, photoz_features
from echoes.completion import complete_catalog_photoz, measure_close_pair_dz
from echoes.clustering import comoving_mpc_h
from echoes.mock_systematics import apply_survey_systematics

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"


def xyz(ra, dec, z):
    d = comoving_mpc_h(z); r = np.radians(ra); dd = np.radians(dec)
    return np.column_stack([d * np.cos(dd) * np.cos(r), d * np.cos(dd) * np.sin(r), d * np.sin(dd)])


def stat_vector(gal, q_knn, cen_cic, ks, knn_radii, cic_R):
    """Higher-order statistic vector for one catalog: kNN-CDF(k,r) ++ CiC(var/mean, skew)."""
    tree = cKDTree(gal)
    dist, _ = tree.query(q_knn, k=max(ks), workers=-1)
    parts = []
    for k in ks:
        dk = np.sort(dist[:, k - 1])
        parts.append(np.searchsorted(dk, knn_radii) / len(q_knn))      # CDF at fixed radii
    out = [np.concatenate(parts)]
    for R in cic_R:
        n = tree.query_ball_point(cen_cic, R, return_length=True).astype(float)
        m = max(n.mean(), 1e-9)
        out.append(np.array([n.var() / m, ((n - n.mean()) ** 3).mean() / max(n.std() ** 3, 1e-9)]))
    return np.concatenate(out)


def corr(C):
    d = np.sqrt(np.clip(np.diag(C), 1e-30, None))
    return C / np.outer(d, d)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-real", type=int, default=40)
    p.add_argument("--inject-seed", type=int, default=0)
    p.add_argument("--n-query", type=int, default=20000)
    args = p.parse_args()
    rng0 = np.random.default_rng(12345)

    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    ra = np.asarray(cat.ra_data); dec = np.asarray(cat.dec_data); z = np.asarray(cat.z_data)
    feat = photoz_features(cat.colors_data, cat.mags_data)
    good = np.isfinite(feat).all(1) & (cat.imatch_data == 1)
    pz = PhotoZKNN(k=100).fit(feat[good], z[good])

    obs, tg, kept, true_z = apply_survey_systematics(
        ra, dec, z, cat.colors_data, cat.mags_data, np.asarray(cat.w_sys_data),
        coll_frac=0.6, zfail_frac=0.014, zfail_faint_bias=1.5, seed=args.inject_seed)
    dz = measure_close_pair_dz(obs, 62 / 3600.)
    n_miss = len(np.atleast_1d(true_z))
    print(f"inject-and-recover: {len(np.atleast_1d(obs.ra_data)):,} observed + {n_miss:,} missing", flush=True)

    # fixed query / cell points (shared by BOTH engines and EVERY realization) drawn
    # from the truth volume so the statistic is well-sampled and differences are
    # purely from the imputed redshifts.
    gtruth = xyz(ra, dec, z)
    lo, hi = gtruth.min(0), gtruth.max(0)
    q_knn = rng0.uniform(lo, hi, size=(args.n_query, 3))
    cen_cic = rng0.uniform(lo, hi, size=(args.n_query, 3))
    ks = [1, 2, 4]
    knn_radii = np.array([8.0, 16.0, 28.0])          # h^-1 Mpc, ~ kNN-CDF rising edge
    cic_R = [12.0, 25.0]                              # small + large spheres (large = coherent-shift sensitive)
    labels = ([f"kNN{k}@{int(r)}" for k in ks for r in knn_radii]
              + [f"CiC{int(R)}:{s}" for R in cic_R for s in ("v/m", "skew")])

    def ensemble(engine):
        S = []
        ckw = {}
        if engine == "fieldpost":
            from echoes.fieldpost import build_field_context
            fctx = build_field_context(obs, seed=args.inject_seed, n_samples=max(args.n_real, 2),
                                       sel_map=cat.sel_map, nside=cat.nside)
            ckw = dict(z_mode="fieldpost", field_ctx=fctx)
        for s in range(args.n_real):
            c = complete_catalog_photoz(obs, tg, pz, seed=1000 * args.inject_seed + s,
                                        dz_pool=dz, **ckw)
            S.append(stat_vector(xyz(np.asarray(c["ra"]), np.asarray(c["dec"]), np.asarray(c["z"])),
                                 q_knn, cen_cic, ks, knn_radii, cic_R))
            if (s + 1) % 10 == 0:
                print(f"  {engine}: {s+1}/{args.n_real}", flush=True)
        return np.array(S)                            # (n_real, nbin)

    print("\nbuilding factorized (released 'field') ensemble ...", flush=True)
    Sf = ensemble("field")
    print("building joint Matheron ('fieldpost') ensemble ...", flush=True)
    Sj = ensemble("fieldpost")

    # also the truth statistic (single point) for context
    s_truth = stat_vector(gtruth, q_knn, cen_cic, ks, knn_radii, cic_R)

    Cf, Cj = np.cov(Sf, rowvar=False), np.cov(Sj, rowvar=False)
    Rf, Rj = corr(Cf), corr(Cj)
    sf, sj = np.sqrt(np.diag(Cf)), np.sqrt(np.diag(Cj))
    iu = np.triu_indices(len(labels), k=1)

    print(f"\n=== completion covariance: factorized (released) vs joint Matheron "
          f"(n_real={args.n_real}) ===")
    print(f"{'bin':14s}{'mean':>10}{'std_fac':>11}{'std_joint':>11}{'ratio j/f':>11}")
    for i, lab in enumerate(labels):
        print(f"{lab:14s}{Sf[:,i].mean():10.4g}{sf[i]:11.3g}{sj[i]:11.3g}{sj[i]/max(sf[i],1e-30):11.2f}")

    print(f"\nSIZE   total completion variance (trace C): factorized {np.trace(Cf):.3g}  "
          f"joint {np.trace(Cj):.3g}  (ratio joint/fac {np.trace(Cj)/max(np.trace(Cf),1e-30):.2f})")
    print(f"       median per-bin std ratio joint/factorized: {np.median(sj/np.maximum(sf,1e-30)):.2f}")
    print(f"SHAPE  mean |off-diagonal corr|: factorized {np.mean(np.abs(Rf[iu])):.3f}   "
          f"joint {np.mean(np.abs(Rj[iu])):.3f}")
    print(f"       ||R_fac - R_joint||_F = {np.linalg.norm(Rf - Rj):.3f}  "
          f"(0 = identical shape; ~sqrt(2*npair) is orthogonal)")
    # leading-eigenvector overlap of the two completion covariances
    wf, Vf = np.linalg.eigh(Cf); wj, Vj = np.linalg.eigh(Cj)
    overlap = abs(Vf[:, -1] @ Vj[:, -1])
    print(f"       leading-eigvec overlap |v_fac . v_joint| = {overlap:.3f}  "
          f"(1 = same dominant completion mode)")
    print(f"\nINTERPRETATION: if joint std >> factorized std and joint off-diagonal corr >> "
          f"factorized, the released sampler UNDER-disperses and DECORRELATES the completion\n"
          f"term it tells users to add (the factorization/plug-in deficiency). If they match, "
          f"the factorization is empirically vindicated at this precision.")

    os.makedirs("output", exist_ok=True)
    np.savez("output/completion_covariance_shape.npz",
             labels=np.array(labels), Sf=Sf, Sj=Sj, Cf=Cf, Cj=Cj, s_truth=s_truth)
    print("\nsaved output/completion_covariance_shape.npz")


if __name__ == "__main__":
    main()
