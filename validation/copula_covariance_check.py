"""Production check: does build_package(copula=True) -> draw() close the variance gap?

The prototype (completion_copula_prototype.py) used a hand-rolled exponential kernel.
This runs the SHIPPED code path: build the posterior package with the field-correlation
copula (measured xi(r) kernel via build_field_context), then compare the completion
covariance of the IID draw vs the copula draw — SAME package, SAME marginals, so the
only difference is the cross-object dependence. We also print the joint Matheron trace
(from output/completion_covariance_shape.npz) as the target direction.

PASS: copula draw raises the total completion variance and lifts the coherence-sensitive
(large-scale CiC, intermediate kNN) bins toward the joint, with per-object marginals
unchanged (proven separately in tests/test_copula.py).

    OMP_NUM_THREADS=16 JAX_PLATFORMS=cpu ~/.venv/k3d/bin/python3 \
        validation/copula_covariance_check.py [--n-real 48 --copula-modes 128]
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from echoes.surveys.boss import load_boss
from echoes.photoz import PhotoZKNN, photoz_features
from echoes.completion import measure_close_pair_dz
from echoes.mock_systematics import apply_survey_systematics
from echoes.fieldpost import build_field_context
from echoes.posterior import build_package, draw
from completion_covariance_shape import xyz, stat_vector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-real", type=int, default=48)
    ap.add_argument("--inject-seed", type=int, default=0)
    ap.add_argument("--n-query", type=int, default=20000)
    ap.add_argument("--copula-modes", type=int, default=128)
    args = ap.parse_args()
    rng0 = np.random.default_rng(12345)                  # match completion_covariance_shape.py

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

    fctx = build_field_context(obs, seed=args.inject_seed, sel_map=cat.sel_map,
                               nside=cat.nside, verbose=True)
    pkg = build_package(obs, tg, pz, dz_pool=dz, copula=True, field_ctx=fctx,
                        copula_modes=args.copula_modes, verbose=True)
    assert pkg.get("cmodes") is not None, "copula modes not attached"
    print(f"package: {pkg['n_miss']} missing, copula modes {pkg['cmodes'].shape}", flush=True)

    gtruth = xyz(ra, dec, z); lo, hi = gtruth.min(0), gtruth.max(0)
    q_knn = rng0.uniform(lo, hi, size=(args.n_query, 3))
    cen_cic = rng0.uniform(lo, hi, size=(args.n_query, 3))
    ks = [1, 2, 4]; knn_radii = np.array([8.0, 16.0, 28.0]); cic_R = [12.0, 25.0]
    labels = ([f"kNN{k}@{int(r)}" for k in ks for r in knn_radii]
              + [f"CiC{int(R)}:{s}" for R in cic_R for s in ("v/m", "skew")])

    def ens(use_copula):
        S = []
        for s in range(args.n_real):
            c = draw(pkg, seed=1000 * args.inject_seed + s, systot=False, copula=use_copula)
            S.append(stat_vector(xyz(c["ra"], c["dec"], c["z"]),
                                 q_knn, cen_cic, ks, knn_radii, cic_R))
        return np.array(S)

    print("IID draws ...", flush=True);    Si = ens(False)
    print("copula draws ...", flush=True); Sc = ens(True)
    Ci, Cc = np.cov(Si, rowvar=False), np.cov(Sc, rowvar=False)
    si, sc = np.sqrt(np.diag(Ci)), np.sqrt(np.diag(Cc))

    trj = None
    p = "output/completion_covariance_shape.npz"
    if os.path.exists(p):
        trj = float(np.trace(np.load(p, allow_pickle=True)["Cj"]))

    print(f"\n=== production copula vs IID (same package/marginals, n_real={args.n_real}) ===")
    print(f"{'bin':14s}{'std_IID':>11}{'std_cop':>11}{'cop/IID':>10}")
    for i, lab in enumerate(labels):
        print(f"{lab:14s}{si[i]:11.3g}{sc[i]:11.3g}{sc[i]/max(si[i],1e-30):10.2f}")
    print(f"\ntrace(C): IID {np.trace(Ci):.3g}  copula {np.trace(Cc):.3g}  "
          f"(copula/IID {np.trace(Cc)/max(np.trace(Ci),1e-30):.2f})"
          + (f";  joint Matheron target {trj:.3g}" if trj else ""))
    print(f"median per-bin std ratio copula/IID: {np.median(sc/np.maximum(si,1e-30)):.2f}")
    print("marginals (hence per-object PIT) identical by construction — see tests/test_copula.py")
    np.savez("output/copula_covariance_check.npz", labels=np.array(labels), Si=Si, Sc=Sc)
    print("saved output/copula_covariance_check.npz")


if __name__ == "__main__":
    main()
