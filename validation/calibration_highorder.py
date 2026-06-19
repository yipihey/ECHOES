"""Beyond-two-point calibration — is the ensemble a calibrated posterior for the
nearest-neighbour and counts-in-cells statistics that are the paper's point?

`calibration.py` shows the completion ensemble is a calibrated posterior for
w_p(r_p). The higher-order statistics are more sensitive to the small-scale
redshift assignment, so the calibration must be demonstrated there too. Across
MultiDark-Patchy mocks we inject the systematics, build the completion ensemble,
and test whether the truth falls within the ensemble credible interval at the
nominal rate (coverage / PIT) for:

  * the angular-3D kNN-CDF P_{>=k}(<r), k = 1, 2, 4 (Banerjee & Abel 2021), and
  * the counts-in-cells variance σ²_N/⟨N⟩ in fixed spheres,

with the same fixed random query/cell points used for truth and every
realization of every mock. PIT uniformity is quantified with KS + χ² (a U-shaped
over-confident PIT also has mean 0.5; see echoes.pit).

    OMP_NUM_THREADS=32 JAX_PLATFORMS=cpu ~/.venv/k3d/bin/python3 \
        validation/calibration_highorder.py [--n-mocks 6 --n-real 10]
"""
import argparse, glob, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from echoes.surveys.boss import load_boss
from echoes.photoz import PhotoZKNN, photoz_features
from echoes.completion import complete_catalog_photoz, measure_close_pair_dz
from echoes.clustering import comoving_mpc_h
from echoes.mock_systematics import (apply_survey_systematics, load_patchy_truth,
                                     load_patchy_randoms)
from echoes.pit import pit_uniformity, format_pit

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"
MOCKDIR = "data/boss/mocks"


def xyz(ra, dec, z):
    d = comoving_mpc_h(z); r = np.radians(ra); dd = np.radians(dec)
    return np.column_stack([d*np.cos(dd)*np.cos(r), d*np.cos(dd)*np.sin(r), d*np.sin(dd)])


def knn_cdf(gal_xyz, q_xyz, ks, redges):
    dist, _ = cKDTree(gal_xyz).query(q_xyz, k=max(ks), workers=-1)
    return np.concatenate([np.searchsorted(np.sort(dist[:, k-1]), redges) / len(q_xyz) for k in ks])


def cic_varmean(gal_xyz, cen_xyz, R):
    n = cKDTree(gal_xyz).query_ball_point(cen_xyz, R, return_length=True).astype(float)
    return np.array([n.var() / max(n.mean(), 1e-9)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-mocks", type=int, default=6)
    p.add_argument("--n-real", type=int, default=10)
    p.add_argument("--out", default="output/calibration_highorder.png")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    z = np.asarray(cat.z_data); feat = photoz_features(cat.colors_data, cat.mags_data)
    good = np.isfinite(feat).all(1) & (cat.imatch_data == 1)
    pz = PhotoZKNN(k=100).fit(feat[good], z[good])

    rar, decr, zr = load_patchy_randoms(
        f"{MOCKDIR}/Patchy-Mocks-Randoms-DR12SGC-COMPSAM_V6C_x10.dat",
        z_min=0.43, z_max=0.7, max_n=300_000)
    rng = np.random.default_rng(3)
    ks = (1, 2, 4); redges = np.logspace(np.log10(3.0), np.log10(35.0), 12)
    Rcic = 10.0

    mocks = sorted(glob.glob(f"{MOCKDIR}/Patchy-Mocks-DR12SGC-COMPSAM_V6C_*.dat"))[:args.n_mocks]
    knn_truth, knn_ens, cic_truth, cic_ens = [], [], [], []
    for mi, mf in enumerate(mocks):
        ra, dec, zz, colors, mags, wsys = load_patchy_truth(mf, cat, z_min=0.43, z_max=0.7)
        obs, tg, kept, _ = apply_survey_systematics(ra, dec, zz, colors, mags, wsys,
                                                    coll_frac=0.6, zfail_frac=0.014,
                                                    zfail_faint_bias=1.5, seed=mi)
        dz = measure_close_pair_dz(obs, 62/3600.)
        # fixed query / cell points for this mock (footprint-uniform from randoms)
        qi = rng.choice(len(rar), 40000, replace=False); q_xyz = xyz(rar[qi], decr[qi], zr[qi])
        ci = rng.choice(len(rar), 6000, replace=False); c_xyz = xyz(rar[ci], decr[ci], zr[ci])
        knn_truth.append(knn_cdf(xyz(ra, dec, zz), q_xyz, ks, redges))
        cic_truth.append(cic_varmean(xyz(ra, dec, zz), c_xyz, Rcic))
        Kn, Ci = [], []
        for s in range(args.n_real):
            c = complete_catalog_photoz(obs, tg, pz, seed=100*mi+s, dz_pool=dz)
            gx = xyz(np.asarray(c["ra"]), np.asarray(c["dec"]), np.asarray(c["z"]))
            Kn.append(knn_cdf(gx, q_xyz, ks, redges)); Ci.append(cic_varmean(gx, c_xyz, Rcic))
        knn_ens.append(np.array(Kn)); cic_ens.append(np.array(Ci))
        print(f"  mock {mi+1}/{len(mocks)} done", flush=True)

    def cov_pit(truth, ens):
        truth = np.array(truth); ens = np.array(ens)               # (Nm,nbin), (Nm,nreal,nbin)
        lo = np.percentile(ens, 16, axis=1); hi = np.percentile(ens, 84, axis=1)
        inside = (truth >= lo) & (truth <= hi)
        pit = (ens < truth[:, None, :]).mean(axis=1).ravel()
        return float(inside.mean()), pit

    cov_k, pit_k = cov_pit(knn_truth, knn_ens)
    cov_c, pit_c = cov_pit(cic_truth, cic_ens)
    pu_k, pu_c = pit_uniformity(pit_k), pit_uniformity(pit_c)
    print(f"\n=== beyond-2pt calibration over {len(mocks)} mocks × {args.n_real} realizations ===")
    print(f"kNN-CDF (k=1,2,4): coverage {cov_k:.2f} (target 0.68)   PIT {format_pit(pu_k)}")
    print(f"CIC σ²/⟨N⟩ (R={Rcic}): coverage {cov_c:.2f} (target 0.68)   PIT {format_pit(pu_c)}")
    print("\n(coverage≈0.68 and KS/χ² p≳0.05 => the ensemble is a calibrated posterior for the "
          "non-Gaussian statistics, not just w_p.)")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for a, pit, pu, name, cov in [(ax[0], pit_k, pu_k, "kNN-CDF (k=1,2,4)", cov_k),
                                  (ax[1], pit_c, pu_c, f"CIC σ²/⟨N⟩ (R={Rcic} Mpc/h)", cov_c)]:
        a.hist(pit, bins=10, range=(0, 1), color="#3a6ea8", alpha=0.85, edgecolor="white")
        a.axhline(len(pit)/10, color="r", ls="--", label="uniform (calibrated)")
        a.set_xlabel("PIT: rank of truth in ensemble"); a.set_ylabel("count"); a.legend(fontsize=8)
        a.set_title(f"{name}\ncoverage {cov:.2f}/0.68, KS p={pu['ks_p']:.2f}, χ² p={pu['chi2_p']:.2f}")
    fig.suptitle("Beyond-two-point calibration of the completion ensemble", y=1.02)
    fig.tight_layout(); fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
