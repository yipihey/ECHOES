"""Redshift-failure assignment check — are z-failures calibrated against truth?

Fiber-collision targets get the empirical close-pair Δz prior anchoring them to
their host; redshift failures get NO such anchor, so they are effectively drawn
from their angular neighbours' redshift distribution. That is well-motivated for
collisions but assumes failures share the neighbours' n(z) — which real low-S/N
failures might not. We test it directly on real-BOSS-truth (the true redshift of
each injected failure is known): per missing galaxy, the ensemble of assigned
redshifts is a sample of its posterior, and the rank of the true redshift within
that ensemble (the PIT) is uniform iff the assignment is calibrated. We split
collided vs zfail and quantify uniformity with KS + χ² (echoes.pit).

    JAX_PLATFORMS=cpu OMP_NUM_THREADS=16 ~/.venv/k3d/bin/python3 \
        validation/zfail_check.py [--n-real 24 --zfail-frac 0.03]
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from echoes.surveys.boss import load_boss
from echoes.photoz import PhotoZKNN, photoz_features
from echoes.completion import complete_catalog_photoz, measure_close_pair_dz
from echoes.mock_systematics import apply_survey_systematics
from echoes.pit import pit_uniformity, format_pit

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-real", type=int, default=24)
    p.add_argument("--coll-frac", type=float, default=0.6)
    p.add_argument("--zfail-frac", type=float, default=0.03)
    p.add_argument("--out", default="output/zfail_check.png")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    ra = np.asarray(cat.ra_data); dec = np.asarray(cat.dec_data); z = np.asarray(cat.z_data)
    colors = np.asarray(cat.colors_data); mags = np.asarray(cat.mags_data); wsys = np.asarray(cat.w_sys_data)

    obs, tg, kept, true_z = apply_survey_systematics(
        ra, dec, z, colors, mags, wsys, coll_frac=args.coll_frac,
        zfail_frac=args.zfail_frac, zfail_faint_bias=1.5, seed=0)
    kind = np.asarray(tg.miss_kind)
    print(f"missing N={tg.N:,} ({int((kind=='collided').sum()):,} collided + "
          f"{int((kind=='zfail').sum()):,} zfail)")

    of = photoz_features(obs.colors_data, obs.mags_data); ogood = np.isfinite(of).all(1)
    pz = PhotoZKNN(k=100).fit(of[ogood], np.asarray(obs.z_data)[ogood])
    dz = measure_close_pair_dz(obs, 62/3600.)
    N_obs = obs.N_data

    # ensemble of assigned redshifts per missing galaxy. systot_thin is disabled
    # here only so the base array (observed + missing) is returned intact and the
    # per-target redshift z_miss can be sliced out; systot thinning is orthogonal
    # to the redshift assignment this test probes.
    Z = np.empty((args.n_real, tg.N))
    for s in range(args.n_real):
        c = complete_catalog_photoz(obs, tg, pz, seed=s, dz_pool=dz, systot_thin=False)
        Z[s] = np.asarray(c["z"])[N_obs:N_obs + tg.N]
    rng = np.random.default_rng(1)
    # ensemble PIT = rank of true z among the realizations (+ tie jitter)
    pit = ((Z < true_z[None, :]).sum(0) + rng.uniform(size=tg.N)
           * (Z == true_z[None, :]).sum(0)) / args.n_real
    dz_mean = Z.mean(0) - true_z                              # mean assignment error

    print(f"\n=== per-galaxy redshift assignment vs truth (ensemble PIT; uniform=calibrated) ===")
    for sub in ("collided", "zfail"):
        m = kind == sub
        pu = pit_uniformity(pit[m])
        d = dz_mean[m]
        print(f"  {sub:9s} (N={int(m.sum()):,}): PIT {format_pit(pu)}")
        print(f"             Δz=⟨z⟩-z_true: median {np.median(d):+.4f}  RMS {np.sqrt(np.mean(d**2)):.4f}  "
              f"catastrophic(|Δz|>0.05) {100*np.mean(np.abs(d)>0.05):.1f}%")
    print("\n(zfail PIT uniform + small Δz bias => failures are NOT mis-assigned by the "
          "no-close-pair-anchor treatment; a U-shaped or skewed PIT would flag it.)")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for a, sub, col in [(ax[0], "collided", "#3a6ea8"), (ax[1], "zfail", "#c0392b")]:
        m = kind == sub; pu = pit_uniformity(pit[m])
        a.hist(pit[m], bins=12, range=(0, 1), color=col, alpha=0.85, edgecolor="white")
        a.axhline(m.sum()/12, color="k", ls="--", label="uniform (calibrated)")
        a.set_xlabel("ensemble PIT: rank of true z"); a.set_ylabel("count"); a.legend(fontsize=8)
        a.set_title(f"{sub} (N={int(m.sum()):,})\nKS p={pu['ks_p']:.2f}, χ² p={pu['chi2_p']:.2f}")
    fig.suptitle("Redshift-assignment calibration by missing kind (real-BOSS-truth)", y=1.02)
    fig.tight_layout(); fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
