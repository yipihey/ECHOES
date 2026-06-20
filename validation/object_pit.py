"""G7 (absolute) — OBJECT-LEVEL redshift PIT of the completion posterior.

The wp-ensemble coverage test (calibration.py) is dominated by cosmic variance for
a ~96%-complete survey (completion uncertainty is sub-dominant by construction), so
its absolute coverage is low for every engine. The natural ABSOLUTE calibration
statement is per-object: for each spectroscopically-missing galaxy with known true
redshift, the PIT = CDF_post(z_true) should be uniform on [0,1] if the per-object
redshift posterior is calibrated.

Real-data inject-and-recover (no Patchy, no pair-counting → fast): punch fiber
collisions + z-failures into the real CMASS-South truth, build each engine's
per-object posterior package, and test PIT uniformity for:

  * field       — the shipped KNN-KDE local-density posterior (build_package);
  * gen-ident   — generative with transform=identity (== fieldpost posterior);
  * gen-transf  — Tier-A measured non-Gaussian transform.

PASS: KS & χ² p ≳ 0.05 (uniform), and gen-transf no worse than gen-ident — the
monotonic transform is rank-preserving, so it must not degrade PIT.

    OMP_NUM_THREADS=16 JAX_PLATFORMS=cpu ~/.venv/k3d/bin/python3 validation/object_pit.py
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from echoes.surveys.boss import load_boss
from echoes.photoz import PhotoZKNN, photoz_features
from echoes.completion import measure_close_pair_dz
from echoes.mock_systematics import apply_survey_systematics
from echoes.fieldpost import build_field_context
from echoes.generative import build_generative_model
from echoes.posterior import build_package, build_package_generative
from echoes.pit import pit_uniformity, format_pit

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"


def object_pit(pkg, true_z):
    """PIT_i = CDF_i(z_true_i) via the per-object inverse-CDF (qlev, invcdf)."""
    qlev = np.asarray(pkg["qlev"], float)
    invcdf = np.asarray(pkg["invcdf"], float)                  # (M, nq), increasing in qlev
    M = invcdf.shape[0]
    pit = np.empty(M)
    for i in range(M):
        pit[i] = np.interp(true_z[i], invcdf[i], qlev, left=0.0, right=1.0)
    return pit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cic-R", type=float, default=8.0)
    args = p.parse_args()

    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    ra = np.asarray(cat.ra_data); dec = np.asarray(cat.dec_data); z = np.asarray(cat.z_data)
    feat = photoz_features(cat.colors_data, cat.mags_data)
    good = np.isfinite(feat).all(1) & (cat.imatch_data == 1)
    pz = PhotoZKNN(k=100).fit(feat[good], z[good])

    obs, tg, kept, true_z = apply_survey_systematics(
        ra, dec, z, cat.colors_data, cat.mags_data, np.asarray(cat.w_sys_data),
        coll_frac=0.6, zfail_frac=0.014, zfail_faint_bias=1.5, seed=args.seed)
    true_z = np.asarray(true_z)
    dz = measure_close_pair_dz(obs, 62/3600.)
    print(f"inject-and-recover: {len(true_z):,} missing galaxies with known true z")

    fctx = build_field_context(obs, seed=args.seed, n_samples=1, sel_map=cat.sel_map, nside=cat.nside)
    gm_id = build_generative_model(obs, transform="identity", field_ctx=fctx)
    gm_t = build_generative_model(obs, transform="empirical", cic_R=args.cic_R,
                                  field_ctx=fctx, cic_randoms=(np.asarray(cat.ra_random),
                                  np.asarray(cat.dec_random), np.asarray(cat.z_random)))

    pkgs = {
        "field": build_package(obs, tg, pz, dz_pool=dz),
        "gen-ident": build_package_generative(obs, tg, pz, gm_id, dz_pool=dz),
        "gen-transf": build_package_generative(obs, tg, pz, gm_t, dz_pool=dz),
    }
    print(f"\n=== object-level redshift PIT uniformity (PASS: KS & χ² p ≳ 0.05) ===")
    stats = {}
    for name, pkg in pkgs.items():
        pit = object_pit(pkg, true_z)
        pu = pit_uniformity(pit)
        stats[name] = pu
        print(f"  {name:11s} {format_pit(pu)}")
    ok = (stats["gen-transf"]["ks_p"] >= 0.05 and stats["gen-transf"]["chi2_p"] >= 0.05)
    no_worse = stats["gen-transf"]["ks"] <= stats["gen-ident"]["ks"] + 0.02
    print(f"\nG7(object) {'PASS' if ok and no_worse else 'CHECK'}: "
          f"transform {'preserves' if no_worse else 'changes'} calibration "
          f"(gen-transf KS={stats['gen-transf']['ks']:.3f} vs gen-ident KS={stats['gen-ident']['ks']:.3f})")


if __name__ == "__main__":
    main()
