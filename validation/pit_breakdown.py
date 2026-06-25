"""What drives the residual per-object redshift PIT non-uniformity?

object_pit.py shows every engine fails absolute PIT uniformity (KS 0.07-0.18, p≈0) — the "mildly
over-confident" posterior. This decomposes that miscalibration to find the DRIVER: split the PIT by
missing-galaxy kind (collided = fiber-collision partner, governed by the close-pair Δz model; zfail =
redshift failure, governed by the photo-z posterior) and by brightness, and inspect the histogram
SHAPE (U-shaped → over-confident/too-narrow; sloped → biased mean; peaked → under-confident/too-wide).

Real-data inject-and-recover (real CMASS = truth), best engines only (field KNN + gen-ident Gaussian).

  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 JAX_PLATFORMS=cpu python validation/pit_breakdown.py
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
    qlev = np.asarray(pkg["qlev"], float); invcdf = np.asarray(pkg["invcdf"], float)
    return np.array([np.interp(true_z[i], invcdf[i], qlev, left=0.0, right=1.0)
                     for i in range(invcdf.shape[0])])


def hist_shape(pit, n=10):
    h, _ = np.histogram(np.clip(pit, 0, 1), bins=n, range=(0, 1))
    h = h / max(h.sum(), 1)
    edge = h[0] + h[-1]; mid = h[n // 2 - 1] + h[n // 2]      # mass in end bins vs middle bins
    shape = "U/over-confident" if edge > 0.25 else ("peaked/under-confident" if mid > 0.25 else "~flat")
    return h, shape


def report(name, pit, mask=None):
    p = pit if mask is None else pit[mask]
    if len(p) < 20:
        print(f"  {name:24s} (n={len(p)} too few)"); return
    pu = pit_uniformity(p); h, shape = hist_shape(p)
    with np.printoptions(precision=2, suppress=True):
        print(f"  {name:24s} {format_pit(pu)}  [{shape}]  hist={h}")


def main():
    p = argparse.ArgumentParser(); p.add_argument("--seed", type=int, default=0); args = p.parse_args()
    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    ra = np.asarray(cat.ra_data); dec = np.asarray(cat.dec_data); z = np.asarray(cat.z_data)
    feat = photoz_features(cat.colors_data, cat.mags_data)
    good = np.isfinite(feat).all(1) & (np.asarray(cat.imatch_data) == 1)
    pz = PhotoZKNN(k=100).fit(feat[good], z[good])

    obs, tg, kept, true_z = apply_survey_systematics(
        ra, dec, z, cat.colors_data, cat.mags_data, np.asarray(cat.w_sys_data),
        coll_frac=0.6, zfail_frac=0.014, zfail_faint_bias=1.5, seed=args.seed)
    true_z = np.asarray(true_z)
    miss = np.asarray(tg.miss_kind)
    # brightness split: i-band mag if available (faint = redshift-failure-prone)
    imag = np.asarray(tg.mags)[:, 3] if tg.mags is not None else np.full(len(true_z), np.nan)
    faint = np.isfinite(imag) & (imag > np.nanmedian(imag))
    dz = measure_close_pair_dz(obs, 62/3600.)
    print(f"inject-and-recover: {len(true_z):,} missing  "
          f"(collided {np.mean(miss=='collided'):.0%}, zfail {np.mean(miss=='zfail'):.0%})")

    fctx = build_field_context(obs, seed=args.seed, n_samples=1, sel_map=cat.sel_map, nside=cat.nside)
    gm_id = build_generative_model(obs, transform="identity", field_ctx=fctx)
    pkgs = {
        "field(KNN)": build_package(obs, tg, pz, dz_pool=dz),
        "gen-ident(Gauss)": build_package_generative(obs, tg, pz, gm_id, dz_pool=dz),
    }
    for ename, pkg in pkgs.items():
        pit = object_pit(pkg, true_z)
        print(f"\n=== {ename} ===  (PIT shape: U=over-confident, sloped=biased mean, peaked=too-wide)")
        report("ALL", pit)
        report("collided (close-pair)", pit, miss == "collided")
        report("zfail (photo-z)", pit, miss == "zfail")
        report("zfail · faint", pit, (miss == "zfail") & faint)
        report("zfail · bright", pit, (miss == "zfail") & ~faint)


if __name__ == "__main__":
    main()
