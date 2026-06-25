"""Prototype: sharpen the photo-z posterior to fix the Z-FAIL redshift under-confidence.

pit_breakdown.py found the z-failure (31% of missing) per-object redshift PIT is peaked/under-confident
(KS ~0.20, std 0.184 ≪ uniform 0.289) — the KNN photo-z posterior (inverse-distance-weighted spread of
the k=100 nearest neighbours in colour) is WIDER than the true colour→z scatter, so the true z lands
near the median too often. Worst for FAINT z-fails (KS 0.24, also biased).

Fix: a weight TEMPERATURE β — `w → w^β` (renormalised) concentrates the posterior on the nearest /
most colour-similar neighbours, keeping the full k support (no sparse-support artifact). β=1 is current;
β>1 sharpens. Sweep β, re-measure the z-fail PIT (and faint/bright split); check the COLLIDED arm
(close-pair-dominated) is unaffected.

  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 JAX_PLATFORMS=cpu python validation/pit_photoz_prototype.py
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from echoes.surveys.boss import load_boss
from echoes.photoz import PhotoZKNN, photoz_features
from echoes.completion import measure_close_pair_dz
from echoes.mock_systematics import apply_survey_systematics
from echoes.posterior import build_package
from echoes.pit import pit_uniformity

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"


class SharpPhotoZ:
    """Wrap a PhotoZKNN, sharpening posterior weights by a temperature: w -> w^power (renormalised)."""
    def __init__(self, base, power=1.0):
        self.base = base; self.power = power

    def posterior(self, features):
        zk, wk = self.base.posterior(features)
        finite = np.isfinite(wk)
        w = np.where(finite, wk, 0.0) ** self.power
        s = w.sum(axis=1, keepdims=True)
        w = w / np.where(s > 0, s, 1.0)
        return zk, np.where(finite, w, np.nan)

    def __getattr__(self, name):
        return getattr(self.base, name)


def object_pit(pkg, true_z):
    qlev = np.asarray(pkg["qlev"], float); invcdf = np.asarray(pkg["invcdf"], float)
    return np.array([np.interp(true_z[i], invcdf[i], qlev, left=0.0, right=1.0)
                     for i in range(invcdf.shape[0])])


def shape(pit, n=10):
    h, _ = np.histogram(np.clip(pit, 0, 1), bins=n, range=(0, 1)); h = h / max(h.sum(), 1)
    return ("U/over-conf" if h[0] + h[-1] > 0.25 else
            ("peaked/under-conf" if h[n // 2 - 1] + h[n // 2] > 0.25 else "~flat"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    # the REAL z-fail width knob is the build_package KDE bandwidths (bw_f field, bw_p photo-z),
    # NOT the photo-z weights (washed out by the bandwidths). Sweep a scale on both.
    p.add_argument("--scales", type=float, nargs="+", default=[1.0, 0.7, 0.5, 0.35, 0.25])
    p.add_argument("--knob", choices=["bw", "K"], default="bw",
                   help="bw = scale bw_f/bw_p; K = scale the spatial-neighbour count for the field KDE")
    args = p.parse_args()
    BW_F, BW_P, K0 = 0.004, 0.02, 150        # build_package defaults

    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    ra = np.asarray(cat.ra_data); dec = np.asarray(cat.dec_data); z = np.asarray(cat.z_data)
    feat = photoz_features(cat.colors_data, cat.mags_data)
    good = np.isfinite(feat).all(1) & (np.asarray(cat.imatch_data) == 1)
    pz = PhotoZKNN(k=100).fit(feat[good], z[good])
    obs, tg, kept, true_z = apply_survey_systematics(
        ra, dec, z, cat.colors_data, cat.mags_data, np.asarray(cat.w_sys_data),
        coll_frac=0.6, zfail_frac=0.014, zfail_faint_bias=1.5, seed=args.seed)
    true_z = np.asarray(true_z); miss = np.asarray(tg.miss_kind)
    zf = miss == "zfail"; coll = miss == "collided"
    imag = np.asarray(tg.mags)[:, 3] if tg.mags is not None else np.full(len(true_z), np.nan)
    faint = np.isfinite(imag) & (imag > np.nanmedian(imag))
    dz = measure_close_pair_dz(obs, 62 / 3600.)
    print(f"inject-and-recover: {len(true_z):,} missing ({zf.mean():.0%} zfail)")
    print(f"  knob={args.knob}; uniform PIT std=0.289; zfail was peaked/under-confident "
          f"(std 0.184) at scale=1\n")
    print(f"  {'scale':>5} {'ZFAIL KS':>9} {'std':>6} {'shape':>16} {'zf·faint KS':>12} "
          f"{'zf·bright KS':>13} {'COLLIDED KS':>12} {'ALL KS':>8}")
    for s in args.scales:
        kw = dict(bw_f=BW_F * s, bw_p=BW_P * s) if args.knob == "bw" \
            else dict(K=max(8, int(round(K0 * s))))
        pit = object_pit(build_package(obs, tg, pz, dz_pool=dz, **kw), true_z)
        z_ks = pit_uniformity(pit[zf])
        print(f"  {s:>5.2f} {z_ks['ks']:>9.3f} {z_ks['std']:>6.3f} {shape(pit[zf]):>16} "
              f"{pit_uniformity(pit[zf & faint])['ks']:>12.3f} "
              f"{pit_uniformity(pit[zf & ~faint])['ks']:>13.3f} "
              f"{pit_uniformity(pit[coll])['ks']:>12.3f} {pit_uniformity(pit)['ks']:>8.3f}", flush=True)
    print("\n(narrower bandwidth/K sharpens; zfail std should rise toward 0.289 then overshoot. "
          "COLLIDED KS ~flat = close-pair-dominated.)")


if __name__ == "__main__":
    main()
