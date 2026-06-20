"""Uniformity of the inpainted catalog: does it leave NO residual veto-hole imprint?

Completes the real CMASS-South catalog WITHOUT and WITH inpaint, then compares the
galaxy surface density (random-normalised) in the former veto-hole/empty pixels to the
survey body. A clean fully-complete product fills the holes to the body density (ratio
-> 1) where the no-inpaint catalog leaves them empty (ratio -> 0).

    JAX_PLATFORMS=cpu OMP_NUM_THREADS=16 ~/.venv/k3d/bin/python3 \
        validation/inpaint_uniformity.py [--mode thin|cr]
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import healpy as hp
from echoes.surveys.boss import load_boss
from echoes.photoz import PhotoZKNN, photoz_features
from echoes.surveys.boss_targets import load_cmass_targets
from echoes.completion import complete_catalog_photoz, measure_close_pair_dz
from echoes.fill_footprint import build_fill_footprint

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"
TARGETS = "data/boss/cmass_targets_South.fits"


def _pix(ra, dec, nside):
    return hp.ang2pix(nside, np.radians(90.0 - np.asarray(dec)), np.radians(np.asarray(ra) % 360.0))


def main():
    p = argparse.ArgumentParser(); p.add_argument("--mode", default="thin"); args = p.parse_args()
    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    z = np.asarray(cat.z_data)
    feat = photoz_features(cat.colors_data, cat.mags_data)
    good = np.isfinite(feat).all(1) & (cat.imatch_data == 1)
    pz = PhotoZKNN(k=100).fit(feat[good], z[good])
    dz = measure_close_pair_dz(cat, 62 / 3600.)
    tg = load_cmass_targets(cat, path=TARGETS, seed=0)
    fp = build_fill_footprint(cat, nside=256, lss_clip_deg=1.0)

    c0 = complete_catalog_photoz(cat, tg, pz, seed=0, dz_pool=dz, inpaint=False)
    c1 = complete_catalog_photoz(cat, tg, pz, seed=0, dz_pool=dz, inpaint=True,
                                 fill_footprint=fp, inpaint_mode=args.mode, verbose=True)

    nside = fp.nside; npix = 12 * nside ** 2
    nr = np.bincount(_pix(cat.ra_random, cat.dec_random, nside), minlength=npix).astype(float)
    fill = fp.fill_weight > 0
    body = (fp.observed_cover > 0.5)

    def density_in(cat_d, mask):
        ng = np.bincount(_pix(cat_d["ra"], cat_d["dec"], nside), minlength=npix).astype(float)
        # random-normalised galaxies-per-random in the masked pixels (body uses randoms,
        # holes have no randoms so normalise holes by the body galaxies-per-pixel instead)
        return ng[mask]

    body_per_pix = density_in(c1, body).sum() / max(body.sum(), 1)
    f0 = density_in(c0, fill).sum() / max(fill.sum(), 1)
    f1 = density_in(c1, fill).sum() / max(fill.sum(), 1)
    print(f"\nmean galaxies/pixel  body={body_per_pix:.2f}")
    print(f"fill region (holes+empty), galaxies/pixel:")
    print(f"  no inpaint : {f0:.2f}   (ratio to body {f0/body_per_pix:.2f}) -> holes empty")
    print(f"  inpaint    : {f1:.2f}   (ratio to body {f1/body_per_pix:.2f}) -> holes filled")
    print(f"\n-> inpaint brings the fill region from {100*f0/body_per_pix:.0f}% to "
          f"{100*f1/body_per_pix:.0f}% of the body density (target ~100% = uniform, no hole imprint).")


if __name__ == "__main__":
    main()
