"""Build the FULLY-CONTIGUOUS ECHOES product: inpaint every interior hole.

The default release masks interior veto holes; this product paints them in with the
data-driven non-Gaussian field so the catalog has only the survey's OUTER boundary
and no interior holes — what topological / kNN / field-level statistics need.

Writes to ``data_release/``:
  contiguous/inpaint_seed_XXXX.npz   per-seed inpaint galaxies (ra,dec,z,prov=5,uncert)
                                     — the only seed-varying part; the observed+missing
                                     base is the shared posterior package.
  cmass_south_randoms_contiguous.npz uniform randoms over the FILLED footprint (the
                                     matching window; pair with the contiguous catalog).
  contiguous/manifest.json           seeds, counts, footprint area, parameters.

The contiguous catalog for a seed = draw(cmass_south_posterior.npz, seed) + the seed's
inpaint galaxies. (Inpaint count is stochastic per seed, so it cannot live in the
fixed-base inverse-CDF package.)

    OMP_NUM_THREADS=16 JAX_PLATFORMS=cpu python pipeline/build_contiguous_release.py --seeds 0 1 2 3
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import healpy as hp

from echoes.surveys.boss import load_boss
from echoes.fill_footprint import build_fill_footprint
from echoes.generative import build_generative_model
from echoes.inpaint_field import sample_inpaint_catalog
from echoes.randoms import make_random_from_selection_function

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"
OUT = "data_release"
N_RAND_MULT = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--nside", type=int, default=512, help="fill footprint nside (resolves thin stripes/holes)")
    ap.add_argument("--field-nside", type=int, default=128,
                    help="coarse nside for the field modulation (smooth at ~deg; big speedup vs per-pixel)")
    ap.add_argument("--transform", default="lognormal", help="non-Gaussian fill transform")
    args = ap.parse_args()
    cdir = os.path.join(OUT, "contiguous")
    os.makedirs(cdir, exist_ok=True)

    cat = load_boss([DATA], [RAND], sample="CMASS", nside=args.nside, with_photometry=True)
    ra_o = np.asarray(cat.ra_data); dec_o = np.asarray(cat.dec_data); z_o = np.asarray(cat.z_data)
    pixarea = hp.nside2pixarea(args.nside, degrees=True)

    print("building contiguous footprint (all interior holes) ...", flush=True)
    # exact mangle-based completeness (shot-noise-free, random-independent) when the
    # cached selection map is present — required for clean fills at nside>=512.
    fp = build_fill_footprint(cat, nside=args.nside, contiguous=True, analytic_completeness=True)
    fill_deg2 = float((fp.fill_weight > 0).sum() * pixarea)
    tm_deg2 = float(fp.target_mask.sum() * pixarea)
    print(f"  target_mask {tm_deg2:.0f} deg^2, fill (holes) {fill_deg2:.0f} deg^2", flush=True)

    print("building generative model (deconvolved non-Gaussian field) ...", flush=True)
    gm = build_generative_model(cat, transform=args.transform, deconv=True, verbose=False)
    tf = gm.los_transform()
    # completeness scale for the fill amplitude (matches completion.py)
    wc = 1.0
    if cat.w_sys_data is not None and cat.w_cp_data is not None:
        wc = float(np.mean(np.asarray(cat.w_sys_data) *
                           (np.asarray(cat.w_cp_data) + np.asarray(cat.w_noz_data) - 1.0)))

    # ---- filled randoms over the contiguous footprint (the matching window) ----
    rng = np.random.default_rng(0)
    n_rand = N_RAND_MULT * cat.N_data
    rar, decr, zr = make_random_from_selection_function(
        sel_map=fp.target_mask.astype(float), n_random=n_rand, z_data=z_o, nside=args.nside, rng=rng)
    rpath = os.path.join(OUT, "cmass_south_randoms_contiguous.npz")
    np.savez_compressed(rpath, ra=rar.astype(np.float32), dec=decr.astype(np.float32),
                        z=zr.astype(np.float32))
    print(f"  filled randoms: {len(rar):,} -> {os.path.basename(rpath)} "
          f"({os.path.getsize(rpath)/1e6:.2f} MB)", flush=True)

    # ---- per-seed inpaint galaxies (the only seed-varying part) ----
    seed_meta = []
    for s in args.seeds:
        ip = sample_inpaint_catalog(
            fp, donor_ra=ra_o, donor_dec=dec_o, donor_z=z_o,
            rand_ra=np.asarray(cat.ra_random), rand_dec=np.asarray(cat.dec_random),
            donor_mags=getattr(cat, "mags_data", None),    # z-matched photometry for PROV=5
            mode="cr", seed=int(s), density_boost=wc, field_ctx=gm.field_ctx, transform=tf,
            field_nside=args.field_nside)
        p = os.path.join(cdir, f"inpaint_seed_{int(s):04d}.npz")
        extra = {"mags": ip["mags"], "colors": ip["colors"]} if "mags" in ip else {}
        np.savez_compressed(p, ra=ip["ra"], dec=ip["dec"], z=ip["z"],
                            prov=ip["prov"], uncert=ip["uncert"], **extra)
        density = len(ip["ra"]) / max(fill_deg2, 1e-9)
        seed_meta.append({"seed": int(s), "n_inpaint": int(len(ip["ra"])),
                          "inpaint_density_per_deg2": round(density, 1)})
        print(f"  seed {s}: +{len(ip['ra']):,} inpaint ({density:.0f}/deg^2)", flush=True)

    parent_density = len(ra_o) / max(tm_deg2 - fill_deg2, 1e-9)
    manifest = {
        "product": "cmass_south_contiguous",
        "description": "Fully-contiguous BOSS CMASS-South: every interior hole inpainted "
                       "with the data-driven non-Gaussian field; only the outer boundary remains.",
        "base_package": "cmass_south_posterior.npz",
        "randoms": "cmass_south_randoms_contiguous.npz",
        "nside": args.nside, "transform": args.transform,
        "target_mask_deg2": round(tm_deg2, 1), "fill_deg2": round(fill_deg2, 1),
        "parent_density_per_deg2": round(parent_density, 1),
        "seeds": seed_meta,
        "note": "Contiguous catalog(seed) = draw(base_package, seed) + contiguous/inpaint_seed_*.npz. "
                "Pair with cmass_south_randoms_contiguous.npz. PROV=5 inpaint carries an 'uncert' "
                "prior-dominance flag; down-weight uncert>=0.5 for conservative use.",
    }
    with open(os.path.join(cdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote {cdir}/ (inpaint seeds {args.seeds}) + {os.path.basename(rpath)} + manifest.json")
    print(f"parent density {parent_density:.0f}/deg^2; inpaint fills {fill_deg2:.0f} deg^2 of holes.")


if __name__ == "__main__":
    main()
