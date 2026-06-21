"""Exact BOSS DR12 angular selection function from the source mangle maps.

The shipped LSS randoms are just a Monte-Carlo SAMPLING of the angular selection
  S(RA,Dec) = completeness(RA,Dec) × Π_i [ 1 − inside(veto_i) ]
so a random-based completeness estimate is shot-noise-limited (split-half cover
correlation: 0.89 @ nside256, 0.49 @ 512, 0.06 @ 1024). Evaluating the mangle maps
DIRECTLY gives the exact selection at any resolution, shot-noise-free — and makes us
independent of the shipped randoms (we can generate our own at arbitrary density).

This module is meant to run under a numpy<2 interpreter with `pymangle` (the repo's
anaconda python: ~/.local/share/anaconda3/bin/python3). It exposes `selection(ra,dec)`
and a CLI to rasterise S to a cached HEALPix map the main (numpy-2) pipeline loads.

Maps (data/boss/): completeness `mask_DR12v5_CMASS_South.ply`; vetos in `veto/`:
allsky_bright_star, bright_object_mask_rykoff, centerpost, collision_priority,
badfield_postprocess, badfield_unphot_seeing_extinction (a point INSIDE a veto polygon
is excluded).
"""
import os
import numpy as np

COMPLETENESS = "data/boss/mask_DR12v5_CMASS_South.ply"
VETO_DIR = "data/boss/veto"
VETOS = [
    "centerpost_mask_dr12.ply",
    "bright_object_mask_rykoff_pix.ply",
    "collision_priority_mask_dr12.ply",
    "badfield_mask_postprocess_pixs8.ply",
    "badfield_mask_unphot_seeing_extinction_pixs8_dr12.ply",
    "allsky_bright_star_mask_pix.ply",
]


def load_masks(veto_names=None, verbose=True):
    import pymangle, time
    comp = pymangle.Mangle(COMPLETENESS)
    vetos = {}
    for nm in (veto_names if veto_names is not None else VETOS):
        t = time.time()
        vetos[nm] = pymangle.Mangle(os.path.join(VETO_DIR, nm))
        if verbose:
            print(f"  loaded {nm} ({vetos[nm].npoly:,} polys, {time.time()-t:.0f}s)", flush=True)
    return comp, vetos


def selection(ra, dec, comp, vetos):
    """Exact selection S(ra,dec) = completeness × Π(not inside veto). Vectorised."""
    ra = np.asarray(ra, float); dec = np.asarray(dec, float)
    s = np.asarray(comp.weight(ra, dec), float)
    keep = s > 0
    for nm, v in vetos.items():
        if keep.any():
            inside = v.contains(ra[keep], dec[keep])
            idx = np.flatnonzero(keep)
            s[idx[inside]] = 0.0
            keep[idx[inside]] = False
    return s


def _validate():
    from astropy.io import fits
    print("loading masks ...", flush=True)
    comp, vetos = load_masks()
    with fits.open("data/boss/galaxy_DR12v5_CMASS_South.fits.gz") as h:
        d = h[1].data
        gra = np.asarray(d["RA"], float); gdec = np.asarray(d["DEC"], float)
    print(f"\nvalidating against {len(gra):,} LSS galaxies (should be ~100% unvetoed):")
    s = np.asarray(comp.weight(gra, gdec), float)
    print(f"  completeness>0: {100*np.mean(s>0):.2f}%")
    keep = s > 0
    for nm, v in vetos.items():
        inside = v.contains(gra, gdec)
        print(f"  {nm:52s} vetoes {100*np.mean(inside):.3f}% of galaxies")
        keep &= ~inside
    print(f"  galaxies passing FULL selection: {100*np.mean(keep):.2f}%  (high = vetos correct)")


def evaluate_at(npz_in, npz_out, veto_names=None):
    """Evaluate the exact selection at the (ra,dec) in ``npz_in`` (keys ra,dec[,ipix])
    and write (ipix,sel) to ``npz_out``. The cross-env contract: a numpy-2 + healpy
    process supplies HEALPix pixel centres; this numpy<2 + pymangle process evaluates S.
    """
    d = np.load(npz_in)
    comp, vetos = load_masks(veto_names, verbose=False)
    s = selection(d["ra"], d["dec"], comp, vetos)
    out = {"sel": s.astype(np.float32)}
    if "ipix" in d.files:
        out["ipix"] = d["ipix"]
    np.savez(npz_out, **out)


# Regenerating the cached selection map `data/boss/boss_selection_2048.npz`
# (after a numpy upgrade or for another field) — cross-env recipe:
#   1. (k3d/healpy) write nside=2048 footprint pixel centres -> output/sel_pix2048.npz
#        ipix=hp.query_strip(...); ra,dec=hp.pix2ang(...); np.savez(..., ipix,ra,dec)
#   2. (anaconda/pymangle) python pipeline/boss_selection.py \
#        --evaluate output/sel_pix2048.npz output/sel_val2048.npz
#   3. (k3d) keep sel>0, np.savez_compressed('data/boss/boss_selection_2048.npz', ipix,sel,nside=2048)
#   `echoes.fill_footprint.load_analytic_completeness(nside)` ud_grades it to any nside.
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--validate", action="store_true")
    p.add_argument("--evaluate", nargs=2, metavar=("PIX_NPZ", "OUT_NPZ"),
                   help="evaluate the exact selection at the (ra,dec) in PIX_NPZ -> OUT_NPZ")
    args = p.parse_args()
    if args.validate:
        _validate()
    if args.evaluate:
        evaluate_at(args.evaluate[0], args.evaluate[1])
        print(f"wrote {args.evaluate[1]}")
