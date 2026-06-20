"""Projected randoms weight/completeness map vs RA/Dec, and a data<->random hole check.

BOSS randoms are generated over the mangle veto mask (bright-star, bad-field,
centerpost holes), so the holes you see in the galaxy footprint should be absent
in the randoms too. This makes a fine (wrapped-RA, Dec) map of the random surface
density (the angular selection function / "weight" — randoms are laid down
proportional to sector completeness) and the FKP-weighted version, next to the
galaxy density, and quantifies whether the holes coincide.

    JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 ~/.venv/k3d/bin/python3 \
        validation/randoms_weight_map.py [--bin 0.04]
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from astropy.io import fits

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"


def _wrap(ra):
    ra = np.asarray(ra, float)
    return np.where(ra > 180.0, ra - 360.0, ra)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bin", type=float, default=0.04, help="pixel size [deg]")
    p.add_argument("--zoom", type=float, nargs=4, default=None,
                   metavar=("RA0", "RA1", "DEC0", "DEC1"), help="zoom-panel box [deg]")
    p.add_argument("--out", default="output/randoms_weight_map.png")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    with fits.open(RAND) as h:
        rd = h[1].data
        rra = _wrap(rd["RA"]); rdec = np.asarray(rd["DEC"], float)
        wfkp = np.asarray(rd["WEIGHT_FKP"], float); isect = np.asarray(rd["ISECT"])
    with fits.open(DATA) as h:
        gd = h[1].data
        gra = _wrap(gd["RA"]); gdec = np.asarray(gd["DEC"], float)
    print(f"randoms {len(rra):,}  galaxies {len(gra):,}")

    # shared grid over the footprint bbox
    lo_ra, hi_ra = np.percentile(rra, [0.05, 99.95])
    lo_dec, hi_dec = rdec.min(), rdec.max()
    b = args.bin
    ra_edges = np.arange(lo_ra - b, hi_ra + b, b)
    dec_edges = np.arange(lo_dec - b, hi_dec + b, b)
    area = b * b * np.cos(np.radians(0.5 * (lo_dec + hi_dec)))   # approx sq deg / pixel

    Rcount = np.histogram2d(rra, rdec, bins=[ra_edges, dec_edges])[0]
    Rweight = np.histogram2d(rra, rdec, bins=[ra_edges, dec_edges], weights=wfkp)[0]
    Dcount = np.histogram2d(gra, gdec, bins=[ra_edges, dec_edges])[0]
    rho_r = Rcount / area                                        # randoms per sq deg

    # footprint = interior of the random coverage (closing fills small holes); a HOLE
    # is an in-footprint pixel with zero randoms.
    from scipy import ndimage
    occ = Rcount > 0
    interior = ndimage.binary_closing(occ, structure=np.ones((9, 9)))   # ~0.36 deg fill
    interior = ndimage.binary_erosion(interior, structure=np.ones((3, 3)))  # drop ragged edge
    holes = interior & ~occ
    print(f"\nfootprint pixels: {interior.sum():,}   hole pixels (in-footprint, 0 randoms): "
          f"{holes.sum():,}  ({100*holes.sum()/max(interior.sum(),1):.1f}% of footprint)")

    # do the holes coincide with the data? galaxies should also avoid them.
    gi = np.clip(np.digitize(gra, ra_edges) - 1, 0, len(ra_edges) - 2)
    gj = np.clip(np.digitize(gdec, dec_edges) - 1, 0, len(dec_edges) - 2)
    in_hole = holes[gi, gj]
    gal_in_holes = int(in_hole.sum())
    # data density inside holes vs in the filled footprint
    fp_no_hole = interior & occ
    print(f"galaxies landing in random-empty holes: {gal_in_holes:,} "
          f"({100*gal_in_holes/len(gra):.3f}% of all galaxies)")
    print(f"mean galaxy density  in footprint: {Dcount[fp_no_hole].mean()/area:.0f} / sq deg")
    print(f"mean galaxy density  in holes    : {Dcount[holes].mean()/area if holes.sum() else 0:.0f} / sq deg")
    print("\n-> randoms and galaxies share the same veto holes (the mask is applied to both)."
          if gal_in_holes / len(gra) < 0.01 else
          "\n-> WARNING: galaxies appear in random-empty pixels; mask mismatch?")

    # ---- figure: projected random weight map (+ smoothed galaxy density) ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle
    extent = [ra_edges[-1], ra_edges[0], dec_edges[0], dec_edges[-1]]   # RA increases left
    aspect = 1.0 / np.cos(np.radians(0.5 * (lo_dec + hi_dec)))

    # galaxy density needs smoothing: at this pixel scale the mean is <1 gal/pixel
    # (pure shot noise), so smooth counts AND occupancy with a ~10' kernel and divide.
    sig = max(1.0, (10.0 / 60.0) / b)                              # ~10 arcmin
    occf = occ.astype(float)
    Dsm = ndimage.gaussian_filter(Dcount, sig) / np.maximum(ndimage.gaussian_filter(occf, sig), 1e-6)
    gal_dens = np.ma.masked_where(~interior.T, (Dsm / area).T)
    rand_dens = np.ma.masked_where(Rcount.T == 0, rho_r.T)

    fig, ax = plt.subplots(2, 2, figsize=(19, 17), facecolor="white",
                           gridspec_kw=dict(height_ratios=[2.3, 1]))
    # (a) random weight map — log, careful vmin so completeness mosaic shows
    rv = rho_r[occ]
    im0 = ax[0, 0].imshow(rand_dens, origin="lower", extent=extent, aspect=aspect, cmap="viridis",
                          norm=LogNorm(vmin=np.percentile(rv, 8), vmax=np.percentile(rv, 99.5)))
    ax[0, 0].set_title(f"Randoms surface density / completeness (log)\n"
                       f"{len(rra):,} randoms, {b*60:.1f}' pixels — holes = mangle vetoes")
    fig.colorbar(im0, ax=ax[0, 0], fraction=0.025, label="randoms / sq deg")
    # (b) galaxy density — smoothed + log so the LSS shows contrast
    gv = (Dsm / area)[interior]; gv = gv[gv > 0]
    im1 = ax[0, 1].imshow(gal_dens, origin="lower", extent=extent, aspect=aspect, cmap="magma",
                          norm=LogNorm(vmin=np.percentile(gv, 5), vmax=np.percentile(gv, 99.5)))
    ax[0, 1].set_title(f"Galaxy surface density (~{sig*b*60:.0f}' smoothing, log)\n"
                       "same footprint + holes as the randoms")
    fig.colorbar(im1, ax=ax[0, 1], fraction=0.025, label="galaxies / sq deg")

    # zoom on a feature-rich patch to expose the survey geometry (stripes/holes/sectors)
    zb = (args.zoom if args.zoom else (-2.0, 8.0, 8.0, 18.0))       # RA0,RA1,Dec0,Dec1
    zext = [zb[1], zb[0], zb[2], zb[3]]
    for a, (dmap, cmap, vlo, vhi, ttl) in zip(
            ax[1], [(rand_dens, "viridis", np.percentile(rv, 8), np.percentile(rv, 99.5),
                     "randoms (zoom): bright-star holes + rectangular field/scan vetoes"),
                    (gal_dens, "magma", np.percentile(gv, 5), np.percentile(gv, 99.5),
                     "galaxies (zoom): same holes, plus tiling-sector completeness mosaic")]):
        a.imshow(dmap, origin="lower", extent=extent, aspect=aspect, cmap=cmap,
                 norm=LogNorm(vmin=vlo, vmax=vhi))
        a.set_xlim(zb[1], zb[0]); a.set_ylim(zb[2], zb[3]); a.set_title(ttl, fontsize=10)
    ax[0, 0].add_patch(Rectangle((zb[1], zb[2]), zb[0] - zb[1], zb[3] - zb[2],
                                 fill=False, ec="red", lw=1.2))

    for a in ax.ravel():
        a.set_xlabel("RA [deg]"); a.set_ylabel("Dec [deg]"); a.set_facecolor("black")
    fig.suptitle("BOSS CMASS-South: randoms weight map vs galaxy density (RA-Dec)", y=0.995)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"\nsaved {args.out}")

    # ---- second figure: tiling-sector completeness (ISECT) + FKP-weighted map ----
    nDec = len(dec_edges) - 1; npix = (len(ra_edges) - 1) * nDec
    ri = np.clip(np.digitize(rra, ra_edges) - 1, 0, len(ra_edges) - 2)
    rj = np.clip(np.digitize(rdec, dec_edges) - 1, 0, nDec - 1)
    pflat = ri * nDec + rj
    # each pixel sits in one mangle SECTOR (ISECT); per-sector random density (randoms
    # per footprint-pixel) is the angular completeness the randoms were laid down with.
    pix_sect = np.full(npix, -1, dtype=np.int64); pix_sect[pflat] = isect
    usect, inv = np.unique(isect, return_inverse=True)
    rand_per = np.bincount(inv, minlength=len(usect)).astype(float)
    occ_pix = np.flatnonzero(occ.ravel())
    idx = np.searchsorted(usect, pix_sect[occ_pix])
    pix_per = np.bincount(idx, minlength=len(usect)).astype(float)
    dens_per = rand_per / np.maximum(pix_per, 1.0)
    big = pix_per >= 5
    comp_norm = np.percentile(dens_per[big], 95)
    cv = float(np.std(dens_per[big]) / np.mean(dens_per[big]))
    comp_map = np.full(npix, np.nan); comp_map[occ_pix] = dens_per[idx] / comp_norm
    comp_img = np.ma.masked_invalid(comp_map.reshape(len(ra_edges) - 1, nDec))
    print(f"\nsectors: {len(usect):,} mangle sectors; per-sector random-density CV = {cv:.3f} "
          f"({'completeness mosaic present' if cv > 0.03 else 'randoms ~uniform over mask'})")

    fkp_dens = np.ma.masked_where(Rcount.T == 0, (Rweight / area).T)

    fig2, ax2 = plt.subplots(1, 2, figsize=(19, 8.5), facecolor="white")
    cv_lo, cv_hi = np.percentile(comp_map[occ_pix], [2, 98])
    im = ax2[0].imshow(comp_img.T, origin="lower", extent=extent, aspect=aspect, cmap="turbo",
                       vmin=cv_lo, vmax=cv_hi)
    ax2[0].set_title("Tiling-sector completeness (per-ISECT random density)\n"
                     f"{len(usect):,} sectors — the BOSS plate-overlap mosaic")
    fig2.colorbar(im, ax=ax2[0], fraction=0.025, label="relative completeness")
    fv = (Rweight / area)[occ]
    im2 = ax2[1].imshow(fkp_dens, origin="lower", extent=extent, aspect=aspect, cmap="cividis",
                        norm=LogNorm(vmin=np.percentile(fv, 8), vmax=np.percentile(fv, 99.5)))
    ax2[1].set_title("FKP-weighted random density (sum WEIGHT_FKP)\n"
                     "FKP is a radial n(z) weight; angular structure tracks the mask")
    fig2.colorbar(im2, ax=ax2[1], fraction=0.025, label="sum WEIGHT_FKP / sq deg")
    for a in ax2:
        a.set_xlabel("RA [deg]"); a.set_ylabel("Dec [deg]"); a.set_facecolor("black")
    fig2.suptitle("BOSS CMASS-South randoms: sector completeness + FKP weighting", y=0.99)
    fig2.tight_layout()
    out2 = args.out.replace(".png", "_sector_fkp.png")
    fig2.savefig(out2, dpi=120, bbox_inches="tight")
    print(f"saved {out2}")


if __name__ == "__main__":
    main()
