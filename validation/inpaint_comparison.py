"""Before/after of the generative inpaint: completed catalog WITH vs WITHOUT
filling the veto-mask holes (M1, Regime-D analog fill on real CMASS-South).

Completes the real catalog twice at the same seed (inpaint off / on), reports the
provenance census, and renders the galaxy surface density vs RA/Dec for both so the
holes visibly fill in. Full footprint + a zoom on a hole-rich patch.

    JAX_PLATFORMS=cpu OMP_NUM_THREADS=16 ~/.venv/k3d/bin/python3 \
        validation/inpaint_comparison.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from echoes.surveys.boss import load_boss
from echoes.photoz import PhotoZKNN, photoz_features
from echoes.surveys.boss_targets import load_cmass_targets
from echoes.completion import complete_catalog_photoz, measure_close_pair_dz, PROV_NAME
from echoes.fill_footprint import build_fill_footprint

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"
TARGETS = "data/boss/cmass_targets_South.fits"


def _wrap(ra):
    ra = np.asarray(ra, float)
    return np.where(ra > 180, ra - 360, ra)


def _density(ra, dec, edges_ra, edges_dec, occ, sig):
    from scipy import ndimage
    b = np.histogram2d(_wrap(ra), np.asarray(dec), bins=[edges_ra, edges_dec])[0]
    sm = ndimage.gaussian_filter(b, sig) / np.maximum(ndimage.gaussian_filter(occ, sig), 1e-6)
    return sm


def main():
    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    z = np.asarray(cat.z_data)
    feat = photoz_features(cat.colors_data, cat.mags_data)
    good = np.isfinite(feat).all(1) & (cat.imatch_data == 1)
    pz = PhotoZKNN(k=100).fit(feat[good], z[good])
    dz = measure_close_pair_dz(cat, 62 / 3600.)
    tg = load_cmass_targets(cat, path=TARGETS, seed=0)

    print("building fill footprint ...")
    fp = build_fill_footprint(cat, nside=256, lss_clip_deg=1.0)
    print(f"  fill area {fp.fill_area_deg2:.1f} deg^2, interior holes {len(fp.holes)}")

    c0 = complete_catalog_photoz(cat, tg, pz, seed=0, dz_pool=dz, inpaint=False, verbose=True)
    c1 = complete_catalog_photoz(cat, tg, pz, seed=0, dz_pool=dz, inpaint=True,
                                 fill_footprint=fp, inpaint_mode="thin", verbose=True)

    print("\nprovenance census (inpaint on):")
    for code, n in zip(*np.unique(c1["prov"], return_counts=True)):
        print(f"  {int(code)} {PROV_NAME[int(code)]:9s} {n:7d}")
    n_ip = int((c1["prov"] == 5).sum())
    print(f"\ninpainted galaxies: {n_ip:,}; mean uncert "
          f"{float(np.asarray(c1['uncert'])[np.asarray(c1['prov'])==5].mean()):.2f}")

    # density maps (shared grid + footprint mask from the no-inpaint catalog)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle
    b = 0.04
    ra_e = np.arange(_wrap(c0["ra"]).min() - b, _wrap(c0["ra"]).max() + b, b)
    dec_e = np.arange(np.asarray(c0["dec"]).min() - b, np.asarray(c0["dec"]).max() + b, b)
    occ = (np.histogram2d(_wrap(cat.ra_random), np.asarray(cat.dec_random),
                          bins=[ra_e, dec_e])[0] > 0).astype(float)
    sig = max(1.0, (8.0 / 60.0) / b)
    d0 = _density(c0["ra"], c0["dec"], ra_e, dec_e, occ, sig)
    d1 = _density(c1["ra"], c1["dec"], ra_e, dec_e, occ, sig)
    interior = occ > 0
    extent = [ra_e[-1], ra_e[0], dec_e[0], dec_e[-1]]
    aspect = 1.0 / np.cos(np.radians(0.5 * (dec_e[0] + dec_e[-1])))
    vv = d1.T[interior.T]; vv = vv[vv > 0]
    norm = LogNorm(vmin=np.percentile(vv, 5), vmax=np.percentile(vv, 99.5))
    zb = (-2.0, 8.0, 8.0, 18.0)

    ip = np.asarray(c1["prov"]) == 5
    ip_ra = _wrap(c1["ra"])[ip]; ip_dec = np.asarray(c1["dec"])[ip]

    fig, ax = plt.subplots(2, 2, figsize=(19, 16), facecolor="white",
                           gridspec_kw=dict(height_ratios=[2.2, 1]))
    dm = np.ma.masked_where(~interior.T, d0.T)
    for col in (0, 1):
        for row in (0, 1):
            a = ax[row, col]
            a.imshow(dm, origin="lower", extent=extent, aspect=aspect, cmap="magma", norm=norm)
            if col == 1:                                    # overplot the inpaint galaxies
                a.scatter(ip_ra, ip_dec, s=6, c="cyan", marker="o", linewidths=0, alpha=0.9)
        ax[0, col].add_patch(Rectangle((zb[1], zb[2]), zb[0]-zb[1], zb[3]-zb[2], fill=False, ec="cyan", lw=1.0))
        ax[1, col].set_xlim(zb[1], zb[0]); ax[1, col].set_ylim(zb[2], zb[3])
    ax[0, 0].set_title("completed, NO inpaint (density)")
    ax[0, 1].set_title(f"completed, INPAINTED — {n_ip:,} new galaxies (cyan) fill the interior holes")
    ax[1, 0].set_title("zoom — no inpaint"); ax[1, 1].set_title("zoom — inpaint galaxies (cyan)")
    for a in ax.ravel():
        a.set_xlabel("RA [deg]"); a.set_ylabel("Dec [deg]"); a.set_facecolor("black")
    fig.suptitle("ECHOES generative inpaint (M1, analog, Regime D): interior veto holes filled "
                 "(large rectangular vetoes are Regime P -> M2)", y=0.995)
    fig.tight_layout()
    os.makedirs("output", exist_ok=True)
    fig.savefig("output/inpaint_comparison.png", dpi=120, bbox_inches="tight")
    print("\nsaved output/inpaint_comparison.png")


if __name__ == "__main__":
    main()
