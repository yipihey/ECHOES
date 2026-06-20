"""M5 gate: does the generative inpaint preserve 2-point clustering?

Inject synthetic holes into the real CMASS-South truth (punch galaxies AND a copy of
the randoms), inpaint the survivors, and measure w(θ) and wp(rp) against the parent
using the FULL (hole-filled) randoms as the common window:
  * truth   = parent galaxies                      (reference)
  * holey   = survivors only (holes empty)         (the hole imprint -> biased)
  * inpaint = survivors + inpaint galaxies         (thin and cr)
A fill that preserves clustering gives inpaint/truth -> 1, removing the holey bias.

    JAX_PLATFORMS=cpu OMP_NUM_THREADS=16 ~/.venv/k3d/bin/python3 \
        validation/inpaint_clustering.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy.spatial import cKDTree
from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks
from echoes.surveys.boss import load_boss
from echoes.geometry import _radec_to_nhat
from echoes.clustering import wp_rp
from echoes.fill_footprint import build_fill_footprint
from echoes.inpaint_field import sample_inpaint_catalog
from echoes.mock_systematics import inject_mask_holes

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"
NTH = 16
LADDER = {"medium": (0.5, 25), "large": (1.5, 8)}


def wtheta(ra_d, dec_d, ra_r, dec_r, tb, rr=None):
    nd, nr = len(ra_d), len(ra_r)
    dd = DDtheta_mocks(1, NTH, tb, ra_d.astype("f8"), dec_d.astype("f8"))["npairs"].astype(float)
    if rr is None:
        rr = DDtheta_mocks(1, NTH, tb, ra_r.astype("f8"), dec_r.astype("f8"))["npairs"].astype(float)
    dr = DDtheta_mocks(0, NTH, tb, ra_d.astype("f8"), dec_d.astype("f8"),
                       RA2=ra_r.astype("f8"), DEC2=dec_r.astype("f8"))["npairs"].astype(float)
    return np.where(rr > 0, (dd/(nd*(nd-1.)) - 2*dr/(nd*nr) + rr/(nr*(nr-1.)))/(rr/(nr*(nr-1.))), np.nan), rr


def _in_holes(ra, dec, c_ra, c_dec, radii):
    tree = cKDTree(_radec_to_nhat(np.asarray(ra), np.asarray(dec)))
    inside = np.zeros(len(np.atleast_1d(ra)), bool)
    cen = _radec_to_nhat(np.asarray(c_ra), np.asarray(c_dec))
    for k in range(len(c_ra)):
        idx = tree.query_ball_point(cen[k], 2*np.sin(np.radians(radii[k])/2.))
        if idx:
            inside[idx] = True
    return inside


def main():
    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256)
    ra = np.asarray(cat.ra_data); dec = np.asarray(cat.dec_data); z = np.asarray(cat.z_data)
    rng = np.random.default_rng(0)
    ridx = rng.choice(len(cat.ra_random), min(6*len(ra), len(cat.ra_random)), replace=False)
    rra = np.asarray(cat.ra_random)[ridx]; rdec = np.asarray(cat.dec_random)[ridx]; rz = np.asarray(cat.z_random)[ridx]

    ht = inject_mask_holes(ra, dec, hole_ladder=LADDER, seed=3)
    g_in = ht.in_hole
    rand_in = _in_holes(rra, rdec, ht.center_ra, ht.center_dec, ht.radius_deg)   # punched-random copy
    surv = ~g_in
    print(f"removed {g_in.sum():,} galaxies; survivors {surv.sum():,}")

    fp = build_fill_footprint(ra_random=rra[~rand_in], dec_random=rdec[~rand_in], z_data=z[surv],
                              nside=256, lss_clip_deg=2.0, mangle_npy=None)
    from types import SimpleNamespace
    from echoes.fieldpost import build_field_context
    fctx = build_field_context(SimpleNamespace(ra_data=ra[surv], dec_data=dec[surv], z_data=z[surv],
                               sel_map=fp.observed_cover, nside=fp.nside),
                               sel_map=fp.observed_cover, nside=fp.nside, n_rand_factor=2)
    ip = {m: sample_inpaint_catalog(fp, donor_ra=ra[surv], donor_dec=dec[surv], donor_z=z[surv],
                                    rand_ra=rra[~rand_in], rand_dec=rdec[~rand_in], mode=m, seed=0,
                                    field_ctx=(fctx if m == "cr" else None))
          for m in ("thin", "cr")}
    for m in ip:
        print(f"  {m}: +{len(ip[m]['ra']):,} inpaint")

    # ---- w(theta): truth vs holey vs inpaint, common FULL randoms ----
    tb = np.logspace(np.log10(0.03), np.log10(5.0), 12)
    tc = np.sqrt(tb[1:]*tb[:-1])
    w_truth, rr = wtheta(ra, dec, rra, rdec, tb)
    w_holey, _ = wtheta(ra[surv], dec[surv], rra, rdec, tb, rr=rr)
    w_ip = {m: wtheta(np.r_[ra[surv], ip[m]["ra"]], np.r_[dec[surv], ip[m]["dec"]], rra, rdec, tb, rr=rr)[0]
            for m in ip}
    print(f"\n=== w(theta) ratio to truth (1.0 = preserved) ===")
    print(f"{'theta[deg]':>10s} {'holey':>8s} {'thin':>8s} {'cr':>8s}")
    for i in range(len(tc)):
        print(f"{tc[i]:10.3f} {w_holey[i]/w_truth[i]:8.2f} "
              f"{w_ip['thin'][i]/w_truth[i]:8.2f} {w_ip['cr'][i]/w_truth[i]:8.2f}")
    def rms(w): return float(np.sqrt(np.nanmean((w/w_truth - 1.0)**2)))
    print(f"\nRMS |w/truth - 1|:  holey {rms(w_holey):.3f}  thin {rms(w_ip['thin']):.3f}  cr {rms(w_ip['cr']):.3f}")

    # ---- wp(rp): 3D projected (needs z; inpaint carries field-modulated z) ----
    rp = np.logspace(np.log10(0.5), np.log10(40.0), 12)
    wp_t, RR = wp_rp(ra, dec, z, rra, rdec, rz, rp_edges=rp, pimax=40., nthreads=NTH, return_RR=True)
    wp_h = wp_rp(ra[surv], dec[surv], z[surv], rra, rdec, rz, rp_edges=rp, pimax=40., nthreads=NTH, precomp_RR=RR)
    wp_ip = {m: wp_rp(np.r_[ra[surv], ip[m]["ra"]], np.r_[dec[surv], ip[m]["dec"]], np.r_[z[surv], ip[m]["z"]],
                      rra, rdec, rz, rp_edges=rp, pimax=40., nthreads=NTH, precomp_RR=RR) for m in ip}
    print(f"\n=== wp(rp) RMS |ratio-1| over 0.5-40 Mpc/h ===")
    for m, wpv in [("holey", wp_h), ("thin", wp_ip["thin"]), ("cr", wp_ip["cr"])]:
        print(f"  {m:6s} {float(np.sqrt(np.nanmean((wpv/wp_t-1)**2))):.3f}")
    print("\n(PASS = inpaint RMS << holey RMS and -> a few %: the fill removes the hole-induced "
          "clustering bias and preserves 2-point statistics.)")


if __name__ == "__main__":
    main()
