"""Inject-and-recover for the generative inpaint: punch synthetic holes into the
real CMASS-South truth, inpaint them, and test that the parent statistics are
recovered — and where recovery degrades with hole size (the data->prior boundary).

This is the M2 acceptance evidence. For a size ladder of holes we remove the galaxies
AND randoms inside, rebuild the fill footprint from the punched randoms, run the thin
inpaint, and per hole-class report:
  * count recovery   N_inpaint_in_hole / N_truth_removed_in_hole  (-> 1 = right density)
  * n(z) recovery    KS(inpainted z, removed-truth z)             (consistent = good)
The recovery ratio falling below ~1 as holes grow locates the prior-dominated regime.

    JAX_PLATFORMS=cpu OMP_NUM_THREADS=16 ~/.venv/k3d/bin/python3 \
        validation/inpaint_recovery.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy.spatial import cKDTree
from scipy import stats
from echoes.surveys.boss import load_boss
from echoes.geometry import _radec_to_nhat
from echoes.mock_systematics import inject_mask_holes
from echoes.fill_footprint import build_fill_footprint
from echoes.inpaint_field import sample_inpaint_catalog

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"
LADDER = {"medium": (0.5, 25), "large": (1.5, 8)}     # resolved at nside>=512; D-core -> P


def _in_holes(ra, dec, c_ra, c_dec, radii):
    """Boolean: which (ra,dec) fall inside any hole (centre+radius)."""
    pts = _radec_to_nhat(np.asarray(ra), np.asarray(dec))
    tree = cKDTree(pts)
    inside = np.zeros(len(np.atleast_1d(ra)), bool)
    cen = _radec_to_nhat(np.asarray(c_ra), np.asarray(c_dec))
    hid = np.full(len(inside), -1)
    for k in range(len(c_ra)):
        chord = 2.0 * np.sin(np.radians(radii[k]) / 2.0)
        idx = tree.query_ball_point(cen[k], chord)
        if idx:
            inside[idx] = True; hid[idx] = k
    return inside, hid


def main():
    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256)
    ra = np.asarray(cat.ra_data); dec = np.asarray(cat.dec_data); z = np.asarray(cat.z_data)
    rra = np.asarray(cat.ra_random); rdec = np.asarray(cat.dec_random)

    # inject holes (defined by galaxy centres), then punch the same holes in randoms
    ht = inject_mask_holes(ra, dec, hole_ladder=LADDER, seed=3)
    g_in = ht.in_hole
    rand_in, _ = _in_holes(rra, rdec, ht.center_ra, ht.center_dec, ht.radius_deg)
    print(f"injected {len(ht.center_ra)} holes; removed {g_in.sum():,} galaxies "
          f"({100*g_in.mean():.1f}%) and {rand_in.sum():,} randoms")

    # survivors = the degraded "observed" truth; build the fill footprint from punched randoms
    surv = ~g_in
    fp = build_fill_footprint(ra_random=rra[~rand_in], dec_random=rdec[~rand_in], z_data=z[surv],
                              nside=512, lss_clip_deg=2.0, mangle_npy=None)
    ip = sample_inpaint_catalog(fp, donor_ra=ra[surv], donor_dec=dec[surv], donor_z=z[surv],
                                rand_ra=rra[~rand_in], rand_dec=rdec[~rand_in], mode="thin", seed=0)
    print(f"inpainted {len(ip['ra']):,} galaxies; mean uncert {ip['uncert'].mean():.2f}")

    # which inpaint / removed-truth galaxies fall in each hole class
    ip_in, ip_hid = _in_holes(ip["ra"], ip["dec"], ht.center_ra, ht.center_dec, ht.radius_deg)
    hole_class = ht.hole_class
    print(f"\n{'hole class':10s} {'radius':>7s} {'N_truth':>9s} {'N_inpaint':>10s} "
          f"{'recovery':>9s} {'n(z) KS p':>10s}")
    for cls in LADDER:
        cmask = hole_class == cls
        hids = np.flatnonzero(cmask)
        n_truth = int((g_in & np.isin(ht.hole_id, hids)).sum())
        sel_ip = np.isin(ip_hid, hids) & ip_in
        n_ip = int(sel_ip.sum())
        z_truth = z[g_in & np.isin(ht.hole_id, hids)]
        z_ip = ip["z"][sel_ip]
        ksp = stats.ks_2samp(z_truth, z_ip).pvalue if (len(z_truth) > 5 and len(z_ip) > 5) else np.nan
        print(f"{cls:10s} {LADDER[cls][0]:6.2f}° {n_truth:9d} {n_ip:10d} "
              f"{n_ip/max(n_truth,1):9.2f} {ksp:10.2f}")

    # overall n(z) recovery
    ksp_all = stats.ks_2samp(z[g_in], ip["z"][ip_in]).pvalue
    rec_all = ip_in.sum() / max(g_in.sum(), 1)
    print(f"\noverall: count recovery {rec_all:.2f}, n(z) KS p={ksp_all:.2f}")
    print("(recovery -> 1 and n(z) KS p>0.05 = parent statistics preserved; recovery falling "
          "below 1 as holes grow marks the data->prior-dominated boundary.)")


if __name__ == "__main__":
    main()
