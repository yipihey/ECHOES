"""Engine performance in INPAINTING regions — redshift inference where there is
no local spectroscopy (the regime that decides sparse-footprint support).

Interior mask holes have no observed galaxies: a completed galaxy's redshift must
be inferred from the hole *boundary*. As the footprint gets sparser the holes grow
and the inference degrades — the method that degrades most gracefully is the one
that supports sparse surveys. (The production inpaint_holes transplants real donor
patches, a statistical-field product; this test isolates the per-galaxy redshift
inference of the density engines instead.)

Controlled mock: carve circular holes into real-BOSS-truth, remove the interior
galaxies from the observed set, and ask each engine to infer their redshifts from
the surrounding spectroscopy. The data-starvation axis is the angular distance to
the nearest observed galaxy d_nn (≈ depth into the hole). We report the
sampling-free density-quality metric — posterior mass at the truth
P(|z-z_true|<dz_struct) — binned by d_nn, for photo-z-only / field / knn2d, plus
the global-n(z) floor any method reaches when fully data-starved.

    JAX_PLATFORMS=cpu OMP_NUM_THREADS=16 ~/.venv/k3d/bin/python3 \
        validation/knn2d_inpaint.py [--knn2d-bwz 0.005 --knn2d-nz 64]
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy.spatial import cKDTree
from echoes.surveys.boss import load_boss
from echoes.photoz import PhotoZKNN, photoz_features
from echoes.geometry import _radec_to_nhat
from echoes.knn2d_field import build_knn2d_field, _per_sightline_dd, _one_plus_delta_knn

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-holes", type=int, default=300)
    p.add_argument("--radii-deg", default="0.1,0.2,0.35,0.6",
                   help="hole radii to mix (deg)")
    p.add_argument("--dz-struct", type=float, default=0.006)
    p.add_argument("--knn2d-bwz", type=float, default=0.005)
    p.add_argument("--knn2d-nz", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    radii = [float(x) for x in args.radii_deg.split(",")]
    rng = np.random.default_rng(args.seed)

    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    ra = np.asarray(cat.ra_data); dec = np.asarray(cat.dec_data); z = np.asarray(cat.z_data)
    colors = np.asarray(cat.colors_data); mags = np.asarray(cat.mags_data)
    nhat = _radec_to_nhat(ra, dec)
    tree = cKDTree(nhat)

    # carve holes at random galaxy centres, mixed radii; tag interior galaxies.
    centres = rng.choice(len(ra), args.n_holes, replace=False)
    hole_r = rng.choice(radii, args.n_holes)
    in_hole = np.zeros(len(ra), bool)
    for c, R in zip(centres, hole_r):
        chord = 2.0 * np.sin(np.radians(R) / 2.0)
        idx = tree.query_ball_point(nhat[c], chord)
        in_hole[idx] = True
    obs_mask = ~in_hole
    tgt = np.flatnonzero(in_hole)
    print(f"truth N={len(ra):,}  holes={args.n_holes} (radii {radii} deg)  "
          f"observed N={int(obs_mask.sum()):,}  inpaint-targets N={len(tgt):,}")

    ra_o, dec_o, z_o = ra[obs_mask], dec[obs_mask], z[obs_mask]
    col_o, mag_o = colors[obs_mask], mags[obs_mask]

    # photo-z is colour-based (position-independent); train on observed survivors.
    of = photoz_features(col_o, mag_o); ogood = np.isfinite(of).all(1)
    pz = PhotoZKNN(k=100).fit(of[ogood], z_o[ogood])
    feat_t = photoz_features(colors[tgt], mags[tgt])
    zk, wk = pz.posterior(feat_t)
    z_true = z[tgt]

    # distance to nearest OBSERVED galaxy (the data-starvation axis).
    otree = cKDTree(_radec_to_nhat(ra_o, dec_o))
    dchord, _ = otree.query(nhat[tgt], k=1)
    d_nn = np.degrees(2.0 * np.arcsin(np.clip(dchord / 2.0, 0, 1)))

    # build the knn2d field from the holed observed catalog (sel_map from the full
    # footprint — the holes are unmasked, so RD correctly expects data there).
    import types
    obs_cat = types.SimpleNamespace(ra_data=ra_o, dec_data=dec_o, z_data=z_o)
    kf = build_knn2d_field(obs_cat, seed=0, verbose=True, sel_map=cat.sel_map,
                           nside=cat.nside, reduce="knn",
                           bw_z=args.knn2d_bwz, n_z_n=args.knn2d_nz)

    zgrid = np.linspace(z_o.min(), z_o.max(), 256)
    zc = kf.z_n_centres
    nbar = np.interp(zgrid, zc, np.histogram(z_o, bins=kf.z_n_edges)[0].astype(float),
                     left=0.0, right=0.0)
    bw_f, bw_p = 0.004, 0.02
    # photo-z LOS posterior
    PP = np.zeros((len(tgt), zgrid.size))
    for i in range(len(tgt)):
        wi = wk[i]; ok = np.isfinite(wi) & (wi > 0)
        PP[i] = ((wi[ok][None, :] * np.exp(-0.5 * ((zgrid[:, None] - zk[i][ok][None, :]) / bw_p) ** 2)).sum(1)
                 if ok.any() else np.ones_like(zgrid))
    # field pf: KDE of the K=150 nearest observed spec-z (all on the hole boundary)
    K = min(150, len(z_o))
    _, nn = otree.query(nhat[tgt], k=K, workers=-1)
    # knn2d opd along each target sightline
    opd = _one_plus_delta_knn(_per_sightline_dd(kf, ra[tgt], dec[tgt]), kf)
    # global n(z) floor (fully data-starved): pp × nbar only
    def pstruct(pun, i):
        s = pun.sum()
        if s <= 0:
            return np.nan
        return float(pun[np.abs(zgrid - z_true[i]) < args.dz_struct].sum() / s)

    Pm = {"photoz-only": [], "n(z) floor": [], "field (KNN-KDE)": [],
          "knn2d (adaptive)": [], "hybrid (geom)": []}
    # the kth-NN distance of the field-KDE (distance to the K-th neighbour) is a
    # natural data-starvation weight: dense -> trust knn2d, sparse -> trust field.
    dchordK, _ = otree.query(nhat[tgt], k=K, workers=-1)
    theta_K = np.degrees(2.0 * np.arcsin(np.clip(dchordK[:, -1] / 2.0, 0, 1)))  # K-NN radius
    for i in range(len(tgt)):
        pf_field = np.exp(-0.5 * ((zgrid[:, None] - z_o[nn[i]][None, :]) / bw_f) ** 2).sum(1)
        pf_knn = np.interp(zgrid, zc, opd[i], left=0.0, right=0.0) * nbar
        # density-weighted geometric blend: w_knn high where the K-NN radius is
        # small (dense), -> field where the radius is large (sparse / deep in holes)
        w = float(np.clip(0.3 / max(theta_K[i], 1e-3), 0.0, 1.0))   # 0.3deg pivot
        fb = np.maximum(pf_field, 1e-30); kb = np.maximum(pf_knn, 1e-30)
        pf_hyb = np.exp((1 - w) * np.log(fb) + w * np.log(kb))
        Pm["photoz-only"].append(pstruct(PP[i], i))
        Pm["n(z) floor"].append(pstruct(nbar * PP[i], i))
        Pm["field (KNN-KDE)"].append(pstruct(pf_field * PP[i], i))
        Pm["knn2d (adaptive)"].append(pstruct(pf_knn * PP[i], i))
        Pm["hybrid (geom)"].append(pstruct(pf_hyb * PP[i], i))
    for k in Pm:
        Pm[k] = np.array(Pm[k])

    # bin by data-starvation distance.
    edges = np.array([0.0, 0.03, 0.06, 0.12, 0.25, 0.5, 10.0])
    print(f"\n=== posterior mass at truth P(|z-z_true|<{args.dz_struct}) vs distance to "
          f"nearest observed galaxy (median per bin) ===")
    print(f"{'d_nn [deg]':>14s} {'N':>6s} " + "".join(f"{k:>17s}" for k in Pm))
    for b in range(len(edges) - 1):
        m = (d_nn >= edges[b]) & (d_nn < edges[b + 1])
        if m.sum() < 20:
            continue
        lab = f"{edges[b]:.2f}-{edges[b+1]:.2f}" if edges[b+1] < 9 else f">{edges[b]:.2f}"
        row = f"{lab:>14s} {int(m.sum()):6d} "
        row += "".join(f"{np.nanmedian(Pm[k][m]):17.4f}" for k in Pm)
        print(row)
    print("\n(as d_nn grows the holes are larger/sparser; the engine whose posterior mass "
          "at truth stays highest above the n(z) floor degrades most gracefully and best "
          "supports sparse footprints.)")


if __name__ == "__main__":
    main()
