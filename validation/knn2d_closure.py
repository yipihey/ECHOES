"""kNN2D engine closure — does the completed catalog recover its own statistic?

The kNN2D engine builds missing-galaxy redshifts from the 2D angular kNN-CDF
(Yuan, Abel & Wechsler 2024; Banerjee & Abel 2021). The honest closure test is
to re-measure that *same* statistic on the completed catalog and confirm it
recovers the TRUTH — not by construction (the completion never sees the truth),
but as a consequence of placing the missing galaxies on the correct local
density.

Inject-and-recover on real-BOSS-truth: real CMASS-South is TRUTH; we observe an
incomplete mock, complete it with z_mode='knn2d', then measure the joint angular
kNN-CDF

    P_{>=k}(θ; z) = Prob[ a random footprint query in shell z has >= k galaxy
                          neighbours within angular cap θ ]   (k = 1, 2, 4)

on truth / observed / completed with the SAME random queries (RD flavor of
:func:`echoes.knn.joint_knn_cdf`). Recovery = completed ≈ truth while observed
(missing galaxies removed) sits low. This is the kNN2D analogue of the 3D
kNN-CDF higher-order test, tied to the engine's own driving statistic.

    JAX_PLATFORMS=cpu OMP_NUM_THREADS=16 ~/.venv/k3d/bin/python3 \
        validation/knn2d_closure.py
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from echoes.surveys.boss import load_boss
from echoes.photoz import PhotoZKNN, photoz_features
from echoes.completion import complete_catalog_photoz, measure_close_pair_dz, PROV
from echoes.mock_systematics import apply_survey_systematics
from echoes.knn2d_field import build_knn2d_field
from echoes.randoms import make_random_from_selection_function
from echoes.knn import joint_knn_cdf

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"


def knn_cdf_2d(ra_q, dec_q, z_q, ra_g, dec_g, z_g, theta_rad, z_edges, ks):
    """P_{>=k}(θ; z) on the z_q==z_n diagonal: random queries vs galaxies.

    Returns ``P[k, theta, z]`` (k indexes ``ks``). RD flavor — the query is the
    random footprint catalog, the neighbours are the galaxies; no self-pairs."""
    res = joint_knn_cdf(
        np.asarray(ra_q, np.float64), np.asarray(dec_q, np.float64), np.asarray(z_q, np.float64),
        np.asarray(ra_g, np.float64), np.asarray(dec_g, np.float64), np.asarray(z_g, np.float64),
        theta_rad, z_edges, z_edges, k_max=int(max(ks)), flavor="RD",
        nside_lookup=512, diagonal_only=True)
    Nq = res.N_q.astype(np.float64); safe = np.where(Nq > 0, Nq, np.inf)
    P_geq = res.H_geq_k / safe[None, :, None]                 # (theta, z, k)
    return np.stack([P_geq[:, :, k - 1] for k in ks], axis=0)  # (nk, theta, z)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-real", type=int, default=3)
    p.add_argument("--n-rand", type=int, default=120_000)
    p.add_argument("--out", default="output/knn2d_closure.png")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    ra = np.asarray(cat.ra_data); dec = np.asarray(cat.dec_data); z = np.asarray(cat.z_data)
    colors = np.asarray(cat.colors_data); mags = np.asarray(cat.mags_data)
    wsys = np.asarray(cat.w_sys_data)

    obs, tg, kept, true_z = apply_survey_systematics(
        ra, dec, z, colors, mags, wsys, coll_frac=0.6, zfail_frac=0.014, seed=0)
    print(f"truth N={len(ra):,}  observed N={obs.N_data:,}  missing N={tg.N:,}")

    of = photoz_features(obs.colors_data, obs.mags_data); ogood = np.isfinite(of).all(1)
    pz = PhotoZKNN(k=100).fit(of[ogood], np.asarray(obs.z_data)[ogood])
    dz = measure_close_pair_dz(obs, 62 / 3600.)

    field = build_knn2d_field(obs, seed=0, verbose=True, sel_map=cat.sel_map, nside=cat.nside)
    print("[knn2d] completing ...")
    comp = []
    for s in range(args.n_real):
        c = complete_catalog_photoz(obs, tg, pz, seed=s, dz_pool=dz, z_mode="knn2d",
                                    knn2d_field=field)
        m = np.asarray(c["prov"]) != PROV["systot"]
        comp.append((np.asarray(c["ra"])[m], np.asarray(c["dec"])[m], np.asarray(c["z"])[m]))

    # common random footprint queries + the 2D kNN-CDF grid.
    rar, decr, zr = make_random_from_selection_function(
        sel_map=cat.sel_map, n_random=args.n_rand, z_data=z, nside=cat.nside,
        rng=np.random.default_rng(11))
    theta_rad = np.deg2rad(np.geomspace(0.02, 1.2, 12))
    z_edges = np.linspace(z.min(), z.max(), 9)                 # 8 z shells
    ks = (1, 2, 4)

    P_t = knn_cdf_2d(rar, decr, zr, ra, dec, z, theta_rad, z_edges, ks)
    P_o = knn_cdf_2d(rar, decr, zr, np.asarray(obs.ra_data), np.asarray(obs.dec_data),
                     np.asarray(obs.z_data), theta_rad, z_edges, ks)
    P_c = np.mean([knn_cdf_2d(rar, decr, zr, c[0], c[1], c[2], theta_rad, z_edges, ks)
                   for c in comp], axis=0)

    # collapse the z axis (mean over shells) for a clean θ-curve per k.
    thd = np.rad2deg(theta_rad)
    Pt, Po, Pc = P_t.mean(2), P_o.mean(2), P_c.mean(2)         # (nk, theta)
    def closeness(P):
        d = np.abs(P - Pt); m = Pt > 1e-3
        return float(np.nanmedian((d / np.where(m, Pt, np.inf))[m]))
    print("\n=== 2D kNN-CDF P_{>=k}(θ) median |X - truth|/truth (lower=better) ===")
    print(f"  observed (incomplete): {closeness(Po):.3f}")
    print(f"  completed (knn2d):     {closeness(Pc):.3f}")
    for j, k in enumerate(ks):
        print(f"\n  k={k}:  θ[deg]   truth    obs/tru   cmp/tru")
        for i in range(len(thd)):
            if Pt[j, i] > 1e-3:
                print(f"        {thd[i]:7.3f}  {Pt[j,i]:.4f}  {Po[j,i]/Pt[j,i]:7.3f}  {Pc[j,i]/Pt[j,i]:7.3f}")
    print("\n(completed should track truth (~1) where observed sits low — the missing "
          "galaxies restore the kNN-CDF the engine is built from: closure.)")

    # figure
    fig, ax = plt.subplots(1, len(ks), figsize=(5.2 * len(ks), 4.4), squeeze=False)
    for j, k in enumerate(ks):
        a = ax[0, j]
        a.semilogx(thd, Pt[j], "k-", lw=2, label="truth")
        a.semilogx(thd, Po[j], "v--", color="#888", label="observed (incomplete)")
        a.semilogx(thd, Pc[j], "^-", color="#c0392b", lw=2, label="completed (knn2d)")
        a.set_xlabel("θ [deg]"); a.set_ylabel(f"P(>= {k} neigh in cap)")
        a.legend(fontsize=8); a.set_title(f"2D kNN-CDF, k={k}")
    fig.suptitle("kNN2D engine closure — completed catalog recovers the 2D kNN-CDF (Yuan-Abel-Wechsler) of the truth", y=1.02)
    fig.tight_layout(); fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
