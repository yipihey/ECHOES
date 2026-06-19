"""Phase 3 — higher-order clustering recovery of the completion (beyond 2-point).

Two-point closure can hide higher-order errors. Using the real-BOSS-truth
inject-and-recover setup (real 1-halo clustering; Patchy is unreliable sub-Mpc),
we test that the completed ensemble recovers the TRUTH for higher-order,
coincidence-sensitive statistics:
  * kNN-CDF — the CDF of the distance from random query points to the k-th nearest
    galaxy (k=1,2,4), a full-hierarchy clustering probe (Banerjee & Abel 2021);
    also directly sensitive to the Δθ=0 duplicate artifact (a 1-NN spike at 0).
  * counts-in-cells PDF — mean, var/mean, skew in fixed apertures.
3-D distances use a fiducial cosmology (measurement-time only).

    PYTHONPATH=/home/tabel/Projects/graphgp:/home/tabel/Projects/graphGP-cosmology \
    OMP_NUM_THREADS=32 JAX_PLATFORMS=cpu ~/.venv/k3d/bin/python3 demos/validate_completion_highorder.py
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from echoes.surveys.boss import load_boss
from echoes.photoz import PhotoZKNN, photoz_features
from echoes.completion import complete_catalog_photoz, measure_close_pair_dz
from echoes.randoms import make_random_from_selection_function
from echoes.clustering import comoving_mpc_h
from echoes.mock_systematics import apply_survey_systematics

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"


def xyz(ra, dec, z):
    d = comoving_mpc_h(z); r = np.radians(ra); dd = np.radians(dec)
    return np.column_stack([d*np.cos(dd)*np.cos(r), d*np.cos(dd)*np.sin(r), d*np.sin(dd)])


def knn_cdf(gal_xyz, q_xyz, ks, redges):
    tree = cKDTree(gal_xyz)
    dist, _ = tree.query(q_xyz, k=max(ks), workers=-1)
    return {k: np.searchsorted(np.sort(dist[:, k-1]), redges) / len(q_xyz) for k in ks}


def cic(gal_xyz, cen_xyz, radius):
    return cKDTree(gal_xyz).query_ball_point(cen_xyz, radius, return_length=True)


def mom(x):
    x = np.asarray(x, float)
    return (x.mean(), x.var() / max(x.mean(), 1e-9),
            ((x - x.mean()) ** 3).mean() / max(x.var(), 1e-9) ** 1.5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-real", type=int, default=6)
    p.add_argument("--z-modes", default="field,knn2d",
                   help="comma list of completion engines to compare head-to-head "
                        "(e.g. 'field,knn2d' or 'field,knn2d,graphgp')")
    p.add_argument("--out", default="output/completion_highorder.png")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    modes = [m.strip() for m in args.z_modes.split(",") if m.strip()]

    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    ra = np.asarray(cat.ra_data); dec = np.asarray(cat.dec_data); z = np.asarray(cat.z_data)
    colors = np.asarray(cat.colors_data); mags = np.asarray(cat.mags_data); wsys = np.asarray(cat.w_sys_data)
    feat = photoz_features(cat.colors_data, cat.mags_data); good = np.isfinite(feat).all(1) & (cat.imatch_data == 1)
    pz = PhotoZKNN(k=100).fit(feat[good], z[good])

    obs, tg, kept, _ = apply_survey_systematics(ra, dec, z, colors, mags, wsys,
                                                coll_frac=0.6, zfail_frac=0.014, zfail_faint_bias=1.5, seed=0)
    dz = measure_close_pair_dz(obs, 62/3600.)

    # per-engine completed ensembles (build the knn2d / graphgp fields once).
    ckw = {}
    if "knn2d" in modes or "knn2d_cdf" in modes:
        from echoes.knn2d_field import build_knn2d_field
    if "knn2d" in modes:
        ckw["knn2d"] = {"knn2d_field": build_knn2d_field(
            obs, seed=0, verbose=True, sel_map=cat.sel_map, nside=cat.nside)}
    if "knn2d_cdf" in modes:
        ckw["knn2d_cdf"] = {"knn2d_field": build_knn2d_field(
            obs, seed=0, verbose=True, sel_map=cat.sel_map, nside=cat.nside,
            weight="cdf")}
    if "graphgp" in modes:
        from echoes.graphgp_field import sample_posterior_density_field
        ckw["graphgp"] = {"gp_field": sample_posterior_density_field(
            obs, n_samples=args.n_real, nside=64, n_z_bins=64,
            r_edges=np.logspace(np.log10(2.0), np.log10(150.0), 28), seed=0, verbose=False)}
    cats_by_mode = {}
    for mode in modes:
        print(f"[{mode}] completing {args.n_real} realizations ...")
        cats_by_mode[mode] = [
            complete_catalog_photoz(obs, tg, pz, seed=s, dz_pool=dz, z_mode=mode,
                                    **ckw.get(mode, {}))
            for s in range(args.n_real)]

    # query points + cell centres from randoms (footprint-uniform)
    rng = np.random.default_rng(3)
    rar, decr, zr = make_random_from_selection_function(sel_map=cat.sel_map, n_random=2*len(ra),
                                                        z_data=z, nside=cat.nside, rng=rng)
    qsel = rng.choice(len(rar), 60000, replace=False)
    q_xyz = xyz(rar[qsel], decr[qsel], zr[qsel])
    csel = rng.choice(len(rar), 8000, replace=False)
    c_xyz = xyz(rar[csel], decr[csel], zr[csel])

    ks = [1, 2, 4]; redges = np.logspace(np.log10(2.0), np.log10(40.0), 30)
    tru_xyz = xyz(ra, dec, z); obs_xyz = xyz(obs.ra_data, obs.dec_data, obs.z_data)
    knn_t = knn_cdf(tru_xyz, q_xyz, ks, redges)
    knn_o = knn_cdf(obs_xyz, q_xyz, ks, redges)

    Rcic = 8.0
    m_t = cic(tru_xyz, c_xyz, Rcic)
    m_o = cic(obs_xyz, c_xyz, Rcic)
    vpf = lambda m: float(np.mean(np.asarray(m) == 0))   # void probability P(N=0)

    # per-engine higher-order statistics.
    knn_mean = {}; m_mean = {}
    for mode, cats in cats_by_mode.items():
        KNN_c = [knn_cdf(xyz(np.asarray(c["ra"]), np.asarray(c["dec"]), np.asarray(c["z"])),
                         q_xyz, ks, redges) for c in cats]
        knn_mean[mode] = {k: np.mean([K[k] for K in KNN_c], 0) for k in ks}
        m_mean[mode] = np.mean([cic(xyz(np.asarray(c["ra"]), np.asarray(c["dec"]),
                                        np.asarray(c["z"])), c_xyz, Rcic) for c in cats], 0)

    # ----- decisive head-to-head report -----
    def dknn(K):  # mean over k of max|ΔCDF| to truth (lower = better)
        return np.mean([np.max(np.abs(K[k] - knn_t[k])) for k in ks])
    print("\n=== kNN-CDF recovery: max|ΔCDF to truth| per k  (lower = better) ===")
    hdr = "  k       observed " + "".join(f"{m:>12s}" for m in modes)
    print(hdr)
    for k in ks:
        row = f"  k={k}   {np.max(np.abs(knn_o[k]-knn_t[k])):9.4f}"
        row += "".join(f"{np.max(np.abs(knn_mean[m][k]-knn_t[k])):12.4f}" for m in modes)
        print(row)
    print("  " + "-" * (len(hdr) - 2))
    print(f"  <k>     {dknn(knn_o):9.4f}" + "".join(f"{dknn(knn_mean[m]):12.4f}" for m in modes))

    print(f"\n=== counts-in-cells (R={Rcic} Mpc/h):  mean, var/mean, skew | VPF=P(N=0) ===")
    print(f"  truth:      {tuple(np.round(mom(m_t),3))}   VPF={vpf(m_t):.4f}")
    print(f"  observed:   {tuple(np.round(mom(m_o),3))}   VPF={vpf(m_o):.4f}")
    for mode in modes:
        print(f"  {mode:10s}: {tuple(np.round(mom(m_mean[mode]),3))}   VPF={vpf(m_mean[mode]):.4f}")
    print("\n(non-Gaussianity lives in skew + VPF + the high-k kNN-CDF tail; the engine that "
          "tracks truth there best is the better higher-order completion.)")

    # ----- figure: kNN-CDF (truth vs engines) + CIC PDF overlay -----
    emk = {"field": ("#3a6ea8", "o"), "knn2d": ("#c0392b", "^"),
           "knn2d_cdf": ("#e8853a", "s"), "graphgp": ("#2e8b57", "d")}
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.7))
    a = ax[0]
    for k in ks:
        a.semilogx(redges, knn_t[k], "k-", lw=1.6, label="truth" if k == ks[0] else None)
        a.semilogx(redges, knn_o[k], color="#888", ls=":", lw=1.2,
                   label="observed" if k == ks[0] else None)
        for mode in modes:
            col, mk = emk.get(mode, ("C0", "o"))
            a.semilogx(redges, knn_mean[mode][k], mk, color=col, ms=3,
                       label=mode if k == ks[0] else None)
    a.set_xlabel("r [Mpc/h]"); a.set_ylabel("kNN-CDF P(<r), k=1,2,4")
    a.legend(fontsize=8); a.set_title("kNN-CDF recovery (truth vs engines)")
    a = ax[1]
    mx = int(max(np.max(m_t), *[np.max(m_mean[m]) for m in modes])); bins = np.arange(0, mx + 2)
    a.hist(m_t, bins=bins, density=True, histtype="step", color="k", lw=2, label="truth")
    a.hist(m_o, bins=bins, density=True, histtype="step", color="#888", lw=1.2, ls=":", label="observed")
    for mode in modes:
        col, _ = emk.get(mode, ("C0", "o"))
        a.hist(m_mean[mode], bins=bins, density=True, histtype="step", color=col, lw=1.8, label=mode)
    a.set_xlabel(f"galaxies in R={Rcic} Mpc/h sphere"); a.set_ylabel("PDF")
    a.legend(fontsize=8); a.set_title("counts-in-cells PDF")
    fig.suptitle("Higher-order recovery head-to-head — " + " vs ".join(modes), y=1.02)
    fig.tight_layout(); fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
