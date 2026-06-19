"""Per-galaxy redshift accuracy of the completion engines — does a better local
density pick a redshift closer to the truth, within the broad photo-z window?

Catalog-averaged summary statistics (wp, kNN-CDF, CIC) are dominated by the
correctly-placed bulk and by the fiber-collision galaxies, which are localized by
their close-pair host (z_host ≈ z_true) almost regardless of the density. The
density does its real work on the **redshift-failure** galaxies, which have only
photo-z + the local field. This test measures the thing that actually probes the
density quality: the per-missing-galaxy redshift error |z_assign − z_true|, split
by missing kind, against the input photo-z width (so we can see there is room for
the density to help) and against the photo-z-only and host baselines.

Inject-and-recover on real-BOSS-truth: real CMASS-South is truth; the missing
galaxies keep their REAL colours (→ a realistic colour photo-z posterior) and the
observed set is a real spec-z subset (→ a realistic spectroscopic catalog). We
complete with photo-z-only, the KNN-KDE 'field' engine, and the adaptive 2D-kNN
'knn2d' engine, and compare:
  * sigma_photoz  — the input photo-z posterior width (is the window broad?)
  * RMS |Δz|, median |Δz|
  * f_on-structure = P(|Δz| < dz_struct)   (landed on the right structure)
  * f_catastrophic = P(|Δz| > dz_cat)
all split into collided (host-localized) vs zfail (density-localized).

    JAX_PLATFORMS=cpu OMP_NUM_THREADS=16 ~/.venv/k3d/bin/python3 \
        validation/knn2d_zaccuracy.py [--n-real 8] [--zfail-frac 0.03]
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from echoes.surveys.boss import load_boss
from echoes.photoz import PhotoZKNN, photoz_features
from echoes.completion import complete_catalog_photoz, measure_close_pair_dz
from echoes.mock_systematics import apply_survey_systematics

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-real", type=int, default=8)
    p.add_argument("--coll-frac", type=float, default=0.6)
    p.add_argument("--zfail-frac", type=float, default=0.03)
    p.add_argument("--dz-struct", type=float, default=0.006, help="'on-structure' |Δz| (~6 Mpc/h)")
    p.add_argument("--dz-cat", type=float, default=0.04, help="catastrophic |Δz|")
    p.add_argument("--knn2d-bwz", type=float, default=0.008, help="knn2d z-smoothing bandwidth")
    p.add_argument("--knn2d-nz", type=int, default=48, help="knn2d number of z shells")
    args = p.parse_args()

    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    ra = np.asarray(cat.ra_data); dec = np.asarray(cat.dec_data); z = np.asarray(cat.z_data)
    colors = np.asarray(cat.colors_data); mags = np.asarray(cat.mags_data)
    wsys = np.asarray(cat.w_sys_data)

    obs, tg, kept, true_z = apply_survey_systematics(
        ra, dec, z, colors, mags, wsys, coll_frac=args.coll_frac,
        zfail_frac=args.zfail_frac, zfail_faint_bias=1.5, seed=0)
    kind = np.asarray(tg.miss_kind)
    print(f"truth N={len(ra):,}  observed N={obs.N_data:,}  missing N={tg.N:,} "
          f"({int((kind=='collided').sum()):,} collided + {int((kind=='zfail').sum()):,} zfail)")

    of = photoz_features(obs.colors_data, obs.mags_data); ogood = np.isfinite(of).all(1)
    pz = PhotoZKNN(k=100).fit(of[ogood], np.asarray(obs.z_data)[ogood])
    dz = measure_close_pair_dz(obs, 62 / 3600.)

    # input photo-z posterior width for the missing galaxies (the window the
    # density has to work within).
    feat_m = photoz_features(tg.colors, tg.mags)
    zk, wk = pz.posterior(feat_m)
    w = np.where(np.isfinite(wk) & (wk > 0), wk, 0.0)
    wsum = np.where(w.sum(1) > 0, w.sum(1), 1.0)
    zmean = (w * zk).sum(1) / wsum
    zvar = (w * (zk - zmean[:, None]) ** 2).sum(1) / wsum
    sig_pz = np.sqrt(np.maximum(zvar, 0))
    print(f"input photo-z posterior sigma_z: median {np.median(sig_pz):.4f} "
          f"(collided {np.median(sig_pz[kind=='collided']):.4f}, "
          f"zfail {np.median(sig_pz[kind=='zfail']):.4f}); "
          f"photo-z |zmode - ztrue| median {np.median(np.abs(zmean - true_z)):.4f}")

    from echoes.knn2d_field import build_knn2d_field
    knn_field = build_knn2d_field(obs, seed=0, verbose=True, sel_map=cat.sel_map,
                                  nside=cat.nside, reduce="knn",
                                  bw_z=args.knn2d_bwz, n_z_n=args.knn2d_nz)
    N_obs = obs.N_data
    engines = {
        "photoz-only": dict(z_mode="photoz"),
        "nn (host)": dict(z_mode="nn"),
        "field (KNN-KDE)": dict(z_mode="field"),
        "knn2d (adaptive)": dict(z_mode="knn2d", knn2d_field=knn_field),
    }

    def assign(mode_kw, seed):
        out = complete_catalog_photoz(obs, tg, pz, seed=seed, dz_pool=dz, **mode_kw)
        return np.asarray(out["z"])[N_obs:N_obs + tg.N]   # z_assign aligned to targets

    print(f"\nPer-galaxy redshift accuracy (pooled over {args.n_real} draws; "
          f"on-struct |Δz|<{args.dz_struct}, catastrophic |Δz|>{args.dz_cat}):")
    print(f"{'engine':18s} {'subset':9s} {'RMS|Δz|':>8s} {'med|Δz|':>8s} "
          f"{'on-struct':>10s} {'catastr':>8s}")
    results = {}
    for name, kw in engines.items():
        dzs = np.concatenate([assign(kw, s) - true_z for s in range(args.n_real)])
        kk = np.tile(kind, args.n_real)
        results[name] = {}
        for sub in ("collided", "zfail", "all"):
            m = (kk == sub) if sub != "all" else np.ones(len(kk), bool)
            d = dzs[m]
            row = dict(rms=float(np.sqrt(np.mean(d ** 2))),
                       med=float(np.median(np.abs(d))),
                       onstruct=float(np.mean(np.abs(d) < args.dz_struct)),
                       cat=float(np.mean(np.abs(d) > args.dz_cat)))
            results[name][sub] = row
            print(f"{name:18s} {sub:9s} {row['rms']:8.4f} {row['med']:8.4f} "
                  f"{row['onstruct']*100:9.1f}% {row['cat']*100:7.1f}%")
        print()

    # the decisive comparison on the density-localized subset.
    print("=== zfail subset (density does the work) — Δ vs photo-z-only ===")
    base = results["photoz-only"]["zfail"]
    for name in ("field (KNN-KDE)", "knn2d (adaptive)"):
        r = results[name]["zfail"]
        print(f"  {name:18s}: on-structure {r['onstruct']*100:.1f}% "
              f"(photoz {base['onstruct']*100:.1f}%, Δ {100*(r['onstruct']-base['onstruct']):+.1f}pp)  "
              f"catastrophic {r['cat']*100:.1f}% (photoz {base['cat']*100:.1f}%)")
    f_kn = results["knn2d (adaptive)"]["zfail"]["onstruct"]
    f_fl = results["field (KNN-KDE)"]["zfail"]["onstruct"]
    print(f"\n  knn2d vs field on-structure (zfail): {100*f_kn:.1f}% vs {100*f_fl:.1f}% "
          f"(Δ {100*(f_kn-f_fl):+.2f}pp)")
    print("\n(if the photo-z window is broad and the density helps, on-structure should "
          "rise photoz < field ≤ knn2d on the zfail subset; collided is host-localized.)")

    # ---- sampling-free density-quality metric: posterior mass at the truth ----
    # Each draw scatters across p(z); the cleaner probe of "does the density
    # concentrate probability at z_true" is the posterior mass within dz_struct of
    # the truth, P_struct = ∫_{|z-z_true|<dz} p(z) dz, for photoz / field / knn2d.
    # A better local density raises P_struct even if a single sampled draw does not.
    from scipy.spatial import cKDTree
    from echoes.geometry import _radec_to_nhat
    from echoes.completion import _clpair_density
    from echoes.knn2d_field import _per_sightline_dd, _one_plus_delta_knn
    ra_o = np.asarray(obs.ra_data); dec_o = np.asarray(obs.dec_data); z_o = np.asarray(obs.z_data)
    z_host = np.where(np.asarray(tg.host_index) >= 0,
                      z_o[np.clip(np.asarray(tg.host_index), 0, len(z_o) - 1)], np.nan)
    coll = (kind == "collided") & (np.asarray(tg.host_index) >= 0)
    pcl = _clpair_density(dz)
    zgrid = np.linspace(z_o.min(), z_o.max(), 256)
    zc = knn_field.z_n_centres
    nbar = np.interp(zgrid, zc, np.histogram(z_o, bins=knn_field.z_n_edges)[0].astype(float),
                     left=0.0, right=0.0)
    bw_f, bw_p = 0.004, 0.02
    # photo-z LOS posterior pp(z) per target (shared by all engines)
    PP = np.zeros((tg.N, zgrid.size))
    for i in range(tg.N):
        wi = wk[i]; ok = np.isfinite(wi) & (wi > 0)
        PP[i] = ((wi[ok][None, :] * np.exp(-0.5 * ((zgrid[:, None] - zk[i][ok][None, :]) / bw_p) ** 2)).sum(1)
                 if ok.any() else np.ones_like(zgrid))
    # field pf: KDE of the K=150 nearest observed spec-z along the sightline
    K = min(150, len(z_o))
    _, nn = cKDTree(_radec_to_nhat(ra_o, dec_o)).query(
        _radec_to_nhat(np.asarray(tg.ra), np.asarray(tg.dec)), k=K, workers=-1)
    # knn2d opd: adaptive kth-NN overdensity along the sightline
    opd = _one_plus_delta_knn(_per_sightline_dd(knn_field, np.asarray(tg.ra), np.asarray(tg.dec)), knn_field)

    def pstruct(p_unnorm, i):
        p = p_unnorm.copy()
        if coll[i]:
            p = p * pcl(zgrid - z_host[i])
        s = p.sum()
        if s <= 0:
            return np.nan
        return float(p[np.abs(zgrid - true_z[i]) < args.dz_struct].sum() / s)

    Pm = {"photoz-only": [], "field (KNN-KDE)": [], "knn2d (adaptive)": []}
    for i in range(tg.N):
        pf_field = np.exp(-0.5 * ((zgrid[:, None] - z_o[nn[i]][None, :]) / bw_f) ** 2).sum(1)
        pf_knn = np.interp(zgrid, zc, opd[i], left=0.0, right=0.0) * nbar
        Pm["photoz-only"].append(pstruct(PP[i], i))
        Pm["field (KNN-KDE)"].append(pstruct(pf_field * PP[i], i))
        Pm["knn2d (adaptive)"].append(pstruct(pf_knn * PP[i], i))
    print(f"\n=== posterior mass at truth P(|z-z_true|<{args.dz_struct})  (median; higher = density "
          f"concentrates probability at the truth, free of sampling scatter) ===")
    print(f"{'engine':18s} {'collided':>10s} {'zfail':>10s} {'all':>10s}")
    for name, vals in Pm.items():
        v = np.array(vals);
        def med(mask): return float(np.nanmedian(v[mask])) if mask.any() else float("nan")
        print(f"{name:18s} {med(kind=='collided'):10.4f} {med(kind=='zfail'):10.4f} {med(np.ones(len(v),bool)):10.4f}")
    print("\n(uniform-in-window baseline ≈ 2·dz/Δz_window ≈ %.3f; the gain over photo-z is the "
          "density's contribution, separated from the single-draw scatter.)"
          % (2 * args.dz_struct / (z_o.max() - z_o.min())))


if __name__ == "__main__":
    main()
