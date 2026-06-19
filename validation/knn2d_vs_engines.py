"""Experimental kNN2D engine — 3-way truth recovery vs KNN-KDE and graphGP.

Inject-and-recover on real-BOSS-truth (the stringency bar the other two engines
clear). The full real CMASS-South is TRUTH; we inject extra fiber collisions +
redshift failures + imaging thinning, complete the mock-observed catalog with
ALL THREE redshift engines, and compare the recovered wp(rp), ξ₀(s) and n(z) to
the TRUTH (and to an oracle that places each missing galaxy at its true z — the
floor any z-assignment can reach). The completion never sees the truth, so
matching it is a real test, not closure-by-construction.

The kNN2D engine (Yuan, Abel & Wechsler 2024) builds the line-of-sight density
from the 2D angular kNN statistic DD(n̂;θ,z)/RD(θ,z); this script asks whether
that recovers the truth as faithfully as the KNN-KDE 'field' and graphGP engines.

    JAX_PLATFORMS=cpu OMP_NUM_THREADS=16 ~/.venv/k3d/bin/python3 \
        validation/knn2d_vs_engines.py [--with-graphgp]
"""
import argparse, dataclasses, os, sys
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
from echoes.clustering import wp_rp, xi_smu_multipoles

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"


def strip_systot(c):
    m = np.asarray(c["prov"]) != PROV["systot"]
    return {"ra": np.asarray(c["ra"])[m], "dec": np.asarray(c["dec"])[m],
            "z": np.asarray(c["z"])[m]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-real", type=int, default=4)
    p.add_argument("--with-graphgp", action="store_true",
                   help="also run the (slower) graphGP engine")
    p.add_argument("--nside", type=int, default=64)
    p.add_argument("--nz", type=int, default=64)
    p.add_argument("--out", default="output/knn2d_vs_engines.png")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    ra = np.asarray(cat.ra_data); dec = np.asarray(cat.dec_data); z = np.asarray(cat.z_data)
    colors = np.asarray(cat.colors_data); mags = np.asarray(cat.mags_data)
    wsys = np.asarray(cat.w_sys_data)

    # inject systematics: real CMASS = TRUTH, observe an incomplete mock.
    obs, tg, kept, true_z = apply_survey_systematics(
        ra, dec, z, colors, mags, wsys, coll_frac=0.6, zfail_frac=0.014, seed=0)
    print(f"truth N={len(ra):,}  observed N={obs.N_data:,}  missing N={tg.N:,}")

    of = photoz_features(obs.colors_data, obs.mags_data); ogood = np.isfinite(of).all(1)
    pz = PhotoZKNN(k=100).fit(of[ogood], np.asarray(obs.z_data)[ogood])
    dz = measure_close_pair_dz(obs, 62 / 3600.)

    engines = {}

    # KNN-KDE 'field'
    print("[field] completing ...")
    engines["KNN-KDE (field)"] = [
        strip_systot(complete_catalog_photoz(obs, tg, pz, seed=s, dz_pool=dz, z_mode="field"))
        for s in range(args.n_real)]

    # kNN2D (experimental) — RD measured on the MOCK-OBSERVED catalog
    print("[knn2d] building field + completing ...")
    field = build_knn2d_field(obs, seed=0, verbose=True,
                              sel_map=cat.sel_map, nside=cat.nside)
    engines["kNN2D (Yuan-Abel-Wechsler)"] = [
        strip_systot(complete_catalog_photoz(obs, tg, pz, seed=s, dz_pool=dz,
                                             z_mode="knn2d", knn2d_field=field))
        for s in range(args.n_real)]

    # graphGP (optional, slower)
    if args.with_graphgp:
        from echoes.graphgp_field import sample_posterior_density_field
        from validation.graphgp_vs_knn import graphgp_catalogs
        km = kept
        mock_cat = dataclasses.replace(
            cat, ra_data=ra[km], dec_data=dec[km], z_data=z[km],
            xyz_data=np.asarray(cat.xyz_data)[km], w_data=np.ones(int(km.sum())),
            w_sys_data=None, w_cp_data=None, w_noz_data=None, w_fkp_data=None,
            colors_data=None, mags_data=None, colors_finite=None,
            imatch_data=None, icollided_data=None)
        print("[graphgp] building conditional field + completing ...")
        res = sample_posterior_density_field(
            mock_cat, n_samples=args.n_real, nside=args.nside, n_z_bins=args.nz,
            r_edges=np.logspace(np.log10(2.0), np.log10(150.0), 28), seed=0, verbose=False)
        engines["graphGP"] = graphgp_catalogs(res, mock_cat, tg, pz, dz, args.n_real,
                                              np.random.default_rng(3))

    # references: truth / observed / oracle (true z).
    truth = {"ra": ra, "dec": dec, "z": z}
    observed = {"ra": np.asarray(obs.ra_data), "dec": np.asarray(obs.dec_data),
                "z": np.asarray(obs.z_data)}
    oracle = {"ra": np.concatenate([observed["ra"], np.asarray(tg.ra)]),
              "dec": np.concatenate([observed["dec"], np.asarray(tg.dec)]),
              "z": np.concatenate([observed["z"], true_z])}

    rar, decr, zr = make_random_from_selection_function(
        sel_map=cat.sel_map, n_random=3 * cat.N_data, z_data=z, nside=cat.nside,
        rng=np.random.default_rng(7))
    rpe = np.logspace(np.log10(0.5), np.log10(40.0), 13); rpc = np.sqrt(rpe[1:] * rpe[:-1])
    se = np.logspace(np.log10(1.0), np.log10(40.0), 11); sc = np.sqrt(se[1:] * se[:-1])
    wp_t, RR = wp_rp(ra, dec, z, rar, decr, zr, rp_edges=rpe, nthreads=16, return_RR=True)
    _, x0_t, _, RRs = xi_smu_multipoles(ra, dec, z, rar, decr, zr, s_edges=se,
                                        nthreads=16, return_RR=True)

    def wp1(c): return wp_rp(np.asarray(c["ra"]), np.asarray(c["dec"]), np.asarray(c["z"]),
                            rar, decr, zr, rp_edges=rpe, nthreads=16, precomp_RR=RR)
    def x01(c): return xi_smu_multipoles(np.asarray(c["ra"]), np.asarray(c["dec"]),
                                        np.asarray(c["z"]), rar, decr, zr, s_edges=se,
                                        nthreads=16, precomp_RR=RRs)[1]
    def wpe(cc): return np.mean([wp1(c) for c in cc], 0)
    def x0e(cc): return np.mean([x01(c) for c in cc], 0)

    wp_o = wp1(observed); wp_or = wp1(oracle)
    print("\n=== wp(rp) recovered / TRUTH  (median | range) ===")
    def summ(name, w):
        r = w / wp_t
        print(f"  {name:30s} median {np.median(r):.3f}  range {r.min():.3f}-{r.max():.3f}")
    summ("observed (incomplete)", wp_o)
    summ("oracle (true z) — floor", wp_or)
    wp_eng = {}
    for name, cc in engines.items():
        wp_eng[name] = wpe(cc); summ(name, wp_eng[name])

    ok = np.abs(x0_t) > 1e-6
    print("\n=== xi_0(s) recovered / TRUTH  (median over |xi|>0) ===")
    for name, cc in engines.items():
        r = (x0e(cc) / x0_t)[ok]; print(f"  {name:30s} median {np.median(r):.3f}")

    zb = np.linspace(z.min(), z.max(), 40); zc = 0.5 * (zb[1:] + zb[:-1])
    nz_t = np.histogram(z, zb, density=True)[0]
    print("\n=== n(z) max |completed/truth - 1| ===")
    nz_eng = {}
    for name, cc in engines.items():
        nz_eng[name] = np.mean([np.histogram(c["z"], zb, density=True)[0] for c in cc], 0)
        print(f"  {name:30s} {np.nanmax(np.abs(nz_eng[name] / nz_t - 1)):.3f}")
    print("\n(closest-to-1 across rp/s = more faithful to truth; oracle is the z-assignment floor.)")

    # figure: wp ratio | xi0 ratio | n(z)
    colors_map = {"KNN-KDE (field)": "#3a6ea8", "kNN2D (Yuan-Abel-Wechsler)": "#c0392b",
                  "graphGP": "#2e8b57"}
    mark = {"KNN-KDE (field)": "o", "kNN2D (Yuan-Abel-Wechsler)": "^", "graphGP": "d"}
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))
    a = ax[0]
    a.axhline(1, color="k", lw=1); a.fill_between(rpc, 0.98, 1.02, color="green", alpha=0.08, label="±2%")
    a.semilogx(rpc, wp_o / wp_t, "v-", color="#888", label="observed (incomplete)")
    a.semilogx(rpc, wp_or / wp_t, "--", color="#444", label="oracle (true z) — floor")
    for name, w in wp_eng.items():
        a.semilogx(rpc, w / wp_t, mark.get(name, "o") + "-", color=colors_map.get(name, "C0"),
                   lw=2, label=name)
    a.set_ylim(0.9, 1.08); a.set_xlabel("rp [Mpc/h]"); a.set_ylabel("wp(rp) recovered / TRUTH")
    a.legend(fontsize=8); a.set_title("projected wp(rp) truth recovery")
    a = ax[1]
    a.axhline(1, color="k", lw=1); a.fill_between(sc, 0.95, 1.05, color="green", alpha=0.08)
    for name, cc in engines.items():
        a.semilogx(sc, x0e(cc) / x0_t, mark.get(name, "o") + "-", color=colors_map.get(name, "C0"), lw=2, label=name)
    a.set_ylim(0.8, 1.2); a.set_xlabel("s [Mpc/h]"); a.set_ylabel("xi_0(s) recovered / TRUTH")
    a.legend(fontsize=8); a.set_title("monopole ξ₀(s) truth recovery")
    a = ax[2]
    a.step(zc, nz_t, where="mid", color="k", lw=2, label="truth")
    for name, nz in nz_eng.items():
        a.step(zc, nz, where="mid", color=colors_map.get(name, "C0"), lw=1.6, label=name)
    a.set_xlabel("redshift z"); a.set_ylabel("n(z)"); a.legend(fontsize=8); a.set_title("n(z)")
    fig.suptitle("Experimental kNN2D engine vs KNN-KDE / graphGP — inject-and-recover on real-BOSS-truth", y=1.02)
    fig.tight_layout(); fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
