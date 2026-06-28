"""Validate the DESI + Quaia ECHOES products (Phase 5/6 science gates).

DESI (restore_missing): the equal-weight completed catalog must reproduce the
completeness-WEIGHTED observed clustering (the completion absorbs the weights into
points), and the restored-object density must track the local coverage (NTILE).

Quaia (resample_redshift): the resampled n(z) preserves the photo-z n(z); the
per-object resampled redshift respects the photo-z width (marginal calibration);
positions are fixed across realizations.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=16 JAX_PLATFORMS=cpu \
        python validation/validate_new_datasets.py [--desi] [--quaia]
"""
import argparse, glob, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from echoes.posterior import load_package, draw
from echoes.clustering import comoving_mpc_h


def _wp(ra, dec, z, w, ra_r, dec_r, z_r, *, rp_edges, pimax=80.0):
    from echoes.clustering_measure import measure_wp
    cz = comoving_mpc_h(np.asarray(z)); cz_r = comoving_mpc_h(np.asarray(z_r))
    _, wp = measure_wp(np.asarray(ra), np.asarray(dec), cz, np.asarray(w),
                       np.asarray(ra_r), np.asarray(dec_r), cz_r, np.ones(len(ra_r)),
                       rp_edges=rp_edges, pimax=pimax)
    return wp


def validate_desi(n_sub=200_000, seed=0):
    print("\n===== DESI (restore_missing) =====")
    from echoes.surveys.desi import load_desi_qso, load_desi_targets
    dat = ["data/desi/QSO_NGC_clustering.dat.fits", "data/desi/QSO_SGC_clustering.dat.fits"]
    ran = sorted(glob.glob("data/desi/QSO_*_clustering.ran.fits"))
    cat = load_desi_qso(dat, ran, nside=256)
    tg = load_desi_targets(cat, seed=0)

    # (1) integral: equal-weight completed count ~ completeness-weighted observed count
    n_completed = cat.N_data + tg.N
    w_sum = float(np.sum(cat.w_comp_data))
    print(f"[count] completed N={n_completed:,}  vs  sum(WEIGHT_COMP·ZFAIL)={w_sum:,.0f}  "
          f"ratio={n_completed/w_sum:.4f}  (target ~1.00)")

    # (2) uneven coverage: restored fraction must rise as coverage (NTILE) falls
    print("[coverage] restored/observed by NTILE (should fall with more tiles):")
    for nt in (1, 2, 3, 4):
        obs = int((cat.ntile_data == nt).sum())
        res = int((cat.ntile_data[tg.host_index] == nt).sum())
        if obs:
            print(f"           NTILE={nt}: {res/obs:6.3f}   (obs {obs:,})")

    # (3) clustering: the two estimates of the SAME underlying clustering must agree —
    #     observed(w_comp) with the (completeness-imprinted) SURVEY randoms [the standard
    #     DESI measurement], and equal-weight COMPLETED with the shipped UNIFORM-footprint
    #     randoms [the ECHOES measurement]. The completed catalog has the selection removed,
    #     so it must NOT be paired with the survey randoms.
    pkg = load_package("data_release/desi_qso/desi_qso_posterior.npz")
    rnd = np.load("data_release/desi_qso/desi_qso_randoms.npz")
    rng = np.random.default_rng(seed)
    d = draw(pkg, seed=seed, systot=False)
    rp_edges = np.logspace(np.log10(0.8), np.log10(40.0), 14)
    rp_c = np.sqrt(rp_edges[1:] * rp_edges[:-1])
    si = rng.choice(cat.N_data, min(n_sub, cat.N_data), replace=False)
    ci = rng.choice(d["N"], min(int(n_sub * d["N"] / cat.N_data), d["N"]), replace=False)
    nr = 4 * len(si)
    ri = rng.choice(len(cat.ra_random), min(nr, len(cat.ra_random)), replace=False)
    rj = rng.choice(len(rnd["ra"]), min(nr, len(rnd["ra"])), replace=False)
    wp_w = _wp(cat.ra_data[si], cat.dec_data[si], cat.z_data[si], cat.w_comp_data[si],
               cat.ra_random[ri], cat.dec_random[ri], cat.z_random[ri], rp_edges=rp_edges)
    wp_c = _wp(d["ra"][ci], d["dec"][ci], d["z"][ci], np.ones(len(ci)),
               rnd["ra"][rj], rnd["dec"][rj], rnd["z"][rj], rp_edges=rp_edges)
    ratio = wp_c / wp_w
    print("[wp] equal-weight-completed (uniform randoms) / completeness-weighted-observed (survey randoms):")
    for r, q in zip(rp_c, ratio):
        print(f"     rp={r:6.2f}  ratio={q:.3f}")
    print(f"[wp] median ratio over 2-40 Mpc/h = {np.nanmedian(ratio[rp_c > 2]):.3f}  (target ~1.0 ± ~10%)")


def validate_quaia(seed=0):
    print("\n===== Quaia (resample_redshift) =====")
    from echoes.surveys.quaia import load_quaia
    cat = load_quaia("data/quaia/quaia_G20.0.fits", "data/quaia/random_G20.0_10x.fits",
                     selection_path="data/quaia/selection_function_NSIDE64_G20.0.fits")
    pkg = load_package("data_release/quaia/quaia_posterior.npz")
    print(f"[mode] package_mode={pkg.get('package_mode')}  n_obs={pkg['n_obs']}  n_miss={pkg['n_miss']:,}")

    d0 = draw(pkg, seed=0, systot=False)
    d1 = draw(pkg, seed=1, systot=False)
    # positions fixed; redshifts genuinely resampled
    same_pos = np.array_equal(d0["ra"], d1["ra"]) and np.array_equal(d0["dec"], d1["dec"])
    print(f"[fixed] positions identical across seeds: {same_pos}  | z varies: {not np.allclose(d0['z'], d1['z'])}")

    # n(z): resampled n(z) preserves the photo-z n(z) envelope (selfunc inpaint included)
    photoz_rows = (pkg["base_prov"] == _prov("photoz"))
    bins = np.linspace(0.0, 4.6, 47)
    h_in, _ = np.histogram(cat.z_data, bins=bins, density=True)
    h_out, _ = np.histogram(d0["z"][photoz_rows], bins=bins, density=True)
    l1 = float(np.abs(h_in - h_out).sum() * (bins[1] - bins[0]))
    print(f"[n(z)] L1(resampled photo-z rows, input photo-z) = {l1:.3f}  (0=identical; <~0.1 good)")

    # marginal width: per-object posterior 1σ vs photo-z input (sharpened, never broader)
    q = np.asarray(pkg["qlev"]); iv = np.asarray(pkg["invcdf"])
    idx = np.where(photoz_rows[pkg["n_obs"]:])[0]                 # photoz rows among the M invcdf rows
    take = idx[:: max(1, len(idx) // 3000)]
    sig_post = np.median([(np.interp(0.84, q, iv[i]) - np.interp(0.16, q, iv[i])) / 2 for i in take])
    print(f"[width] median posterior 1σ = {sig_post:.4f}  vs  photo-z input median = "
          f"{np.median(cat.z_data_err):.4f}  (sharpened, ≤ input)")

    # provenance census
    u, c = np.unique(d0["prov"], return_counts=True)
    print("[prov] " + "  ".join(f"{_prov_name(uu)}={cc:,}" for uu, cc in zip(u, c)))


def _prov(name):
    from echoes.completion import PROV
    return PROV[name]


def _prov_name(code):
    from echoes.completion import PROV_NAME
    return PROV_NAME[int(code)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desi", action="store_true")
    ap.add_argument("--quaia", action="store_true")
    args = ap.parse_args()
    both = not (args.desi or args.quaia)
    if args.desi or both:
        validate_desi()
    if args.quaia or both:
        validate_quaia()


if __name__ == "__main__":
    main()
