"""Prototype: broaden the close-pair Δz to fix the COLLIDED redshift over-confidence.

pit_breakdown.py found the collided (69% of missing) per-object redshift PIT is U-shaped/over-confident:
the measured close-pair Δz pool (`measure_close_pair_dz`, Δz of OBSERVED survivor pairs) is too narrow
— survivor pairs are biased toward physical Δz≈0, under-representing the chance-projection background, so
the collided posterior `z_host + Δz` is too tight and the true z lands in the tails.

Fix (data-driven): mix the empirical pool with the chance-projection BACKGROUND — Δz of uncorrelated
pairs, i.e. differences of two redshifts drawn from the data n(z) — by a fraction `f`. f=0 is current.
Sweep f, re-measure the COLLIDED PIT (KS, shape). Also report the wp(rp) small-scale recovery so the
broadening is not bought at the cost of the radial clustering the close-pair model exists to preserve.

  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 JAX_PLATFORMS=cpu python validation/pit_closepair_prototype.py
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from echoes.surveys.boss import load_boss
from echoes.photoz import PhotoZKNN, photoz_features
from echoes.completion import measure_close_pair_dz, complete_catalog_photoz
from echoes.mock_systematics import apply_survey_systematics
from echoes.posterior import build_package
from echoes.pit import pit_uniformity, format_pit
from echoes.randoms import make_random_from_selection_function
from echoes import clustering_measure as clm
from echoes.distance import comoving_distance

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"


def object_pit(pkg, true_z):
    qlev = np.asarray(pkg["qlev"], float); invcdf = np.asarray(pkg["invcdf"], float)
    return np.array([np.interp(true_z[i], invcdf[i], qlev, left=0.0, right=1.0)
                     for i in range(invcdf.shape[0])])


def hist_shape(pit, n=10):
    h, _ = np.histogram(np.clip(pit, 0, 1), bins=n, range=(0, 1)); h = h / max(h.sum(), 1)
    edge = h[0] + h[-1]; mid = h[n // 2 - 1] + h[n // 2]
    return ("U/over-conf" if edge > 0.25 else ("peaked/under-conf" if mid > 0.25 else "~flat"))


def bg_dz(z_obs, n, rng):
    """Chance-projection Δz background: difference of two redshifts drawn from the data n(z)."""
    a = rng.choice(z_obs, n); b = rng.choice(z_obs, n)
    d = a - b
    return np.concatenate([d, -d])


def mix_pool(dz_base, bg, f, rng):
    """Pool with fraction f drawn from the background, (1−f) from the empirical survivor pool."""
    m = len(dz_base)
    take_bg = rng.random(m) < f
    out = dz_base.copy()
    out[take_bg] = rng.choice(bg, int(take_bg.sum()))
    return out


def cz(z, cosmo):
    import jax; jax.config.update("jax_enable_x64", True)
    return np.asarray(comoving_distance(np.asarray(z, np.float64), cosmo))


def run_one_seed(cat, pz, seed, fracs, wp_check, wp_ctx=None):
    ra = np.asarray(cat.ra_data); dec = np.asarray(cat.dec_data); z = np.asarray(cat.z_data)
    obs, tg, kept, true_z = apply_survey_systematics(
        ra, dec, z, cat.colors_data, cat.mags_data, np.asarray(cat.w_sys_data),
        coll_frac=0.6, zfail_frac=0.014, zfail_faint_bias=1.5, seed=seed)
    true_z = np.asarray(true_z); coll = np.asarray(tg.miss_kind) == "collided"
    dz_base = measure_close_pair_dz(obs, 62 / 3600.)
    bg = bg_dz(np.asarray(obs.z_data), len(dz_base), np.random.default_rng(seed + 5))
    ks = {}
    for f in fracs:
        dz_f = dz_base if f == 0 else mix_pool(dz_base, bg, f, np.random.default_rng(seed + 11))
        pit = object_pit(build_package(obs, tg, pz, dz_pool=dz_f), true_z)
        ks[f] = pit_uniformity(pit[coll])["ks"]
    f_opt = min(ks, key=ks.get)
    return ks, f_opt, int(coll.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="if given, run the f-sweep for each seed and report optimal-f stability")
    p.add_argument("--fracs", type=float, nargs="+", default=[0.0, 0.15, 0.3, 0.45, 0.6])
    p.add_argument("--wp-check", action="store_true", help="also measure wp(rp) recovery vs truth")
    args = p.parse_args()

    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    ra = np.asarray(cat.ra_data); dec = np.asarray(cat.dec_data); z = np.asarray(cat.z_data)
    feat = photoz_features(cat.colors_data, cat.mags_data)
    good = np.isfinite(feat).all(1) & (np.asarray(cat.imatch_data) == 1)
    pz = PhotoZKNN(k=100).fit(feat[good], z[good])

    if args.seeds is not None:
        print(f"=== collided-PIT KS vs f, across seeds (optimal-f stability) ===")
        print(f"  {'seed':>4} {'N_coll':>7} " + " ".join(f"f={f:<5.2f}" for f in args.fracs) + "  f_opt")
        fopts = []
        for sd in args.seeds:
            ks, f_opt, ncoll = run_one_seed(cat, pz, sd, args.fracs, False)
            fopts.append(f_opt)
            print(f"  {sd:>4} {ncoll:>7,} " + " ".join(f"{ks[f]:>7.3f}" for f in args.fracs)
                  + f"   {f_opt:.2f}", flush=True)
        print(f"\n  optimal f: {fopts}  (mean {np.mean(fopts):.2f} ± {np.std(fopts):.2f})")
        print("  STABLE if f_opt clusters tightly — then ship that f; else the knob is seed-sensitive.")
        return

    obs, tg, kept, true_z = apply_survey_systematics(
        ra, dec, z, cat.colors_data, cat.mags_data, np.asarray(cat.w_sys_data),
        coll_frac=0.6, zfail_frac=0.014, zfail_faint_bias=1.5, seed=args.seed)
    true_z = np.asarray(true_z); miss = np.asarray(tg.miss_kind); coll = miss == "collided"
    dz_base = measure_close_pair_dz(obs, 62 / 3600.)
    rng = np.random.default_rng(args.seed + 5)
    bg = bg_dz(np.asarray(obs.z_data), len(dz_base), rng)
    print(f"inject-and-recover: {len(true_z):,} missing ({coll.mean():.0%} collided)")
    print(f"close-pair Δz: empirical std={dz_base.std():.4f}  background std={bg.std():.4f}\n")

    # wp(rp) truth + randoms (optional, the clustering tradeoff)
    if args.wp_check:
        rar, decr, zr = make_random_from_selection_function(
            sel_map=cat.sel_map, n_random=3 * cat.N_data, z_data=z, nside=cat.nside,
            rng=np.random.default_rng(7))
        rp_edges = np.logspace(np.log10(0.5), np.log10(40.0), 13); rpc = np.sqrt(rp_edges[1:] * rp_edges[:-1])
        czr = cz(zr, cat.fid_cosmo)
        rr = clm.build_random_pairs(rar, decr, czr, np.ones(len(rar)), rp_edges=rp_edges, pimax=80.0,
                                    npibins=80, s_edges=np.linspace(1, 40, 2), nmu=2, nthreads=8)
        _, wp_t = clm.measure_wp(ra, dec, cz(z, cat.fid_cosmo), np.ones(len(ra)), rar, decr, czr,
                                 np.ones(len(rar)), rp_edges=rp_edges, pimax=80.0, npibins=80, nthreads=8, rr=rr)

    print(f"  {'f_bg':>5} {'COLLIDED KS':>12} {'shape':>16} {'std':>7} {'ALL KS':>8}"
          + ("  wp<2Mpc/truth" if args.wp_check else ""))
    for f in args.fracs:
        dz_f = dz_base if f == 0 else mix_pool(dz_base, bg, f, np.random.default_rng(args.seed + 11))
        pkg = build_package(obs, tg, pz, dz_pool=dz_f)
        pit = object_pit(pkg, true_z)
        pc = pit_uniformity(pit[coll]); pa = pit_uniformity(pit)
        line = (f"  {f:>5.2f} {pc['ks']:>12.3f} {hist_shape(pit[coll]):>16} {pc['std']:>7.3f} "
                f"{pa['ks']:>8.3f}")
        if args.wp_check:
            c = complete_catalog_photoz(cat, tg, pz, seed=0, z_mode="nn", dz_pool=dz_f)
            _, wp_c = clm.measure_wp(np.asarray(c["ra"]), np.asarray(c["dec"]),
                                     cz(np.asarray(c["z"]), cat.fid_cosmo), np.ones(c["N"]),
                                     rar, decr, czr, np.ones(len(rar)), rp_edges=rp_edges,
                                     pimax=80.0, npibins=80, nthreads=8, rr=rr)
            sm = rpc < 2.0
            line += f"   {np.median(wp_c[sm] / wp_t[sm]):.3f}"
        print(line, flush=True)
    print("\n(uniform PIT std=0.289; collided was U/over-confident at f=0. wp<2/truth≈1 = clustering kept.)")


if __name__ == "__main__":
    main()
