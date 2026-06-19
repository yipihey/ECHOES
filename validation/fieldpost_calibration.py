"""Does propagating field uncertainty calibrate the per-galaxy posterior?

The density engines (and fieldpost with the posterior *mean*) draw every
realization's redshift from the SAME line-of-sight density, so the only spread is
the sampling of one fixed posterior — the per-galaxy ensemble is mildly
over-confident (true z lands in the tails too often). The field engine can do
better: draw a different Matheron FIELD SAMPLE per realization, so the ensemble
also spans the reconstruction uncertainty. This test compares the per-galaxy
ensemble PIT (rank of the true redshift) for fieldpost with the posterior mean vs
with field-sample propagation, on real-BOSS-truth inject-and-recover.

    JAX_PLATFORMS=cpu OMP_NUM_THREADS=16 ~/.venv/k3d/bin/python3 \
        validation/fieldpost_calibration.py [--n-real 20 --max-targets 2500]
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from echoes.surveys.boss import load_boss
from echoes.photoz import PhotoZKNN, photoz_features
from echoes.completion import measure_close_pair_dz, _clpair_density
from echoes.mock_systematics import apply_survey_systematics
from echoes.fieldpost import build_field_context, los_overdensity
from echoes.pit import pit_uniformity, format_pit

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-real", type=int, default=20)
    p.add_argument("--zfail-frac", type=float, default=0.03)
    p.add_argument("--max-targets", type=int, default=2500)
    args = p.parse_args()

    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    ra = np.asarray(cat.ra_data); dec = np.asarray(cat.dec_data); z = np.asarray(cat.z_data)
    colors = np.asarray(cat.colors_data); mags = np.asarray(cat.mags_data); wsys = np.asarray(cat.w_sys_data)
    obs, tg, kept, true_z = apply_survey_systematics(
        ra, dec, z, colors, mags, wsys, coll_frac=0.6, zfail_frac=args.zfail_frac,
        zfail_faint_bias=1.5, seed=0)
    kind = np.asarray(tg.miss_kind)
    of = photoz_features(obs.colors_data, obs.mags_data); og = np.isfinite(of).all(1)
    pz = PhotoZKNN(k=100).fit(of[og], np.asarray(obs.z_data)[og])
    dz = measure_close_pair_dz(obs, 62 / 3600.); z_o = np.asarray(obs.z_data)

    M = tg.N
    sub = np.arange(M) if M <= args.max_targets else np.random.default_rng(0).choice(M, args.max_targets, replace=False)
    ra_m = np.asarray(tg.ra)[sel := sub]; dec_m = np.asarray(tg.dec)[sub]; kind = kind[sub]
    ztrue = true_z[sub]; host = np.asarray(tg.host_index)[sub]
    z_host = np.where(host >= 0, z_o[np.clip(host, 0, len(z_o) - 1)], np.nan)
    coll = (kind == "collided") & (host >= 0)
    feat = photoz_features(np.asarray(tg.colors)[sub], np.asarray(tg.mags)[sub])
    zk, wk = pz.posterior(feat)
    print(f"missing N={len(sub):,} ({int((kind=='collided').sum()):,} collided + {int((kind=='zfail').sum()):,} zfail)")

    zgrid = np.linspace(z_o.min(), z_o.max(), 160)
    nzc = np.linspace(z_o.min(), z_o.max(), 64)
    nbar_z = np.interp(zgrid, nzc, np.histogram(z_o, bins=np.linspace(z_o.min(), z_o.max(), 65))[0].astype(float), 0, 0)
    pcl = _clpair_density(dz); bw_p = 0.02
    PP = np.zeros((len(sub), zgrid.size))
    for i in range(len(sub)):
        wi = wk[i]; ok = np.isfinite(wi) & (wi > 0)
        PP[i] = ((wi[ok][None, :] * np.exp(-0.5 * ((zgrid[:, None] - zk[i][ok][None, :]) / bw_p) ** 2)).sum(1)
                 if ok.any() else np.ones_like(zgrid))

    fc = build_field_context(obs, sel_map=cat.sel_map, nside=cat.nside, seed=0, verbose=True)
    print("[fieldpost] posterior mean + field draws ...")
    opd_mean = los_overdensity(fc, ra_m, dec_m, zgrid)                       # (M, n_z)
    opd_draws = los_overdensity(fc, ra_m, dec_m, zgrid, n_samples=args.n_real, seed=7)  # (M, n_real, n_z)

    rng = np.random.default_rng(1)
    def ensemble_pit(get_opd):
        Z = np.empty((args.n_real, len(sub)))
        for s in range(args.n_real):
            for i in range(len(sub)):
                pf = np.clip(get_opd(i, s), 0, None) * nbar_z
                p = pf * PP[i]
                if coll[i]:
                    p = p * pcl(zgrid - z_host[i])
                tot = p.sum()
                Z[s, i] = rng.choice(zgrid, p=p / tot) if tot > 0 else (z_host[i] if np.isfinite(z_host[i]) else np.median(z_o))
        pit = ((Z < ztrue[None, :]).sum(0) + rng.uniform(size=len(sub)) * (Z == ztrue[None, :]).sum(0)) / args.n_real
        return pit

    pit_mean = ensemble_pit(lambda i, s: opd_mean[i])               # same field every realization
    pit_draw = ensemble_pit(lambda i, s: opd_draws[i, s])           # a field sample per realization

    print(f"\n=== per-galaxy ensemble PIT (uniform = calibrated) ===")
    for nm, pit in [("posterior MEAN (over-confident)", pit_mean),
                    ("FIELD-SAMPLE propagation", pit_draw)]:
        print(f"  {nm:32s}: {format_pit(pit_uniformity(pit))}")
        for sub_k in ("collided", "zfail"):
            m = kind == sub_k
            print(f"      {sub_k:9s}: {format_pit(pit_uniformity(pit[m]))}")
    print("\n(field-sample propagation should move the PIT std toward 0.289 / raise the KS,χ² "
          "p-values vs the over-confident posterior-mean version.)")


if __name__ == "__main__":
    main()
