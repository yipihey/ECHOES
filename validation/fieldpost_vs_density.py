"""Does the field-level conditional posterior beat the density engines?

The bar set by our earlier diagnostics: a local density raises the posterior mass
at the true redshift ~20-37% over photo-z, but the density engines (KNN-KDE
'field', kNN2D) tie each other and leave the per-galaxy posterior mildly
over-confident, because they use only the LOCAL density. The field-level
'fieldpost' engine evaluates the proper GP posterior of the overdensity field
along each sightline (conditioned on the nearby observed galaxies + their
correlations). This test asks whether that non-local conditioning raises the
sampling-free density-quality metric — posterior mass at the truth
P(|z-z_true|<dz) — above the KNN-KDE engine, especially on the density-localized
redshift-failure galaxies.

Inject-and-recover on real-BOSS-truth; the field context is built on the
mock-observed catalog. Reports photo-z-only / field (KNN-KDE) / fieldpost posterior
mass at truth, split by collided vs zfail.

    JAX_PLATFORMS=cpu OMP_NUM_THREADS=16 ~/.venv/k3d/bin/python3 \
        validation/fieldpost_vs_density.py [--zfail-frac 0.03]
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy.spatial import cKDTree
from echoes.surveys.boss import load_boss
from echoes.photoz import PhotoZKNN, photoz_features
from echoes.geometry import _radec_to_nhat
from echoes.completion import measure_close_pair_dz, _clpair_density
from echoes.mock_systematics import apply_survey_systematics
from echoes.fieldpost import build_field_context, los_overdensity

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coll-frac", type=float, default=0.6)
    p.add_argument("--zfail-frac", type=float, default=0.03)
    p.add_argument("--dz-struct", type=float, default=0.006)
    p.add_argument("--max-targets", type=int, default=4000, help="subsample targets for speed")
    args = p.parse_args()

    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    ra = np.asarray(cat.ra_data); dec = np.asarray(cat.dec_data); z = np.asarray(cat.z_data)
    colors = np.asarray(cat.colors_data); mags = np.asarray(cat.mags_data); wsys = np.asarray(cat.w_sys_data)

    obs, tg, kept, true_z = apply_survey_systematics(
        ra, dec, z, colors, mags, wsys, coll_frac=args.coll_frac,
        zfail_frac=args.zfail_frac, zfail_faint_bias=1.5, seed=0)
    kind = np.asarray(tg.miss_kind)
    of = photoz_features(obs.colors_data, obs.mags_data); ogood = np.isfinite(of).all(1)
    pz = PhotoZKNN(k=100).fit(of[ogood], np.asarray(obs.z_data)[ogood])
    dz = measure_close_pair_dz(obs, 62 / 3600.)
    z_o = np.asarray(obs.z_data); ra_o = np.asarray(obs.ra_data); dec_o = np.asarray(obs.dec_data)

    # subsample targets for the (per-sightline dense-solve) speed
    M = tg.N
    sel = np.arange(M) if M <= args.max_targets else np.random.default_rng(0).choice(M, args.max_targets, replace=False)
    ra_m = np.asarray(tg.ra)[sel]; dec_m = np.asarray(tg.dec)[sel]; kind = kind[sel]
    ztrue = true_z[sel]; host = np.asarray(tg.host_index)[sel]
    z_host = np.where(host >= 0, z_o[np.clip(host, 0, len(z_o) - 1)], np.nan)
    coll = (kind == "collided") & (host >= 0)
    feat = photoz_features(np.asarray(tg.colors)[sel], np.asarray(tg.mags)[sel])
    zk, wk = pz.posterior(feat)
    print(f"missing N={len(sel):,} ({int((kind=='collided').sum()):,} collided + {int((kind=='zfail').sum()):,} zfail)")

    zgrid = np.linspace(z_o.min(), z_o.max(), 160)
    nzc = np.linspace(z_o.min(), z_o.max(), 64); nbar_z = np.interp(
        zgrid, nzc, np.histogram(z_o, bins=np.linspace(z_o.min(), z_o.max(), 65))[0].astype(float),
        left=0.0, right=0.0)
    pcl = _clpair_density(dz)
    bw_p, bw_f = 0.02, 0.004

    # photo-z LOS posterior
    PP = np.zeros((len(sel), zgrid.size))
    for i in range(len(sel)):
        wi = wk[i]; ok = np.isfinite(wi) & (wi > 0)
        PP[i] = ((wi[ok][None, :] * np.exp(-0.5 * ((zgrid[:, None] - zk[i][ok][None, :]) / bw_p) ** 2)).sum(1)
                 if ok.any() else np.ones_like(zgrid))
    # field (KNN-KDE): KDE of K=150 nearest observed spec-z
    K = min(150, len(z_o))
    _, nn = cKDTree(_radec_to_nhat(ra_o, dec_o)).query(_radec_to_nhat(ra_m, dec_m), k=K, workers=-1)
    # fieldpost: conditional field overdensity along each sightline
    print("[fieldpost] building context + scoring sightlines ...")
    fc = build_field_context(obs, sel_map=cat.sel_map, nside=cat.nside, seed=0, verbose=True)
    opd_fp = los_overdensity(fc, ra_m, dec_m, zgrid)

    def pstruct(pun, i):
        s = pun.sum()
        if s <= 0: return np.nan
        return float(pun[np.abs(zgrid - ztrue[i]) < args.dz_struct].sum() / s)

    Pm = {"photoz-only": [], "field (KNN-KDE)": [], "fieldpost (field GP)": []}
    for i in range(len(sel)):
        pf_field = np.exp(-0.5 * ((zgrid[:, None] - z_o[nn[i]][None, :]) / bw_f) ** 2).sum(1)
        pf_fp = opd_fp[i] * nbar_z
        base = pcl(zgrid - z_host[i]) if coll[i] else 1.0
        Pm["photoz-only"].append(pstruct(PP[i] * base, i))
        Pm["field (KNN-KDE)"].append(pstruct(pf_field * PP[i] * base, i))
        Pm["fieldpost (field GP)"].append(pstruct(pf_fp * PP[i] * base, i))
    for k in Pm: Pm[k] = np.array(Pm[k])

    print(f"\n=== posterior mass at truth P(|z-z_true|<{args.dz_struct})  (median; higher = better) ===")
    print(f"{'engine':22s} {'collided':>10s} {'zfail':>10s} {'all':>10s}")
    for name, v in Pm.items():
        def med(m): return float(np.nanmedian(v[m])) if m.any() else float('nan')
        print(f"{name:22s} {med(kind=='collided'):10.4f} {med(kind=='zfail'):10.4f} {med(np.ones(len(v),bool)):10.4f}")
    fp = Pm["fieldpost (field GP)"]; fl = Pm["field (KNN-KDE)"]
    for sub in ("collided", "zfail", "all"):
        m = (kind == sub) if sub != "all" else np.ones(len(fp), bool)
        d = 100 * (np.nanmedian(fp[m]) - np.nanmedian(fl[m]))
        print(f"  fieldpost - field ({sub}): {d:+.2f} pp")
    print("\n(the bar: fieldpost should EXCEED field (KNN-KDE), especially on zfail, by using "
          "the field's non-local correlations rather than only the local density.)")


if __name__ == "__main__":
    main()
