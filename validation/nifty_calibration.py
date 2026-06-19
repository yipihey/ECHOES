"""Does the NIFTy LGCP shape uncertainty calibrate the per-galaxy redshift?

The graphGP field-sample propagation barely calibrated the per-galaxy posterior
(KS 0.144→0.132) because its linearized field draws vary in amplitude, which
normalises out of p(z) ∝ (1+δ)·n̄·p_photoz, not in the z-shape that localises the
redshift. The diagnosis: proper calibration needs the field posterior's SHAPE
uncertainty — exactly what the full Poisson-lognormal LGCP (NIFTy geoVI) carries.

This test runs the NIFTy LGCP on a comoving sub-box of real-BOSS-truth
inject-and-recover, draws posterior field SAMPLES (each a full reconstruction
with its own structure), and uses them for the redshifts of the missing galaxies
in that box. We compare, on those galaxies:
  * the per-galaxy ensemble PIT (is it calibrated now?) — NIFTy LGCP samples vs
    the graphGP posterior-mean, and
  * the posterior mass at the true redshift (does the LGCP push it further?).

    JAX_PLATFORMS=cpu OMP_NUM_THREADS=16 ~/.venv/k3d/bin/python3 \
        validation/nifty_calibration.py [--ngrid 40 --box 240 --niter 10]
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from echoes.surveys.boss import load_boss
from echoes.photoz import PhotoZKNN, photoz_features
from echoes.geometry import _radec_to_nhat
from echoes.clustering import comoving_mpc_h
from echoes.completion import measure_close_pair_dz, _clpair_density
from echoes.mock_systematics import apply_survey_systematics
from echoes.fieldpost import build_field_context, los_overdensity
from echoes.pit import pit_uniformity, format_pit

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ngrid", type=int, default=40)
    p.add_argument("--box", type=float, default=240.0)
    p.add_argument("--niter", type=int, default=10)
    p.add_argument("--nsamp", type=int, default=8)
    p.add_argument("--zfail-frac", type=float, default=0.05)
    args = p.parse_args()

    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    ra = np.asarray(cat.ra_data); dec = np.asarray(cat.dec_data); z = np.asarray(cat.z_data)
    colors = np.asarray(cat.colors_data); mags = np.asarray(cat.mags_data); wsys = np.asarray(cat.w_sys_data)
    obs, tg, kept, true_z = apply_survey_systematics(
        ra, dec, z, colors, mags, wsys, coll_frac=0.6, zfail_frac=args.zfail_frac,
        zfail_faint_bias=1.5, seed=0)
    of = photoz_features(obs.colors_data, obs.mags_data); og = np.isfinite(of).all(1)
    pz = PhotoZKNN(k=100).fit(of[og], np.asarray(obs.z_data)[og])
    dz = measure_close_pair_dz(obs, 62 / 3600.); z_o = np.asarray(obs.z_data)
    ra_o = np.asarray(obs.ra_data); dec_o = np.asarray(obs.dec_data)
    x_o = comoving_mpc_h(z_o)[:, None] * _radec_to_nhat(ra_o, dec_o)

    # comoving cube centred on the median observed position
    c0 = np.median(x_o, axis=0); half = args.box / 2.0
    inbox_o = np.all(np.abs(x_o - c0) < half, axis=1)
    lo = c0 - half; n = args.ngrid
    edges = [np.linspace(lo[d], lo[d] + args.box, n + 1) for d in range(3)]
    counts, _ = np.histogramdd(x_o[inbox_o], bins=edges)
    nbar_vox = max(inbox_o.sum() / n ** 3, 1e-6)
    cell = args.box / n
    print(f"box {args.box} Mpc/h, grid {n}³, {int(inbox_o.sum()):,} observed in box, cell {cell:.1f} Mpc/h")

    # missing galaxies physically inside the cube (true position) — the test set
    x_t = comoving_mpc_h(true_z)[:, None] * _radec_to_nhat(np.asarray(tg.ra), np.asarray(tg.dec))
    inbox_t = np.all(np.abs(x_t - c0) < half * 0.9, axis=1)
    sel = np.flatnonzero(inbox_t)
    kind = np.asarray(tg.miss_kind)[sel]
    ra_m = np.asarray(tg.ra)[sel]; dec_m = np.asarray(tg.dec)[sel]
    ztrue = true_z[sel]; host = np.asarray(tg.host_index)[sel]
    z_host = np.where(host >= 0, z_o[np.clip(host, 0, len(z_o) - 1)], np.nan)
    coll = (kind == "collided") & (host >= 0)
    feat = photoz_features(np.asarray(tg.colors)[sel], np.asarray(tg.mags)[sel])
    zk, wk = pz.posterior(feat)
    print(f"missing-in-box N={len(sel):,} ({int((kind=='collided').sum())} collided + {int((kind=='zfail').sum())} zfail)")
    if len(sel) < 40:
        print("too few missing galaxies in box; increase --box"); return

    # --- NIFTy LGCP on the box: posterior field samples ---
    import jax.numpy as jnp, jax.random as random
    import nifty8.re as jft
    from scipy.interpolate import RegularGridInterpolator
    cfm = jft.CorrelatedFieldMaker("cf")
    cfm.set_amplitude_total_offset(0.0, (1e-1, 1e-2))
    cfm.add_fluctuations((n, n, n), distances=cell, fluctuations=(1.0, 0.5),
                         loglogavgslope=(-4.0, 1.0), prefix="")
    cf = cfm.finalize()
    data = jnp.asarray(counts.ravel().astype(np.int32))
    signal = jft.Model(lambda L: nbar_vox * jnp.exp(cf(L).reshape(-1)), domain=cf.domain, init=cf.init)
    lh = jft.Poissonian(data).amend(signal)
    key = random.PRNGKey(0)
    print(f"[nifty] geoVI {args.niter} iters, {args.nsamp} samples ...")
    samples, _ = jft.optimize_kl(lh, jft.Vector(cf.init(key)), key=key,
                                 n_total_iterations=args.niter, n_samples=args.nsamp,
                                 sample_mode="nonlinear_resample")
    centres_1d = [0.5 * (e[1:] + e[:-1]) for e in edges]
    opd_grids = [np.asarray(jnp.exp(cf(s)).reshape(n, n, n)) for s in samples]   # (1+δ) per sample
    interps = [RegularGridInterpolator(centres_1d, g, bounds_error=False, fill_value=1.0) for g in opd_grids]
    K = len(interps)
    print(f"[nifty] {K} posterior field samples")

    # --- graphGP posterior-mean field along the same sightlines ---
    fc = build_field_context(obs, sel_map=cat.sel_map, nside=cat.nside, seed=0)
    zgrid = np.linspace(z_o.min(), z_o.max(), 160)
    chi_grid = comoving_mpc_h(zgrid)
    nbar_z = np.interp(zgrid, np.linspace(z_o.min(), z_o.max(), 64),
                       np.histogram(z_o, bins=np.linspace(z_o.min(), z_o.max(), 65))[0].astype(float), 0, 0)
    opd_gp = los_overdensity(fc, ra_m, dec_m, zgrid)        # (M, n_z) mean
    pcl = _clpair_density(dz); bw_p = 0.02
    nhat_m = _radec_to_nhat(ra_m, dec_m)

    # photo-z LOS posterior
    PP = np.zeros((len(sel), zgrid.size))
    for i in range(len(sel)):
        wi = wk[i]; ok = np.isfinite(wi) & (wi > 0)
        PP[i] = ((wi[ok][None, :] * np.exp(-0.5 * ((zgrid[:, None] - zk[i][ok][None, :]) / bw_p) ** 2)).sum(1)
                 if ok.any() else np.ones_like(zgrid))

    # NIFTy (1+δ) samples along each sightline (interp at comoving LOS points)
    opd_nifty = np.ones((len(sel), K, zgrid.size))
    los_xyz = chi_grid[None, :, None] * nhat_m[:, None, :]    # (M, n_z, 3)
    for k in range(K):
        opd_nifty[:, k, :] = interps[k](los_xyz.reshape(-1, 3)).reshape(len(sel), zgrid.size)

    rng = np.random.default_rng(1)
    def post(i, opd):  # build normalised p(z) for galaxy i with overdensity opd
        pf = np.clip(opd, 0, None) * nbar_z
        p = pf * PP[i]
        if coll[i]:
            p = p * pcl(zgrid - z_host[i])
        s = p.sum()
        return p / s if s > 0 else None

    def pstruct(p, i, dzs=0.006):
        return float(p[np.abs(zgrid - ztrue[i]) < dzs].sum()) if p is not None else np.nan

    # posterior mass at truth
    mass = {"photoz": [], "graphGP mean": [], "NIFTy LGCP (mean of samples)": []}
    pit = {"graphGP mean": np.empty(len(sel)), "NIFTy LGCP samples": np.empty(len(sel))}
    NS = 20
    for i in range(len(sel)):
        p_pz = post(i, np.ones_like(zgrid)); p_gp = post(i, opd_gp[i])
        p_nf = post(i, opd_nifty[i].mean(0))
        mass["photoz"].append(pstruct(p_pz, i)); mass["graphGP mean"].append(pstruct(p_gp, i))
        mass["NIFTy LGCP (mean of samples)"].append(pstruct(p_nf, i))
        # ensemble PIT: graphGP draws z from the SAME mean posterior NS times
        zg = np.array([rng.choice(zgrid, p=p_gp) if p_gp is not None else ztrue[i] for _ in range(NS)])
        pit["graphGP mean"][i] = ((zg < ztrue[i]).sum() + rng.uniform() * (zg == ztrue[i]).sum()) / NS
        # NIFTy: one draw per FIELD SAMPLE (shape varies) -> spans reconstruction uncertainty
        zn = []
        for k in range(K):
            pk = post(i, opd_nifty[i, k])
            zn.append(rng.choice(zgrid, p=pk) if pk is not None else ztrue[i])
        zn = np.array(zn)
        pit["NIFTy LGCP samples"][i] = ((zn < ztrue[i]).sum() + rng.uniform() * (zn == ztrue[i]).sum()) / K

    print("\n=== posterior mass at truth P(|z-z_true|<0.006), median ===")
    for nm, v in mass.items():
        print(f"  {nm:30s} {np.nanmedian(v):.4f}")
    print("\n=== per-galaxy ensemble PIT (uniform = calibrated) ===")
    for nm, v in pit.items():
        print(f"  {nm:24s}: {format_pit(pit_uniformity(v))}")
    print("\n(if the LGCP shape uncertainty calibrates, the NIFTy-samples PIT std → 0.289 and its "
          "KS/χ² p-values rise above the over-confident graphGP-mean version.)")


if __name__ == "__main__":
    main()
