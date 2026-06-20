"""Stage-0 DECISION PROBE — does a measured monotonic transform T(g) close the
non-Gaussian (kNN-CDF) gap of the Gaussian GraphGP field?

This is the gate that decides Tier A (measured transform) vs Tier B (disco-dj
field-level fit) BEFORE any production wiring. The logic:

  1. Measure the real CMASS-South one-point overdensity PDF via counts-in-cells.
     Fit ``DensityTransform`` (empirical + lognormal) to it — the data is the only
     input, so this is purely data-driven.
  2. Build the existing GraphGP/FKP field and Poisson-sample galaxies two ways:
       * GAUSSIAN  — the field as-is (the current product; right P(k), wrong PDF).
       * TRANSFORM — T applied to the gaussianised field (reshaped 1-pt PDF).
  3. Score BOTH against the real data with:
       * counts-in-cells (var/mean, skew)  — the IN-SAMPLE target T is fit to.
       * kNN-CDF (k=1,2,4)                  — the OUT-OF-SAMPLE non-Gaussian probe
         T does NOT directly target; the real decision metric.

  DECISION: PASS Tier A if the transform's kNN-CDF max|ΔCDF| vs data is
  substantially below the Gaussian baseline's AND its CiC skew tracks the data.
  If the residual is irreducible (phase-limited) → escalate to Tier B.

    OMP_NUM_THREADS=16 JAX_PLATFORMS=cpu ~/.venv/k3d/bin/python3 \
        validation/transform_probe.py
"""
import argparse, copy, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from echoes.surveys.boss import load_boss
from echoes.graphgp_field import sample_posterior_density_field
from echoes.density_transform import fit_density_transform, _moments
from echoes.clustering import comoving_mpc_h
from validation.higher_order import xyz, knn_cdf, cic

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"


def cic_overdensity(gal_xyz, cen_xyz, radius):
    """Counts-in-cells (1+δ) = N_cell / ⟨N_cell⟩ at the cell centres."""
    n = cic(gal_xyz, cen_xyz, radius).astype(float)
    return n / max(n.mean(), 1e-9)


def sample_variant(res, cat, dt, *, n_samples, seed0=100):
    """Sample galaxy catalogs from a DensityFieldResult through the PRODUCTION seam
    (``DensityFieldResult.sample_catalog(transform=...)``, echoes.graphgp_field) —
    so this exercises the real Tier-A code path, not an offline field edit. ``dt`` is
    a :class:`DensityTransform`; its ``apply_to_field`` reshapes each (1+δ) draw
    over occupied voxels before Poisson sampling."""
    tf = None if dt.kind == "identity" else dt.apply_to_field
    return [res.sample_catalog(cat, sample_idx=s, seed=seed0 + s, transform=tf)
            for s in range(n_samples)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nside", type=int, default=64)
    p.add_argument("--nz", type=int, default=64)
    p.add_argument("--n-samples", type=int, default=3)
    p.add_argument("--cic-R", type=float, default=8.0, help="CiC radius [Mpc/h]")
    p.add_argument("--out", default="output/transform_probe.png")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256)
    ra = np.asarray(cat.ra_data); dec = np.asarray(cat.dec_data); z = np.asarray(cat.z_data)
    print(f"loaded CMASS-South: {len(ra):,} galaxies, {len(cat.ra_random):,} randoms")

    # ---- query/cell points from randoms (footprint-uniform) ----
    rng = np.random.default_rng(3)
    nr = len(cat.ra_random)
    rar = np.asarray(cat.ra_random); decr = np.asarray(cat.dec_random); zr = np.asarray(cat.z_random)
    qsel = rng.choice(nr, min(50_000, nr), replace=False)
    q_xyz = xyz(rar[qsel], decr[qsel], zr[qsel])
    csel = rng.choice(nr, min(8_000, nr), replace=False)
    c_xyz = xyz(rar[csel], decr[csel], zr[csel])

    # ---- 1. data truth: kNN-CDF + CiC, and the transform target PDF ----
    ks = [1, 2, 4]; redges = np.logspace(np.log10(2.0), np.log10(40.0), 30)
    data_xyz = xyz(ra, dec, z)
    knn_d = knn_cdf(data_xyz, q_xyz, ks, redges)
    counts_data = cic(data_xyz, c_xyz, args.cic_R).astype(float)   # raw counts (for deconv)
    opd_data = counts_data / max(counts_data.mean(), 1e-9)
    from echoes.density_transform import field_moments_from_counts
    print(f"\ndata CiC (R={args.cic_R}) raw moments (mean,var,skew): "
          f"{tuple(round(x,3) for x in _moments(opd_data))}  "
          f"<N>={counts_data.mean():.2f}/cell")
    print(f"  shot-noise-free field moments (var,skew): "
          f"{tuple(round(x,3) for x in field_moments_from_counts(counts_data)[1:])}")

    dt_emp = fit_density_transform(opd_data, kind="empirical", scale=args.cic_R)
    dt_log = fit_density_transform(opd_data, kind="lognormal", scale=args.cic_R)
    dt_log_dec = fit_density_transform(opd_data, kind="lognormal", scale=args.cic_R,
                                       counts=counts_data)         # shot-noise-deconvolved
    print(f"  lognormal sigma_g: raw={dt_log.sigma_g:.3f}  deconv={dt_log_dec.sigma_g:.3f}")

    # ---- 2. build the field, sample GAUSSIAN + TRANSFORM mocks ----
    print("\nbuilding GraphGP/FKP field ...")
    res = sample_posterior_density_field(cat, n_samples=args.n_samples, n_z_bins=args.nz,
                                         nside=args.nside, seed=0, verbose=False)
    from echoes.density_transform import DensityTransform
    variants = {"gaussian": DensityTransform(kind="identity"),
                "transform-emp": dt_emp,
                "transform-log": dt_log,
                "transform-log-deconv": dt_log_dec}

    knn_v = {}; cic_v = {}
    for name, dt in variants.items():
        cats = sample_variant(res, cat, dt, n_samples=args.n_samples)
        kk = []; oo = []
        for g in cats:
            gx = xyz(np.asarray(g["ra"]), np.asarray(g["dec"]), np.asarray(g["z"]))
            kk.append(knn_cdf(gx, q_xyz, ks, redges))
            oo.append(cic_overdensity(gx, c_xyz, args.cic_R))
        knn_v[name] = {k: np.mean([K[k] for K in kk], 0) for k in ks}
        cic_v[name] = np.concatenate(oo)

    # ---- 3. report ----
    print("\n=== counts-in-cells (var/mean, skew) — T is fit to match the DATA ===")
    def vm_sk(o): m, v, s = _moments(o); return v / max(m, 1e-9), s
    print(f"  {'data':14s} var/mean={vm_sk(opd_data)[0]:6.3f}  skew={vm_sk(opd_data)[1]:6.3f}")
    for name in variants:
        a, b = vm_sk(cic_v[name])
        print(f"  {name:14s} var/mean={a:6.3f}  skew={b:6.3f}")

    print("\n=== kNN-CDF max|ΔCDF| vs DATA  (the out-of-sample DECISION metric) ===")
    print(f"  {'variant':14s} " + "  ".join(f"k={k}" for k in ks))
    score = {}
    for name in variants:
        devs = [float(np.max(np.abs(knn_v[name][k] - knn_d[k]))) for k in ks]
        score[name] = devs
        print(f"  {name:14s} " + "  ".join(f"{d:.4f}" for d in devs))

    g_mean = np.mean(score["gaussian"])
    best_t = min(np.mean(score[n]) for n in variants if n != "gaussian")
    improve = 100.0 * (g_mean - best_t) / max(g_mean, 1e-9)
    print(f"\nmean kNN max|ΔCDF|:  gaussian={g_mean:.4f}  best-transform={best_t:.4f}  "
          f"=> {improve:+.0f}% gap reduction")
    verdict = ("PASS Tier A (transform closes the non-Gaussian gap)" if improve >= 25
               else "MARGINAL — inspect per-k; consider Tier B (disco-dj)" if improve >= 10
               else "FAIL — phase-limited; ESCALATE to Tier B (disco-dj)")
    print(f"DECISION: {verdict}")

    # ---- 4. plot ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    a = ax[0]
    cols = {"gaussian": "#888888", "transform-emp": "#3a6ea8", "transform-log": "#e8853a",
            "transform-log-deconv": "#2ca02c"}
    for k, ls in zip(ks, ["-", "--", ":"]):
        a.semilogx(redges, knn_d[k], color="k", lw=2.4, ls=ls, label=f"data k={k}")
        for name in variants:
            a.semilogx(redges, knn_v[name][k], color=cols[name], lw=1.3, ls=ls, alpha=0.9)
    a.set_xlabel("r [Mpc/h]"); a.set_ylabel("kNN-CDF P(<r)")
    a.set_title("kNN-CDF: data (black) vs gaussian/transform"); a.legend(fontsize=7, ncol=3)
    a = ax[1]
    mx = np.percentile(np.concatenate([opd_data, cic_v["gaussian"], cic_v["transform-emp"]]), 99.5)
    bins = np.linspace(0, mx, 40)
    a.hist(opd_data, bins=bins, density=True, histtype="step", color="k", lw=2.4, label="data")
    for name in variants:
        a.hist(cic_v[name], bins=bins, density=True, histtype="step", color=cols[name], lw=1.5, label=name)
    a.set_xlabel(f"1+δ in R={args.cic_R} Mpc/h sphere"); a.set_ylabel("PDF")
    a.set_title("counts-in-cells one-point PDF"); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
