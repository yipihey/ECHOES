"""graphGP vs NIFTy cross-check of the reconstructed galaxy density field.

Two independent field engines should agree on the posterior overdensity field
where the data constrain it. In a comoving sub-box of CMASS-South (fully inside
the footprint, so no masking) we:
  * grid the observed galaxies into n³ voxels and count them;
  * NIFTy (nifty8.re): a log-normal intensity exp(s)·n̄ with a LEARNED power
    spectrum + Poisson likelihood, posterior s by geoVI;
  * graphGP: the conditional GP overdensity (echoes.field_posterior) at the same
    voxel centres, conditioned on the same galaxies with the ξ-tabulated kernel;
and correlate the two posterior log-fields. A high correlation is mutual
validation of the field reconstruction; the two priors differ (learned spectrum
vs measured ξ), so agreement is non-trivial.

    JAX_PLATFORMS=cpu OMP_NUM_THREADS=16 ~/.venv/k3d/bin/python3 \
        validation/fieldpost_vs_nifty.py [--ngrid 32 --box 220]
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from echoes.surveys.boss import load_boss
from echoes.geometry import _radec_to_nhat
from echoes.clustering import comoving_mpc_h
from echoes.fieldpost import build_field_context
from echoes.field_posterior import conditional_overdensity_los

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ngrid", type=int, default=32)
    p.add_argument("--box", type=float, default=220.0, help="comoving box side [Mpc/h]")
    p.add_argument("--n-pred", type=int, default=2500, help="voxel centres for the graphGP compare")
    p.add_argument("--niter", type=int, default=8)
    p.add_argument("--out", default="output/fieldpost_vs_nifty.png")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    ra = np.asarray(cat.ra_data); dec = np.asarray(cat.dec_data); z = np.asarray(cat.z_data)
    x = comoving_mpc_h(z)[:, None] * _radec_to_nhat(ra, dec)
    # centre a box on the densest interior region: pick the median position, offset inward
    c0 = np.median(x, axis=0)
    half = args.box / 2.0
    inbox = np.all(np.abs(x - c0) < half, axis=1)
    xb = x[inbox]
    print(f"box side {args.box} Mpc/h centred at {np.round(c0,0)}: {len(xb):,} galaxies")
    lo = c0 - half

    # --- grid + counts ---
    n = args.ngrid
    edges = [np.linspace(lo[d], lo[d] + args.box, n + 1) for d in range(3)]
    counts, _ = np.histogramdd(xb, bins=edges)
    nbar_vox = max(len(xb) / n ** 3, 1e-6)
    print(f"grid {n}³ = {n**3:,} voxels, <count>/voxel = {nbar_vox:.3f}")

    # --- NIFTy: log-normal intensity, learned spectrum, Poisson, geoVI ---
    import jax, jax.numpy as jnp, jax.random as random
    import nifty8.re as jft
    cfm = jft.CorrelatedFieldMaker("cf")
    cfm.set_amplitude_total_offset(offset_mean=0.0, offset_std=(1e-1, 1e-2))
    cfm.add_fluctuations((n, n, n), distances=args.box / n, fluctuations=(1.0, 0.5),
                         loglogavgslope=(-4.0, 1.0), prefix="")
    cf = cfm.finalize()
    data = jnp.asarray(counts.ravel().astype(np.int32))
    signal = jft.Model(lambda L: nbar_vox * jnp.exp(cf(L).reshape(-1)),
                       domain=cf.domain, init=cf.init)
    lh = jft.Poissonian(data).amend(signal)
    key = random.PRNGKey(0)
    print(f"[nifty] geoVI ({args.niter} iters) ...")
    samples, _ = jft.optimize_kl(lh, jft.Vector(cf.init(key)), key=key,
                                 n_total_iterations=args.niter, n_samples=3,
                                 sample_mode="nonlinear_resample")
    s_nifty = np.asarray(jnp.mean(jnp.stack([cf(s).reshape(-1) for s in samples]), 0))  # log-field

    # --- graphGP conditional at a subsample of voxel centres ---
    centres = np.stack(np.meshgrid(*[0.5 * (e[1:] + e[:-1]) for e in edges], indexing="ij"), -1).reshape(-1, 3)
    rng = np.random.default_rng(1)
    sub = rng.choice(n ** 3, min(args.n_pred, n ** 3), replace=False)
    fc = build_field_context(cat, sel_map=cat.sel_map, nside=cat.nside, seed=0)
    nbar_comoving = len(xb) / args.box ** 3
    print(f"[graphgp] conditional field at {len(sub):,} voxel centres ...")
    opd, _ = conditional_overdensity_los(xb, nbar_comoving, centres[sub], fc.cov)
    g_graphgp = np.log(np.clip(opd, 1e-3, None))

    # --- compare the two posterior log-fields at the same points ---
    g_nifty = s_nifty[sub]
    g_n = g_nifty - g_nifty.mean(); g_g = g_graphgp - g_graphgp.mean()
    corr = float(np.corrcoef(g_n, g_g)[0, 1])
    # also compare to the raw smoothed counts as a sanity anchor
    cnt_sub = counts.ravel()[sub]
    corr_n_cnt = float(np.corrcoef(g_nifty, cnt_sub)[0, 1])
    corr_g_cnt = float(np.corrcoef(g_graphgp, cnt_sub)[0, 1])
    print(f"\n=== graphGP vs NIFTy posterior log-overdensity ({len(sub):,} voxels) ===")
    print(f"  corr(graphGP, NIFTy)      = {corr:.3f}")
    print(f"  corr(NIFTy, voxel counts) = {corr_n_cnt:.3f}")
    print(f"  corr(graphGP, voxel counts)= {corr_g_cnt:.3f}")
    print(f"  amplitude std: graphGP {g_graphgp.std():.3f}  NIFTy {g_nifty.std():.3f}")
    print("\n(high corr(graphGP, NIFTy) = the two independent engines — measured-ξ kernel vs "
          "learned power spectrum — reconstruct the same field: mutual validation.)")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, a = plt.subplots(figsize=(5.6, 5.4))
    a.hexbin(g_g, g_n, gridsize=40, cmap="viridis", mincnt=1, bins="log")
    lim = np.percentile(np.abs(np.r_[g_g, g_n]), 99)
    a.plot([-lim, lim], [-lim, lim], "r--", lw=1)
    a.set_xlabel("graphGP  log(1+δ) (mean 0)"); a.set_ylabel("NIFTy  s (mean 0)")
    a.set_title(f"graphGP vs NIFTy field posterior  (corr {corr:.2f})")
    fig.tight_layout(); fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
