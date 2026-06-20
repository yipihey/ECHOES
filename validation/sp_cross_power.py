"""G2 gate — galaxy × SP cross-correlation of the GENERATED product.

A systematics-free catalog must not correlate with the imaging systematics-
potential (SP) templates. This measures the zero-lag angular cross-correlation
between the generated-galaxy overdensity and each SP map (skyflux, depth, seeing,
airmass, E(B-V); echoes.sp_maps), with jackknife errors, for:

  * observed   — the raw catalog (SP correlation EXISTS here, the thing to remove);
  * gaussian   — generated from the WEIGHT_SYSTOT-weighted field (graphgp_field);
  * transform  — the Tier-A non-Gaussian field (must NOT re-leak SP: the monotonic
                 transform is SP-blind, so |A×| should stay at the gaussian level).

PASS: |A×|/σ_jk < 2 for every SP template on the generated products (consistent
with zero), and the transform does not inflate it over the gaussian field.

    OMP_NUM_THREADS=16 JAX_PLATFORMS=cpu ~/.venv/k3d/bin/python3 \
        validation/sp_cross_power.py
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from echoes.surveys.boss import load_boss
from echoes.graphgp_field import sample_posterior_density_field
from echoes.density_transform import fit_density_transform, DensityTransform
from echoes.sp_maps import load_sp_maps, _pix, NSIDE_SP
from echoes.systematics import JackknifeMap

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"


def overdensity_map(ra, dec, ra_r, dec_r, nside):
    """Pixel galaxy overdensity δ = n/(α·n_r) − 1 over the footprint pixels."""
    npix = 12 * nside ** 2
    ng = np.bincount(_pix(ra, dec, nside), minlength=npix).astype(float)
    nr = np.bincount(_pix(ra_r, dec_r, nside), minlength=npix).astype(float)
    foot = nr > 0
    alpha = ng[foot].sum() / nr[foot].sum()
    delta = np.full(npix, np.nan)
    delta[foot] = ng[foot] / (alpha * nr[foot]) - 1.0
    return delta, foot


def cross_amp(delta, sp, foot, reg, n_reg):
    """Zero-lag cross-correlation A× = corr(δ_g, SP) over footprint pixels, with a
    jackknife error over angular regions. Returns (A×, σ_jk)."""
    pix = np.flatnonzero(foot)
    d = delta[pix]; s = sp[pix]; r = reg[pix]
    ok = np.isfinite(d) & np.isfinite(s)
    pix, d, s, r = pix[ok], d[ok], s[ok], r[ok]
    def corr(dd, ss):
        dd = dd - dd.mean(); ss = ss - ss.mean()
        den = np.sqrt((dd**2).sum() * (ss**2).sum())
        return float((dd*ss).sum() / den) if den > 0 else 0.0
    A = corr(d, s)
    jk = []
    for k in range(n_reg):
        m = r != k
        if m.sum() > 10:
            jk.append(corr(d[m], s[m]))
    jk = np.asarray(jk)
    sig = float(np.sqrt((len(jk)-1)/len(jk) * ((jk - jk.mean())**2).sum())) if len(jk) > 1 else np.nan
    return A, sig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nside", type=int, default=NSIDE_SP)
    p.add_argument("--nz", type=int, default=64)
    p.add_argument("--n-samples", type=int, default=3)
    p.add_argument("--n-jk", type=int, default=48)
    p.add_argument("--cic-R", type=float, default=8.0)
    p.add_argument("--isd", action="store_true",
                   help="fold ISD residual-decontamination weights into the field "
                        "generation (on top of WEIGHT_SYSTOT) — should null the residuals")
    args = p.parse_args()

    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    ra = np.asarray(cat.ra_data); dec = np.asarray(cat.dec_data); z = np.asarray(cat.z_data)
    rar = np.asarray(cat.ra_random); decr = np.asarray(cat.dec_random)
    print(f"loaded CMASS-South: {len(ra):,} galaxies")

    sp = load_sp_maps(RAND, nside=args.nside, verbose=False)
    jk = JackknifeMap(rar, decr, n_reg=args.n_jk)
    npix = 12 * args.nside ** 2
    reg = np.full(npix, -1)
    rpix = _pix(rar, decr, args.nside)
    reg[rpix] = jk.assign(rar, decr)                            # region id per footprint pixel

    # transform fit from the data CiC PDF (purely data-driven)
    from echoes.generative import _cic_overdensity
    dt_emp = fit_density_transform(_cic_overdensity(cat, R=args.cic_R), kind="empirical", scale=args.cic_R)

    w_extra = None
    if args.isd:
        from echoes.sp_maps import isd_decontamination
        isd = isd_decontamination(cat, sp)
        w_extra = isd.weight
        print(f"ISD residual weights folded in: range [{w_extra.min():.3f}, {w_extra.max():.3f}], "
              f"removed {[sp.names[j] for j in isd.removal_order]}")
    print(f"building GraphGP/FKP field (WEIGHT_SYSTOT{'+ISD' if args.isd else ''}-weighted) ...")
    res = sample_posterior_density_field(cat, n_samples=args.n_samples, n_z_bins=args.nz,
                                         nside=args.nside, seed=0, w_extra=w_extra, verbose=False)

    # generated products: gaussian (identity) and transform
    variants = {"gaussian": DensityTransform(kind="identity"), "transform": dt_emp}
    gen = {}
    for name, dt in variants.items():
        tf = None if dt.kind == "identity" else dt.apply_to_field
        cats = [res.sample_catalog(cat, sample_idx=s, seed=200+s, transform=tf)
                for s in range(args.n_samples)]
        gen[name] = cats

    print(f"\n=== |A×| = |corr(δ_g, SP)|  (PASS: |A×| < 2σ_jk; transform ≤ gaussian) ===")
    hdr = f"{'SP template':14s} {'observed':>16s} {'gaussian':>16s} {'transform':>16s}"
    print(hdr)
    worst = {"gaussian": 0.0, "transform": 0.0}
    for nm in sp.names:
        smap = sp.maps[nm]
        # observed reference
        d_o, foot = overdensity_map(ra, dec, rar, decr, args.nside)
        Ao, So = cross_amp(d_o, smap, foot, reg, args.n_jk)
        row = f"{nm:14s} {Ao:7.3f}±{So:5.3f}"
        for name in ("gaussian", "transform"):
            As, Ss = [], []
            for g in gen[name]:
                d, foot = overdensity_map(np.asarray(g["ra"]), np.asarray(g["dec"]), rar, decr, args.nside)
                A, S = cross_amp(d, smap, foot, reg, args.n_jk)
                As.append(A); Ss.append(S)
            Am = float(np.mean(As)); Sm = float(np.mean(Ss))
            nsig = abs(Am)/Sm if Sm > 0 else np.nan
            worst[name] = max(worst[name], nsig)
            row += f" {Am:7.3f}±{Sm:5.3f}"
        print(row)
    print(f"\nworst |A×|/σ_jk:  gaussian={worst['gaussian']:.2f}  transform={worst['transform']:.2f}")
    ok = worst["gaussian"] < 2.0 and worst["transform"] < 2.0 and worst["transform"] <= worst["gaussian"] + 0.5
    print(f"G2 {'PASS' if ok else 'CHECK'}: generated products consistent with zero SP cross-correlation"
          f"{' (transform does not re-leak SP)' if ok else '; inspect per-template'}")


if __name__ == "__main__":
    main()
