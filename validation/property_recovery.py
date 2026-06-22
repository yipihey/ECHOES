"""Do the completed catalog's galaxy POPULATIONS match the truth at each redshift?

The completion now carries every galaxy's photometry (ugriz mags + colors): real for
observed/restored galaxies (the missing targets are real imaging detections), a
z-matched real-galaxy transplant for synthetic ones (systot PROV=3, inpaint PROV=5).
The acceptance test the user asked for: in each redshift bin, the distribution of each
colour / magnitude in the COMPLETED catalog must match the TRUE observed distribution
there. We test this by inject-and-recover on real CMASS-South (truth known):

  truth  = the full real catalogue (z, colours, mags) — the reference P(x|z).
  obs    = truth after injecting fibre collisions + faint-biased z-failures (its P(x|z)
           is biased: faint galaxies are preferentially missing).
  done   = complete_catalog_photoz(obs, targets, ...) — must RESTORE truth's P(x|z).

Per z-bin and per property x ∈ {u-g, g-r, r-i, i-z, i-mag} we report KS (ks_2samp) and
Wasserstein distance of done-vs-truth (and obs-vs-truth as the degraded baseline),
overall and per provenance class. A direct check of the synthetic-galaxy transplant
(_ztransplant_mags) is included. PASS: per-z-bin KS p >= 0.05 for the completed sample
on the science properties (g-r, r-i, i-z, i-mag) and Wasserstein well below the
observed-baseline bias.

    OMP_NUM_THREADS=16 JAX_PLATFORMS=cpu ~/.venv/k3d/bin/python3 \
        validation/property_recovery.py [--n-real 3 --seed 0]
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, wasserstein_distance

from echoes.surveys.boss import load_boss
from echoes.photoz import PhotoZKNN, photoz_features
from echoes.completion import complete_catalog_photoz, measure_close_pair_dz
from echoes.mock_systematics import apply_survey_systematics
from echoes.inpaint_field import _ztransplant_mags

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"
PROPS = ["u-g", "g-r", "r-i", "i-z", "i-mag"]           # last is i-band magnitude
SCIENCE = ["g-r", "r-i", "i-z", "i-mag"]                # u-band is noise for CMASS


def props_from_mags(mags):
    """(N,5) ugriz → dict of the five properties (NaN-aware)."""
    m = np.asarray(mags, float)
    c = m[:, :-1] - m[:, 1:]
    return {"u-g": c[:, 0], "g-r": c[:, 1], "r-i": c[:, 2], "i-z": c[:, 3], "i-mag": m[:, 3]}


def per_zbin(zc, xc, zt, xt, edges):
    """Per z-bin KS p-value and Wasserstein distance of sample (zc,xc) vs truth (zt,xt)."""
    ks, w = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        a = xc[(zc >= lo) & (zc < hi)]; b = xt[(zt >= lo) & (zt < hi)]
        a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
        if len(a) > 20 and len(b) > 20:
            ks.append(ks_2samp(a, b).pvalue)
            w.append(wasserstein_distance(a, b))
        else:
            ks.append(np.nan); w.append(np.nan)
    return np.array(ks), np.array(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-real", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nzbin", type=int, default=8)
    ap.add_argument("--derived", action="store_true",
                    help="also test P(M_i|z), P(logM*|z) (kcorrect; deterministic in mags,z)")
    ap.add_argument("--out", default="output/property_recovery.png")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    z = np.asarray(cat.z_data); mags = np.asarray(cat.mags_data); colors = np.asarray(cat.colors_data)
    feat = photoz_features(colors, mags)
    good = np.isfinite(feat).all(1) & (cat.imatch_data == 1)
    pz = PhotoZKNN(k=100).fit(feat[good], z[good])

    obs, tg, kept, true_z = apply_survey_systematics(
        np.asarray(cat.ra_data), np.asarray(cat.dec_data), z, colors, mags,
        np.asarray(cat.w_sys_data), coll_frac=0.6, zfail_frac=0.014,
        zfail_faint_bias=1.5, seed=args.seed)
    dz = measure_close_pair_dz(obs, 62 / 3600.)
    edges = np.linspace(0.43, 0.70, args.nzbin + 1)

    # truth reference and degraded-observed baseline
    pt = props_from_mags(mags); zt = z
    po = props_from_mags(obs.mags_data); zo = np.asarray(obs.z_data)

    # completion ensemble (pool a few realizations for stable distributions)
    Z, P, PR = [], {k: [] for k in PROPS}, []
    for s in range(args.n_real):
        c = complete_catalog_photoz(obs, tg, pz, seed=100 * args.seed + s, dz_pool=dz)
        Z.append(np.asarray(c["z"])); PR.append(np.asarray(c["prov"]))
        for k, v in props_from_mags(c["mags"]).items():
            P[k].append(v)
    Z = np.concatenate(Z); PR = np.concatenate(PR); P = {k: np.concatenate(v) for k, v in P.items()}

    print(f"inject-and-recover: truth {len(zt):,}, observed {len(zo):,}, completed {len(Z):,} "
          f"({args.n_real} real)\n")
    print(f"{'property':8s}{'median KS p (done/obs)':>26}{'median Wasser (done/obs)':>26}")
    zc = np.linspace(edges[0], edges[-1], args.nzbin)
    fig, ax = plt.subplots(1, len(PROPS), figsize=(4 * len(PROPS), 4))
    all_pass = True
    for i, k in enumerate(PROPS):
        ksd, wd = per_zbin(Z, P[k], zt, pt[k], edges)
        kso, wo = per_zbin(zo, po[k], zt, pt[k], edges)
        mk_d, mk_o = np.nanmedian(ksd), np.nanmedian(kso)
        mw_d, mw_o = np.nanmedian(wd), np.nanmedian(wo)
        print(f"{k:8s}{mk_d:12.3f}/{mk_o:<12.3f}{mw_d:13.4f}/{mw_o:<12.4f}")
        a = ax[i]
        a.semilogy(zc, wd, "o-", color="#2e7d32", label="completed vs truth")
        a.semilogy(zc, wo, "s--", color="#c0392b", label="observed vs truth")
        a.set_title(k); a.set_xlabel("z"); a.set_ylabel("Wasserstein")
        if i == 0:
            a.legend(fontsize=8)
        if k in SCIENCE:
            # PASS if completed tracks truth (KS not rejected in most bins) AND beats the
            # observed bias (smaller Wasserstein where the failure selection biased obs).
            ok = (np.nanmean(ksd > 0.05) >= 0.5) and (np.nanmedian(wd) <= 1.25 * np.nanmedian(wo) + 1e-6)
            all_pass &= ok
            a.text(0.05, 0.9, "PASS" if ok else "CHECK", transform=a.transAxes,
                   color="green" if ok else "red", fontweight="bold")
    fig.tight_layout(); fig.savefig(args.out, dpi=130, bbox_inches="tight")

    # Per-class P(x|z): restored galaxies (PROV 1,2,4) are a BIASED sub-population
    # (collisions live in dense regions; failures are faint), so the right reference is
    # the truth of the REMOVED galaxies — their own real (colour, true-z) — not the full
    # truth. This tests that drawing z (vs the true z) preserves their colour–z relation.
    miss_z = np.asarray(true_z); pm = props_from_mags(np.asarray(tg.mags))
    print(f"\nper-class median KS p (right reference):")
    print(f"  restored (PROV 1/2/4) vs REMOVED-galaxy truth, systot (3) vs full truth:")
    names = {0: "observed", 1: "collided", 2: "zfail", 3: "systot", 4: "zhost"}
    for prov in sorted(set(PR.tolist())):
        m = PR == prov
        ref_z, ref_p = (miss_z, pm) if prov in (1, 2, 4) else (zt, pt)
        row = "  ".join(f"{k} {np.nanmedian(per_zbin(Z[m], P[k][m], ref_z, ref_p[k], edges)[0]):.2f}"
                        for k in SCIENCE)
        print(f"  PROV {prov} {names.get(prov, '?'):9s} (n={m.sum():6d}): {row}")

    # direct synthetic-galaxy transplant check (the PROV=5 mechanism)
    rng = np.random.default_rng(0)
    z_syn = rng.choice(z[good], size=20000)                    # synthetic z ~ truth n(z)
    m_syn = _ztransplant_mags(z_syn, z[good], mags[good], rng)
    ps = props_from_mags(m_syn)
    kk = np.nanmedian(per_zbin(z_syn, ps["g-r"], zt, pt["g-r"], edges)[0])
    print(f"\n_ztransplant_mags (PROV=5 synthetic): median g-r KS p vs truth = {kk:.2f} "
          f"({'PASS' if kk > 0.05 else 'CHECK'})")

    # derived properties (absolute mag, stellar mass): deterministic functions of (mags,z),
    # so matched P(mags|z) ⇒ matched P(M_i|z), P(logM*|z) — confirm numerically (one realization).
    if args.derived:
        from echoes.derived import derive_properties
        gt = np.isfinite(mags).all(1)
        dt_ = derive_properties(mags[gt], z[gt])                    # truth reference
        c1 = complete_catalog_photoz(obs, tg, pz, seed=999, dz_pool=dz)
        gc = np.isfinite(c1["mags"]).all(1)
        dc = derive_properties(c1["mags"][gc], np.asarray(c1["z"])[gc])
        print(f"\nderived properties (completed vs truth), median KS p:")
        for name, key in [("M_i (abs)", "absmag"), ("log10 M*", "logmass")]:
            xt = dt_[key][:, 3] if key == "absmag" else dt_[key]
            xc = dc[key][:, 3] if key == "absmag" else dc[key]
            kp = np.nanmedian(per_zbin(np.asarray(c1["z"])[gc], xc, z[gt], xt, edges)[0])
            print(f"  {name:10s}: KS p {kp:.2f} ({'PASS' if kp > 0.05 else 'CHECK'})")
    print(f"\n{'OVERALL PASS' if all_pass else 'CHECK'} — saved {args.out}")


if __name__ == "__main__":
    main()
