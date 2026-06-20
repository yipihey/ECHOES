"""Correlate galaxy density with the Schlegel-Finkbeiner-Davis (SFD) dust map.

Fetches/queries the actual SFD E(B-V) map (dustmaps) — independent of the catalog's
own SFD-derived EXTINCTION column — and measures the density of the CMASS-South
catalogs as a function of foreground Galactic extinction:

  * spectroscopic sample  (observed BOSS DR12 CMASS galaxies), raw and
    WEIGHT_SYSTOT-corrected, and
  * photometric / photo-z sample (the full CMASS imaging target list).

Density vs E(B-V) is the classic imaging systematic: dust dims background galaxies
below the selection threshold, so the detected density falls with extinction. We
bin the data/random density contrast in E(B-V) deciles with a jackknife covariance
(echoes.systematics) and report the slope, the low-vs-high-decile ratio, and the
Spearman rank correlation, for each sample.

    JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 ~/.venv/k3d/bin/python3 \
        validation/dust_correlation.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy import stats

from echoes.surveys.boss import load_boss
from echoes.systematics import density_vs_template_jk, _chi2_flat, JackknifeMap

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"
TARGETS = "data/boss/cmass_targets_South.fits"
N_BINS = 10
N_JK = 48


def sfd_ebv(ra, dec):
    """SFD E(B-V) at ICRS (ra, dec) [deg] via the real SFD dust map (dustmaps)."""
    os.environ.setdefault("DUSTMAPS_DATA_DIR", os.path.expanduser("~/.dustmaps"))
    from dustmaps.config import config
    config["data_dir"] = os.path.expanduser("~/.dustmaps")
    from dustmaps.sfd import SFDQuery
    c = SkyCoord(np.asarray(ra) * u.deg, np.asarray(dec) * u.deg, frame="icrs")
    return np.asarray(SFDQuery()(c), float)


def _trend(ebv_d, ebv_r, reg_d, reg_r, edges, w_d=None):
    """Density contrast F vs E(B-V) (jackknife), plus slope, lo/hi ratio, Spearman."""
    F, sig, ok = density_vs_template_jk(ebv_d, ebv_r, edges, reg_d, reg_r, w_data=w_d)
    cen = 0.5 * (edges[1:] + edges[:-1])
    m = ok & np.isfinite(sig) & (sig > 0)
    # weighted linear fit F = 1 + slope*(E - <E>)
    slope = np.polyfit(cen[m], F[m], 1, w=1.0 / sig[m])[0]
    lo, hi = F[m][0], F[m][-1]
    return F, sig, ok, cen, slope, lo, hi, _chi2_flat(F, sig, ok)


def main():
    print("loading catalogs + randoms ...")
    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    ra_s = np.asarray(cat.ra_data); dec_s = np.asarray(cat.dec_data)
    wsys = np.asarray(cat.w_sys_data)
    ra_r = np.asarray(cat.ra_random); dec_r = np.asarray(cat.dec_random)
    # photometric / photo-z parent sample: the CMASS imaging target list. It covers a
    # LARGER footprint than the spectroscopic randoms (to Dec~75 vs ~36) and contains
    # star/QSO contaminants, so restrict to the spec sel_map footprint (shared with the
    # randoms) for a valid density-vs-random comparison.
    import healpy as hp
    with fits.open(TARGETS) as h:
        td = h[1].data
        ra_p = np.asarray(td["ra"], float); dec_p = np.asarray(td["dec"], float)
        zw = np.asarray(td["zwarning"]); cls = np.asarray(td["spec_class"]).astype(str)
    pix = hp.ang2pix(cat.nside, np.radians(90 - dec_p), np.radians(ra_p % 360))
    inmask = cat.sel_map[pix] > 0
    n_all = len(ra_p)
    ra_p, dec_p, zw, cls = ra_p[inmask], dec_p[inmask], zw[inmask], cls[inmask]
    spec_ok = zw == 0
    is_gal = spec_ok & (cls == "GALAXY"); is_star = spec_ok & (cls == "STAR")
    ra_pg, dec_pg = ra_p[is_gal], dec_p[is_gal]      # spec-confirmed galaxies among targets
    ra_ps, dec_ps = ra_p[is_star], dec_p[is_star]    # spec-confirmed stellar contaminants
    print(f"  spectroscopic galaxies: {len(ra_s):,}")
    print(f"  photometric targets:    {len(ra_p):,}  ({100*inmask.mean():.0f}% of {n_all:,} in spec footprint)")
    print(f"  randoms:                {len(ra_r):,}")

    print("querying SFD E(B-V) ...")
    ebv_s = sfd_ebv(ra_s, dec_s)
    ebv_p = sfd_ebv(ra_p, dec_p)
    ebv_pg = sfd_ebv(ra_pg, dec_pg)
    ebv_ps = sfd_ebv(ra_ps, dec_ps)
    ebv_r = sfd_ebv(ra_r, dec_r)

    # cross-check on the RAW FITS (RA/DEC/EXTINCTION aligned): SFD E(B-V) vs the
    # catalog's own SFD-derived A_r (A_r = 2.751 E(B-V)).
    with fits.open(DATA) as h:
        hd = h[1].data
        cra = np.asarray(hd["RA"], float); cdec = np.asarray(hd["DEC"], float)
        ar_cat = np.asarray(hd["EXTINCTION"])[:, 2]
    sub = np.random.default_rng(0).choice(len(cra), min(20000, len(cra)), replace=False)
    rr = stats.pearsonr(sfd_ebv(cra[sub], cdec[sub]) * 2.751, ar_cat[sub])
    b = SkyCoord(ra_s * u.deg, dec_s * u.deg, frame="icrs").galactic.b.deg
    print(f"\nSFD E(B-V): median {np.median(ebv_r):.3f}, 1-99% {np.percentile(ebv_r,[1,99]).round(3)} mag")
    print(f"  |b| range (spec): {np.percentile(np.abs(b),[1,50,99]).round(1)} deg")
    print(f"  cross-check 2.751*E(B-V)_SFD vs catalog A_r: Pearson r={rr.statistic:.4f} "
          f"(should be ~1: catalog EXTINCTION is SFD)")

    # shared E(B-V) bin edges (random deciles) + jackknife regions from the randoms
    edges = np.quantile(ebv_r, np.linspace(0, 1, N_BINS + 1)); edges[0] -= 1e-9; edges[-1] += 1e-9
    jk = JackknifeMap(ra_r, dec_r, n_reg=N_JK)
    reg_r = jk.assign(ra_r, dec_r); reg_s = jk.assign(ra_s, dec_s); reg_p = jk.assign(ra_p, dec_p)
    reg_pg = jk.assign(ra_pg, dec_pg); reg_ps = jk.assign(ra_ps, dec_ps)

    samples = [
        ("spectroscopic (raw)",         ebv_s, reg_s, None),
        ("spectroscopic (w_systot)",    ebv_s, reg_s, wsys),
        ("photometric targets (all)",   ebv_p, reg_p, None),
        ("  -> spec-conf galaxies",     ebv_pg, reg_pg, None),
        ("  -> spec-conf stars",        ebv_ps, reg_ps, None),
    ]
    print(f"\n{'sample':28s} {'slope dF/dE':>12s} {'lo/hi decile':>14s} {'Spearman':>10s} {'chi2/dof':>9s}")
    results = {}
    for name, ebv_d, reg_d, w_d in samples:
        F, sig, ok, cen, slope, lo, hi, chi2 = _trend(ebv_d, ebv_r, reg_d, reg_r, edges, w_d)
        rho = stats.spearmanr(ebv_d, np.interp(ebv_d, cen, F)).statistic
        results[name] = (cen, F, sig, ok)
        print(f"{name:28s} {slope:>+12.3f} {lo/hi:>13.3f}  {rho:>+10.3f} {chi2:>9.2f}")

    print("\n(slope = fractional density change per mag E(B-V); negative = dust suppresses "
          "the sample. lo/hi = density in the lowest vs highest E(B-V) decile. The "
          "WEIGHT_SYSTOT row should flatten the spectroscopic trend toward slope~0.)")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8.4, 5.6))
        styles = {"spectroscopic (raw)": ("#c0392b", "o-"),
                  "spectroscopic (w_systot)": ("#3a6ea8", "s-"),
                  "photometric targets (all)": ("#27ae60", "^-"),
                  "  -> spec-conf galaxies": ("#16a085", "v--"),
                  "  -> spec-conf stars": ("#e67e22", "d--")}
        for name, (cen, F, sig, ok) in results.items():
            col, fmt = styles.get(name, ("#888888", "x-"))
            ax.errorbar(cen[ok], F[ok], sig[ok], fmt=fmt, color=col, capsize=2, label=name.strip())
        ax.axhline(1.0, color="k", lw=0.8, ls=":")
        ax.set_yscale("log"); ax.set_yticks([0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0])
        ax.set_yticklabels(["0.4", "0.6", "0.8", "1.0", "1.5", "2.0", "3.0"])
        ax.set_xlabel("SFD E(B-V)  [mag]"); ax.set_ylabel("density / random  (mean-normalised)")
        ax.set_title("CMASS-South density vs Galactic dust (SFD), jackknife errors\n"
                     "spec galaxies flat; raw target rise is stellar contamination")
        ax.legend(fontsize=8); fig.tight_layout()
        os.makedirs("output", exist_ok=True)
        fig.savefig("output/dust_correlation.png", dpi=130, bbox_inches="tight")
        print("\nsaved output/dust_correlation.png")
    except Exception as e:
        print(f"(plot skipped: {type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
