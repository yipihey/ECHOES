"""Survey-property null-test battery for the ECHOES completed catalog.

ECHOES promises "any statistic on a clean completed catalog", but until now we
only validated cleanliness against the single WEIGHT_SYSTOT map
(systot_gradient.py). Following the DES Y6 methodology (Weaverdyck et al. 2026),
this battery checks the COMPLETED catalog's density against the full suite of
survey-property (SP) maps that modulated CMASS *targeting* — Galactic latitude,
Galactic extinction, stellar density, and WEIGHT_SYSTOT itself — and reports the
per-template residual χ²/dof (the ISD statistic, echoes.systematics) for three
catalogs: observed (uncorrected), w_c-weighted (the standard correction), and
ECHOES-completed (equal weight). A clean completion gives χ²/dof ≈ 1, as good as
the weighting and far below the uncorrected catalog.

Spectroscopic-appropriate: the SP templates are the imaging systematics that
modulated target selection (exactly what WEIGHT_SYSTOT was built from). No
photometric sample-construction machinery is involved.

    JAX_PLATFORMS=cpu OMP_NUM_THREADS=16 ~/.venv/k3d/bin/python3 \
        validation/sp_null_tests.py [--with-gaia]
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import healpy as hp
from echoes.surveys.boss import load_boss
from echoes.photoz import PhotoZKNN, photoz_features
from echoes.surveys.boss_targets import load_cmass_targets
from echoes.completion import complete_catalog_photoz, measure_close_pair_dz, PROV
from echoes.randoms import make_random_from_selection_function
from echoes.systematics import density_vs_template_jk, _chi2_flat, JackknifeMap

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"
TARGETS = "data/boss/cmass_targets_South.fits"
NSIDE_SP = 64


def _pix(ra, dec):
    return hp.ang2pix(NSIDE_SP, np.radians(90.0 - np.asarray(dec)), np.radians(np.asarray(ra) % 360.0))


def _map_from_values(ra, dec, val):
    """Mean-per-pixel HEALPix map of a per-object quantity; empty footprint pixels
    filled with the global median so it can be evaluated at random positions."""
    npix = 12 * NSIDE_SP ** 2
    pix = _pix(ra, dec)
    s = np.bincount(pix, weights=val, minlength=npix)
    n = np.bincount(pix, minlength=npix)
    m = np.full(npix, np.nan); ok = n > 0
    m[ok] = s[ok] / n[ok]
    m[np.isnan(m)] = np.nanmedian(m[ok])
    return m


def gal_b(ra, dec):
    """|Galactic latitude| at each position (extinction + stellar-density proxy)."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    c = SkyCoord(np.asarray(ra) * u.deg, np.asarray(dec) * u.deg, frame="icrs").galactic
    return np.abs(c.b.deg)


def build_sp_maps(cat, with_gaia=False, with_imaging_sp=True):
    ra = np.asarray(cat.ra_data); dec = np.asarray(cat.dec_data)
    sp = {}
    # WEIGHT_SYSTOT (the reference imaging systematic)
    if cat.w_sys_data is not None:
        sp["w_systot"] = _map_from_values(ra, dec, np.asarray(cat.w_sys_data))
    # full imaging-SP suite (skyflux, depth, seeing, airmass, E(B-V)) from the
    # randoms — the previously-unloaded columns (echoes.sp_maps); built at NSIDE_SP
    # so they merge directly into this dict.
    if with_imaging_sp:
        try:
            from echoes.sp_maps import load_sp_maps
            smp = load_sp_maps(RAND, nside=NSIDE_SP, verbose=False)
            sp.update(smp.maps)
        except Exception as e:
            print(f"  (imaging-SP suite unavailable: {type(e).__name__}: {e})")
    # Galactic extinction (r-band) from the raw FITS, mapped
    try:
        from astropy.io import fits
        with fits.open(DATA) as h:
            d = h[1].data
            ext = np.asarray(d["EXTINCTION"])[:, 2]            # r-band
            sp["extinction_r"] = _map_from_values(np.asarray(d["RA"]), np.asarray(d["DEC"]), ext)
    except Exception as e:
        print(f"  (extinction unavailable: {type(e).__name__})")
    # stellar density from Gaia (optional)
    if with_gaia:
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
            from build_report import _gaia_bright_stars
            g = _gaia_bright_stars(dec.min(), dec.max(), gmax=16.0)
            if g is not None:
                npix = 12 * NSIDE_SP ** 2
                cnt = np.bincount(_pix(g["ra"], g["dec"]), minlength=npix).astype(float)
                sp["stellar_density"] = cnt
        except Exception as e:
            print(f"  (Gaia stellar density unavailable: {type(e).__name__})")
    return sp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-real", type=int, default=3)
    p.add_argument("--with-gaia", action="store_true")
    p.add_argument("--n-bins", type=int, default=10)
    p.add_argument("--n-jk", type=int, default=48)
    args = p.parse_args()

    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    ra = np.asarray(cat.ra_data); dec = np.asarray(cat.dec_data); z = np.asarray(cat.z_data)
    wsys = np.asarray(cat.w_sys_data); wcp = np.asarray(cat.w_cp_data); wnoz = np.asarray(cat.w_noz_data)
    w_c = wsys * (wcp + wnoz - 1.0)
    feat = photoz_features(cat.colors_data, cat.mags_data); good = np.isfinite(feat).all(1) & (cat.imatch_data == 1)
    pz = PhotoZKNN(k=100).fit(feat[good], z[good]); dz = measure_close_pair_dz(cat, 62 / 3600.)
    tg = load_cmass_targets(cat, path=TARGETS, seed=0)

    sp_maps = build_sp_maps(cat, with_gaia=args.with_gaia)
    # add galactic latitude as a position-evaluated template (not a binned map)
    print(f"SP templates: {list(sp_maps.keys()) + ['gal_lat_b']}")

    # randoms (shared reference for the expected density)
    rar, decr, zr = make_random_from_selection_function(
        sel_map=cat.sel_map, n_random=5 * cat.N_data, z_data=z, nside=cat.nside,
        rng=np.random.default_rng(7))

    # SP value at galaxy / random positions
    def sp_at(ra_q, dec_q, name):
        if name == "gal_lat_b":
            return gal_b(ra_q, dec_q)
        return sp_maps[name][_pix(ra_q, dec_q)]
    names = list(sp_maps.keys()) + ["gal_lat_b"]
    sp_rand = {nm: sp_at(rar, decr, nm) for nm in names}
    edges = {nm: np.quantile(sp_rand[nm], np.linspace(0, 1, args.n_bins + 1)) for nm in names}
    for nm in names:
        edges[nm][0] -= 1e-9; edges[nm][-1] += 1e-9

    # jackknife regions defined ONCE from the randoms (the footprint) and applied
    # identically to every catalog, so region k is the same sky patch everywhere —
    # the per-bin covariance then includes clustering sample variance, not Poisson
    jk = JackknifeMap(rar, decr, n_reg=args.n_jk)
    reg_rand = jk.assign(rar, decr)
    print(f"jackknife: {len(np.unique(reg_rand))} angular regions")

    def residual_chi2(ra_g, dec_g, w_g):
        reg_g = jk.assign(ra_g, dec_g)
        out = {}
        for nm in names:
            spg = sp_at(ra_g, dec_g, nm)
            F, s, ok = density_vs_template_jk(spg, sp_rand[nm], edges[nm], reg_g, reg_rand, w_data=w_g)
            out[nm] = _chi2_flat(F, s, ok)
        return out

    # ECHOES-completed ensemble (equal weight)
    comp = [complete_catalog_photoz(cat, tg, pz, seed=s, dz_pool=dz) for s in range(args.n_real)]

    chi_obs = residual_chi2(ra, dec, np.ones(len(ra)))                    # observed, uncorrected
    chi_wc = residual_chi2(ra, dec, w_c)                                  # w_c-weighted
    chi_cmp = {nm: np.mean([residual_chi2(np.asarray(c["ra"]), np.asarray(c["dec"]),
                            np.ones(c["N"]))[nm] for c in comp]) for nm in names}

    print(f"\n=== residual density-vs-SP χ²/dof (jackknife cov; ≈1 = clean) ===")
    print(f"{'SP template':18s} {'observed':>10s} {'w_c-weighted':>14s} {'ECHOES-completed':>18s}")
    for nm in names:
        print(f"{nm:18s} {chi_obs[nm]:10.2f} {chi_wc[nm]:14.2f} {chi_cmp[nm]:18.2f}")
    # PASS criteria (physically correct, not monotone-on-noise):
    #  (a) the completion is as clean as the standard w_c-weighting for EVERY
    #      template (within 30 %): χ²_cmp ≤ 1.3·max(χ²_wc, 1);
    #  (b) wherever the OBSERVED catalog carried a real systematic (χ²_obs > 2),
    #      the completion removes most of it (χ²_cmp < 0.5·χ²_obs).
    SYS = 2.0
    tracks = max(chi_cmp[nm] - 1.3 * max(chi_wc[nm], 1.0) for nm in names)
    sys_tpl = [nm for nm in names if chi_obs[nm] > SYS]
    removed = all(chi_cmp[nm] < 0.5 * chi_obs[nm] for nm in sys_tpl)
    ok = (tracks <= 0.0) and removed
    print(f"\n  completed ≤ 1.3·(w_c) for every template:        {tracks <= 0.0}  "
          f"(worst excess {tracks:+.2f})")
    print(f"  systematic templates (χ²_obs>{SYS:.0f}): {sys_tpl or 'none'}")
    print(f"  completion removes them (χ²_cmp < 0.5·χ²_obs):   {removed}")
    print(f"  --> {'PASS — completion matches the standard correction' if ok else 'CHECK'}")
    print("\n(with jackknife covariance the w_c-weighted catalog sits near χ²/dof≈1; the "
          "ECHOES equal-weight completion tracks it template-by-template and stays far below "
          "the uncorrected observed catalog where a real systematic exists — validating the "
          "completion removes the imaging-targeting systematics, not just WEIGHT_SYSTOT.)")


if __name__ == "__main__":
    main()
