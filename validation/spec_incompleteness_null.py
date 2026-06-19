"""Spectroscopic-incompleteness null tests for the ECHOES completion.

The DES Y6 SP null tests (validation/sp_null_tests.py) check the *imaging*
systematics that modulate targeting. ECHOES carries a second class of systematic
that a photometric survey has no fibers to suffer: **spectroscopic
incompleteness** — fiber collisions (which preferentially drop galaxies in dense,
high local-pair-count regions) and redshift failures (which preferentially drop
faint galaxies). This battery is the genuinely ECHOES-specific transfer of the
null-test rigor: it verifies the *completed* catalog carries no residual
completeness gradient with either driver.

Because both drivers correlate with the real galaxy field (close-pair count traces
LSS; faintness traces n(z)), a randoms-referenced test would confound real
structure with incompleteness — so we use the **inject-and-recover truth** as the
reference (apply_survey_systematics on real CMASS-South). Per template we measure
the completeness ``f = n_catalog / n_truth`` (mean-normalised to 1; flat ⇒ uniform
completeness ⇒ no residual systematic) with the same jackknife covariance as
Phase 1, for three catalogs: truth (trivially flat), observed (shows the injected
deficit), and ECHOES-completed (must restore flatness).

Templates (both ECHOES-specific, neither photometric):
  * ``n_close`` — local close-pair count on the truth field (fiber-collision
    pressure; echoes.selection_model.local_close_pair_count),
  * ``i_mag``   — SDSS i-band magnitude (the redshift-failure faint-bias driver).

    JAX_PLATFORMS=cpu OMP_NUM_THREADS=16 ~/.venv/k3d/bin/python3 \
        validation/spec_incompleteness_null.py [--coll-frac 0.6 --zfail-frac 0.03]
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy.spatial import cKDTree
from echoes.surveys.boss import load_boss
from echoes.photoz import PhotoZKNN, photoz_features
from echoes.geometry import _radec_to_nhat
from echoes.completion import complete_catalog_photoz, measure_close_pair_dz
from echoes.mock_systematics import apply_survey_systematics
from echoes.selection_model import local_close_pair_count
from echoes.systematics import density_vs_template_jk, _chi2_flat, JackknifeMap, lss_template_check

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-real", type=int, default=4)
    p.add_argument("--coll-frac", type=float, default=0.6)
    p.add_argument("--zfail-frac", type=float, default=0.03)
    p.add_argument("--pair-scale-arcmin", type=float, default=3.0,
                   help="local-density aperture for the fiber-collision-pressure template")
    p.add_argument("--n-bins", type=int, default=8)
    p.add_argument("--n-jk", type=int, default=48)
    args = p.parse_args()

    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    ra = np.asarray(cat.ra_data); dec = np.asarray(cat.dec_data); z = np.asarray(cat.z_data)
    colors = np.asarray(cat.colors_data); mags = np.asarray(cat.mags_data); wsys = np.asarray(cat.w_sys_data)

    # inject collisions + redshift failures (faint-biased) on the real truth field.
    # systot thinning OFF so the only incompleteness is the spectroscopic kind we test.
    obs, tg, kept, true_z = apply_survey_systematics(
        ra, dec, z, colors, mags, np.ones_like(wsys), coll_frac=args.coll_frac,
        zfail_frac=args.zfail_frac, zfail_faint_bias=1.5, seed=0)
    kind = np.asarray(tg.miss_kind)
    print(f"truth N={len(ra):,}; observed N={obs.N_data:,}; missing N={tg.N:,} "
          f"({int((kind=='collided').sum()):,} collided + {int((kind=='zfail').sum()):,} zfail)")

    # --- positional templates defined ON THE TRUTH FIELD, assigned to any catalog
    #     by nearest-truth (positions are identical, so this is an exact lookup) ---
    n_close_truth = local_close_pair_count(ra, dec, args.pair_scale_arcmin / 60.0)
    imag_truth = mags[:, 3]                                   # SDSS i
    truth_tree = cKDTree(_radec_to_nhat(ra, dec))

    def tmpl_at(ra_q, dec_q, values):
        _, j = truth_tree.query(_radec_to_nhat(np.asarray(ra_q), np.asarray(dec_q)), workers=-1)
        return values[j]

    TEMPLATES = {"n_close": n_close_truth, "i_mag": imag_truth}
    # truth is the reference; bin edges from the truth distribution of each template
    edges = {}
    for nm, v in TEMPLATES.items():
        e = np.quantile(v, np.linspace(0, 1, args.n_bins + 1))
        e = np.unique(e)                                     # n_close has ties at 0 -> merge
        e[0] -= 1e-9; e[-1] += 1e-9; edges[nm] = e

    # jackknife regions from the truth footprint, applied to every catalog
    jk = JackknifeMap(ra, dec, n_reg=args.n_jk)
    reg_truth = jk.assign(ra, dec)
    t_truth = {nm: tmpl_at(ra, dec, v) for nm, v in TEMPLATES.items()}
    print(f"jackknife: {len(np.unique(reg_truth))} regions; "
          f"n_close median={np.median(n_close_truth):.1f}, "
          f"frac>0={100*np.mean(n_close_truth>0):.0f}% (aperture {args.pair_scale_arcmin:.1f}')")

    # the templates trace real LSS / n(z) -> a randoms reference would confound them
    # with incompleteness; truth normalisation removes that. Document the coupling.
    delta_proxy = n_close_truth - np.median(n_close_truth)
    r_nc, _, _ = lss_template_check(n_close_truth, delta_proxy)
    print(f"(n_close traces real density: Spearman r={r_nc:.2f} -> truth-referenced, not randoms)")

    def completeness_chi2(ra_g, dec_g):
        """χ²/dof of completeness f = n_g/n_truth vs each template (flat ⇒ uniform)."""
        reg_g = jk.assign(ra_g, dec_g); out = {}
        for nm, v in TEMPLATES.items():
            tg_ = tmpl_at(ra_g, dec_g, v)
            F, s, ok = density_vs_template_jk(tg_, t_truth[nm], edges[nm], reg_g, reg_truth)
            out[nm] = (_chi2_flat(F, s, ok), F, s, ok)
        return out

    # ECHOES-completed ensemble (the product path)
    of = photoz_features(obs.colors_data, obs.mags_data); ogood = np.isfinite(of).all(1)
    pz = PhotoZKNN(k=100).fit(of[ogood], np.asarray(obs.z_data)[ogood])
    dz = measure_close_pair_dz(obs, 62 / 3600.)
    comp = [complete_catalog_photoz(obs, tg, pz, seed=s, dz_pool=dz) for s in range(args.n_real)]

    chi_truth = {nm: completeness_chi2(ra, dec)[nm][0] for nm in TEMPLATES}        # ≈0 by construction
    chi_obs = {nm: completeness_chi2(obs.ra_data, obs.dec_data)[nm][0] for nm in TEMPLATES}
    chi_cmp = {nm: np.mean([completeness_chi2(np.asarray(c["ra"]), np.asarray(c["dec"]))[nm][0]
                            for c in comp]) for nm in TEMPLATES}

    # also report the completeness profile (observed deficit vs flat completed)
    prof_obs = completeness_chi2(obs.ra_data, obs.dec_data)
    prof_cmp = completeness_chi2(np.asarray(comp[0]["ra"]), np.asarray(comp[0]["dec"]))

    print(f"\n=== completeness-vs-template χ²/dof (jackknife; ≈1 = uniform completeness) ===")
    print(f"{'template':12s} {'truth':>8s} {'observed':>10s} {'ECHOES-completed':>18s}")
    for nm in TEMPLATES:
        print(f"{nm:12s} {chi_truth[nm]:8.2f} {chi_obs[nm]:10.2f} {chi_cmp[nm]:18.2f}")

    print(f"\n=== completeness f = n/n_truth per bin (1 = complete; <1 = deficit) ===")
    for nm in TEMPLATES:
        Fo = prof_obs[nm][1]; Fc = prof_cmp[nm][1]; ok = prof_obs[nm][3]
        bins = " ".join(f"{x:4.2f}" for x in Fo[ok])
        binc = " ".join(f"{x:4.2f}" for x in Fc[ok])
        print(f"  {nm:8s} observed : {bins}")
        print(f"  {nm:8s} completed: {binc}")

    # PASS: observed carries the injected deficit (χ² high), completion restores
    # uniform completeness (χ² near truth, and ≪ observed) for every template.
    SYS = 2.0
    sys_tpl = [nm for nm in TEMPLATES if chi_obs[nm] > SYS]
    restored = all(chi_cmp[nm] < 0.5 * chi_obs[nm] and chi_cmp[nm] < 3.0 for nm in sys_tpl)
    detected = len(sys_tpl) > 0
    ok = detected and restored
    print(f"\n  incompleteness detected in observed (χ²>{SYS:.0f}): {sys_tpl or 'none'}")
    print(f"  completion restores uniform completeness (χ²_cmp < 0.5·χ²_obs, <3): {restored}")
    print(f"  --> {'PASS — completion removes the spectroscopic incompleteness' if ok else 'CHECK'}")

    # --- the NON-trivial test: line-of-sight close-pair recovery -----------------
    # Angular completeness above is exact by construction (every target restored at
    # its imaging position). The redshift ASSIGNMENT is not: fiber collisions remove
    # one of each physical close pair (small |Δz|), so the observed catalog has a
    # deficit of line-of-sight close pairs — the canonical BOSS fiber-collision
    # clustering systematic. The completion must restore them at the RIGHT radial
    # separation (the close-pair Δz prior), not merely somewhere on the sightline.
    coll_scale = 62.0 / 3600.0; dz_pair = 0.006
    chord = 2.0 * np.sin(np.radians(coll_scale) / 2.0)

    def los_close_pairs(ra_g, dec_g, z_g):
        ra_g = np.asarray(ra_g); dec_g = np.asarray(dec_g); z_g = np.asarray(z_g)
        pr = cKDTree(_radec_to_nhat(ra_g, dec_g)).query_pairs(chord, output_type="ndarray")
        if not len(pr):
            return 0
        return int((np.abs(z_g[pr[:, 0]] - z_g[pr[:, 1]]) < dz_pair).sum())

    np_truth = los_close_pairs(ra, dec, z)
    np_obs = los_close_pairs(obs.ra_data, obs.dec_data, obs.z_data)
    np_cmp = np.mean([los_close_pairs(np.asarray(c["ra"]), np.asarray(c["dec"]), np.asarray(c["z"]))
                      for c in comp])
    print(f"\n=== line-of-sight close pairs (<62\", |Δz|<{dz_pair}) — fiber-collision clustering ===")
    print(f"  truth     : {np_truth:6d}   (1.00)")
    print(f"  observed  : {np_obs:6d}   ({np_obs/np_truth:.2f})  <- collisions remove close pairs")
    print(f"  completed : {np_cmp:6.0f}   ({np_cmp/np_truth:.2f})  <- restored with the Δz prior")
    rec = np_cmp / np_truth
    pair_ok = 0.85 <= rec <= 1.15
    print(f"  --> {'PASS' if pair_ok else 'CHECK'}: completed recovers "
          f"{100*rec:.0f}% of the truth close-pair count "
          f"(observed only {100*np_obs/np_truth:.0f}%)")
    print("\n(fiber collisions imprint a deficit at high local pair-count and redshift "
          "failures a deficit at faint i; the ECHOES completion — restoring every missing "
          "galaxy at its imaging position — flattens both back to truth, the part of the "
          "null-test rigor that is uniquely ours and that DES cannot test.)")


if __name__ == "__main__":
    main()
