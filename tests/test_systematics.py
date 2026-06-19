"""ISD systematic decontamination + LSS-template-rejection."""
import numpy as np

from echoes.systematics import (isd_fit, lss_template_check, density_vs_template,
                                 density_vs_template_jk, _chi2_flat, JackknifeMap)


def _inject(a, Nr=60000, seed=0, ntpl=1, sys_tpl=0):
    """Randoms uniform in [0,1]^ntpl; data = randoms thinned by a KNOWN linear
    systematic in template `sys_tpl`: density ∝ (1 + a·(t-0.5))."""
    rng = np.random.default_rng(seed)
    R = rng.uniform(0, 1, (Nr, ntpl))
    rate = 1.0 + a * (R[:, sys_tpl] - 0.5)
    keep = rng.uniform(0, 1, Nr) < rate / rate.max()
    return R[keep], R


def test_isd_recovers_injected_systematic():
    D, R = _inject(a=0.8)
    res = isd_fit(D, R, n_bins=12, thresh=2.0)
    # before: a clear systematic; after: removed (chi2/dof ~ 1); template was removed
    assert res.chi2_before[0] > 5.0
    assert res.chi2_after[0] < 2.0
    assert 0 in res.removal_order and res.clean
    # the weight undoes the injected (1 + a(t-0.5)): weighted density is flat
    F0, s0, ok0 = density_vs_template(D[:, 0], R[:, 0], res.edges[0])
    Fw, sw, okw = density_vs_template(D[:, 0], R[:, 0], res.edges[0], w_data=res.weight)
    assert np.nanstd(Fw[okw]) < 0.5 * np.nanstd(F0[ok0])


def test_isd_clean_field_is_clean():
    D, R = _inject(a=0.0)                               # no systematic
    res = isd_fit(D, R, n_bins=12, thresh=2.0)
    assert res.chi2_before[0] < 2.5
    assert res.clean and len(res.removal_order) == 0


def test_isd_removes_only_the_systematic_template():
    # template 1 carries the systematic, template 0 is clean (independent)
    D, R = _inject(a=1.0, ntpl=2, sys_tpl=1)
    res = isd_fit(D, R, n_bins=12, thresh=2.0, names=["clean", "sys"])
    assert res.chi2_before[1] > res.chi2_before[0]
    assert 1 in res.removal_order
    assert res.chi2_after[1] < 2.0


def test_jackknife_cov_beats_poisson_on_clustered_field():
    """On a clustered field, Poisson errors over-state χ²/dof for a flat trend;
    the jackknife covariance (which sees the sample variance) brings it near 1."""
    rng = np.random.default_rng(3)
    # build a clustered field on the sphere: random patches with boosted density,
    # NO dependence on the template -> the true trend vs template is flat.
    n_clump = 40
    cra = rng.uniform(0, 30, n_clump); cdec = rng.uniform(0, 30, n_clump)
    ra_d, dec_d = [], []
    for k in range(n_clump):
        m = rng.poisson(400)
        ra_d.append(cra[k] + rng.normal(0, 0.6, m)); dec_d.append(cdec[k] + rng.normal(0, 0.6, m))
    ra_d = np.concatenate(ra_d); dec_d = np.concatenate(dec_d)
    # randoms uniform over the same window; template = a smooth sky coordinate
    ra_r = rng.uniform(-2, 32, 200000); dec_r = rng.uniform(-2, 32, 200000)
    tmpl = lambda ra, dec: ra                              # template independent of clumping
    edges = np.quantile(tmpl(ra_r, dec_r), np.linspace(0, 1, 9)); edges[0] -= 1e-9; edges[-1] += 1e-9
    F, sP, ok = density_vs_template(tmpl(ra_d, dec_d), tmpl(ra_r, dec_r), edges)
    jk = JackknifeMap(ra_r, dec_r, n_reg=24, nside=16)
    Fj, sJ, okj = density_vs_template_jk(tmpl(ra_d, dec_d), tmpl(ra_r, dec_r), edges,
                                         jk.assign(ra_d, dec_d), jk.assign(ra_r, dec_r))
    chi2_poisson = _chi2_flat(F, sP, ok)
    chi2_jk = _chi2_flat(Fj, sJ, okj)
    # the trend is truly flat, but clumping inflates Poisson χ²/dof well above 1;
    # jackknife errors absorb the sample variance and are larger -> lower χ²/dof.
    assert chi2_jk < chi2_poisson
    assert np.nanmedian(sJ[okj]) > np.nanmedian(sP[ok])


def test_lss_template_check_flags_structure():
    rng = np.random.default_rng(1)
    delta = rng.normal(0, 1, 4000)
    tpl_lss = delta + rng.normal(0, 0.3, 4000)          # traces the real density
    tpl_rand = rng.normal(0, 1, 4000)                   # independent SP template
    r1, p1, rej1 = lss_template_check(tpl_lss, delta)
    r2, p2, rej2 = lss_template_check(tpl_rand, delta)
    assert rej1 and abs(r1) > 0.5
    assert not rej2
