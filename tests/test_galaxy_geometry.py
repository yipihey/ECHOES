"""Geometry + survey-selection enrichment for the textured local viewer (data-free)."""
import numpy as np

from echoes.surveys.galaxy_geometry import (
    enrich_geometry, survey_preference, angular_size_arcmin, absolute_k_mag,
    SURVEY_LEGACY, SURVEY_PS1, SURVEY_DSS2, SURVEY_2MASS)


def test_angular_size_is_finite_and_shrinks_with_distance():
    # same galaxy luminosity, farther away → smaller on sky (monotone)
    k_near, k_far = 8.0, 8.0 + 5 * np.log10(100.0 / 10.0)   # same M_K at 10 vs 100 Mpc
    a_near = angular_size_arcmin([k_near], [10.0])[0]
    a_far = angular_size_arcmin([k_far], [100.0])[0]
    assert np.isfinite(a_near) and np.isfinite(a_far)
    assert a_far < a_near
    # absolute mag identity
    assert abs(absolute_k_mag([k_far], [100.0])[0] - absolute_k_mag([k_near], [10.0])[0]) < 1e-6


def test_survey_preference_partitions_sky():
    # Legacy footprint (off-plane, Dec in DECam range) → Legacy first
    pref = survey_preference([180.0], [0.0])[0]
    assert pref[0] == SURVEY_LEGACY
    # far north (outside DECam DR10 Dec<34) → PS1 first, no Legacy
    pref = survey_preference([180.0], [55.0])[0]
    assert pref[0] == SURVEY_PS1 and SURVEY_LEGACY not in pref
    # Zone of Avoidance (|b|<10) → near-IR 2MASS first
    pref = survey_preference([280.0], [-5.0])[0]    # low galactic latitude
    assert pref[0] == SURVEY_2MASS
    # every galaxy has an all-sky fallback available
    assert SURVEY_DSS2 in pref or SURVEY_2MASS in pref


def test_enrich_defaults_circular_without_xmatch():
    rng = np.random.default_rng(0)
    n = 200
    ra = rng.uniform(0, 360, n); dec = rng.uniform(-80, 80, n)
    dist = rng.uniform(20, 300, n); k = rng.uniform(8, 11.5, n)
    g = enrich_geometry(ra, dec, dist, k)
    assert g.ang_size_arcmin.shape == (n,) and np.all(np.isfinite(g.ang_size_arcmin))
    assert np.all(g.b_a == 1.0) and np.all(g.pa_deg == 0.0)          # circular default
    assert np.all(g.geom_source == "estimated")
    assert g.survey_pref.shape == (n, 4)


def test_enrich_xmatch_overrides_size_and_shape():
    # one galaxy with an SGA match → its measured D25/b_a/PA override the estimate
    ra = np.array([150.0, 200.0]); dec = np.array([10.0, -20.0])
    dist = np.array([50.0, 80.0]); k = np.array([9.0, 10.0])
    sga = {"ra": np.array([150.0 + 1e-4]), "dec": np.array([10.0 - 1e-4]),
           "d25_arcmin": np.array([3.5]), "b_a": np.array([0.4]), "pa_deg": np.array([57.0]),
           "morph": np.array(["Sb"], dtype=object)}
    g = enrich_geometry(ra, dec, dist, k, sga=sga)
    assert abs(g.ang_size_arcmin[0] - 3.5) < 1e-4
    assert abs(g.b_a[0] - 0.4) < 1e-4 and abs(g.pa_deg[0] - 57.0) < 1e-4
    assert g.geom_source[0] == "sga" and g.morph[0] == "Sb"
    assert g.geom_source[1] == "estimated"                          # unmatched → estimate
