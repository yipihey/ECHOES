"""M0 foundations: fill_footprint (WHERE to inpaint) + mask-hole injection."""
import numpy as np
import pytest

from echoes.fill_footprint import build_fill_footprint
from echoes.mock_systematics import (inject_mask_holes, apply_survey_systematics_with_holes,
                                      DEFAULT_HOLE_LADDER)


def _patch_randoms(n=200_000, lo=0.0, hi=12.0, hole=(6.0, 6.0, 1.5), seed=0):
    """Uniform randoms over a square patch with a circular hole punched out."""
    rng = np.random.default_rng(seed)
    ra = rng.uniform(lo, hi, n); dec = rng.uniform(lo, hi, n)
    cra, cdec, rad = hole
    d = np.hypot((ra - cra) * np.cos(np.radians(dec)), dec - cdec)
    keep = d > rad
    return ra[keep], dec[keep]


def test_fill_footprint_identifies_and_fills_hole():
    ra_r, dec_r = _patch_randoms()
    z = np.random.default_rng(1).uniform(0.45, 0.70, 50_000)
    fp = build_fill_footprint(ra_random=ra_r, dec_random=dec_r, z_data=z,
                              nside=64, mangle_ply=None, mangle_npy=None, lss_clip_deg=3.0)
    # the punched hole pixel(s) exist: zero observed coverage but inside target_mask
    hole_pixels = (fp.observed_cover == 0) & fp.target_mask
    assert hole_pixels.sum() > 0
    # fill_weight is high in the hole, ~0 in covered area (no double counting)
    assert fp.fill_weight[hole_pixels].min() > 0.9
    covered = fp.observed_cover > 0.5
    assert fp.fill_weight[covered].mean() < 0.05
    # interior-hole pixels were found and the fill area is a few deg^2 (~ hole area)
    assert len(fp.hole_pix) > 0
    assert 1.0 < fp.fill_area_deg2 < 15.0
    # n(z) profile is normalised to mean 1 and evaluable
    assert np.isfinite(fp.nbar_z(0.55)) and abs(fp.nz.mean() - 1.0) < 1e-6


def test_fill_footprint_no_fill_when_complete():
    ra_r, dec_r = _patch_randoms(hole=(6.0, 6.0, 0.0))   # no hole
    z = np.random.default_rng(2).uniform(0.45, 0.70, 40_000)
    fp = build_fill_footprint(ra_random=ra_r, dec_random=dec_r, z_data=z,
                              nside=64, mangle_ply=None, mangle_npy=None, lss_clip_deg=3.0)
    # interior is fully covered -> negligible fill area (only ragged edge pixels)
    assert fp.fill_area_deg2 < fp.target_mask.sum() * 0.05 * \
        __import__("healpy").nside2pixarea(64, degrees=True)


def test_inject_mask_holes_marks_interior_galaxies():
    rng = np.random.default_rng(0)
    ra = rng.uniform(0, 20, 8000); dec = rng.uniform(0, 20, 8000)
    ht = inject_mask_holes(ra, dec, hole_ladder={"a": (1.0, 3)}, seed=3)
    assert ht.in_hole.sum() > 0
    # every removed galaxy is within the hole radius and has a positive into-hole depth
    re = ht.r_edge_deg[ht.in_hole]
    assert np.all(re > 0) and np.all(re <= 1.0 + 1e-6)
    assert np.all(ht.hole_id[ht.in_hole] >= 0)
    assert np.all(np.isnan(ht.r_edge_deg[~ht.in_hole]))
    # each removed galaxy is genuinely within 1 deg of its hole centre
    c = ht.hole_id[ht.in_hole]
    dd = np.hypot((ra[ht.in_hole] - ht.center_ra[c]) * np.cos(np.radians(dec[ht.in_hole])),
                  dec[ht.in_hole] - ht.center_dec[c])
    assert dd.max() <= 1.0 + 1e-6


def test_apply_with_holes_removes_truth_and_keeps_targets_valid():
    rng = np.random.default_rng(0)
    n = 12000
    ra = rng.uniform(0, 15, n); dec = rng.uniform(0, 15, n); z = rng.uniform(0.45, 0.70, n)
    colors = rng.normal(0, 1, (n, 4)); mags = rng.normal(20, 1, (n, 5))
    wsys = np.ones(n)
    obs, tg, kept, true_z, ht = apply_survey_systematics_with_holes(
        ra, dec, z, colors, mags, wsys, hole_ladder={"big": (1.5, 4)}, hole_seed=1, seed=0)
    # holes removed some galaxies from the truth, recorded with their true z
    assert ht.in_hole.sum() > 0
    assert ht.removed_z is not None and len(ht.removed_z) == ht.in_hole.sum()
    # none of the hole-removed positions survive into the observed catalogue
    obs_set = set(map(tuple, np.column_stack([obs.ra_data, obs.dec_data]).round(6)))
    rem_set = set(map(tuple, np.column_stack([ht.removed_ra, ht.removed_dec]).round(6)))
    assert obs_set.isdisjoint(rem_set)
    # targets still index validly into the (post-hole) observed catalogue
    if tg.N > 0:
        assert tg.host_index.max() < obs.N_data and tg.host_index.min() >= 0


def test_default_ladder_spans_scales():
    radii = [r for r, _ in DEFAULT_HOLE_LADDER.values()]
    assert min(radii) < 0.05 and max(radii) >= 1.0   # arcmin star masks to degree voids
