"""Contiguous footprint: _fill_interior_holes fills enclosed holes of ANY size,
leaving only the outer boundary. Data-free (synthetic HEALPix masks)."""
import numpy as np
import pytest

hp = pytest.importorskip("healpy")

from echoes.fill_footprint import _fill_interior_holes, build_fill_footprint


def _disk_mask(nside, ra, dec, radius_deg):
    vec = hp.ang2vec(np.radians(90.0 - dec), np.radians(ra))
    npix = 12 * nside ** 2
    m = np.zeros(npix, bool)
    m[hp.query_disc(nside, vec, np.radians(radius_deg))] = True
    return m


def test_fill_interior_holes_any_size():
    nside = 64
    ra, dec = 150.0, 10.0
    mask = _disk_mask(nside, ra, dec, 15.0)
    # punch a small AND a larger interior hole
    small = _disk_mask(nside, ra + 4, dec + 4, 1.0)
    large = _disk_mask(nside, ra - 3, dec - 2, 4.0)
    holed = mask & ~small & ~large
    filled = _fill_interior_holes(holed, nside)
    # both interior holes are filled, the footprint outside is unchanged
    assert filled[small & mask].all() and filled[large & mask].all()
    assert (filled & ~mask).sum() == 0                 # never extends past the outer boundary
    # idempotent: no interior holes remain
    assert np.array_equal(_fill_interior_holes(filled, nside), filled)


def test_fill_interior_holes_noop_when_solid():
    nside = 32
    mask = _disk_mask(nside, 200.0, 0.0, 20.0)
    assert np.array_equal(_fill_interior_holes(mask, nside), mask)


def test_build_fill_footprint_contiguous_has_no_interior_holes():
    # synthetic survey randoms: a filled disk with a punched veto hole; no mangle file
    nside = 128
    rng = np.random.default_rng(0)
    # uniform points in a disk around (150,10), radius ~12 deg
    n = 200_000
    th = np.radians(90.0 - 10.0); ph = np.radians(150.0)
    u = rng.uniform(np.cos(np.radians(12.0)), 1.0, n)
    a = np.arccos(u); b = rng.uniform(0, 2 * np.pi, n)
    # rotate disk to centre (small-angle local frame is fine for a unit test)
    dec = 10.0 + np.degrees(a) * np.cos(b)
    ra = 150.0 + np.degrees(a) * np.sin(b) / np.cos(np.radians(10.0))
    # punch a veto hole at (152,11)
    keep = np.hypot(ra - 152.0, dec - 11.0) > 1.0
    ra, dec = ra[keep], dec[keep]
    z = rng.uniform(0.45, 0.6, len(ra))
    fp = build_fill_footprint(ra_random=ra, dec_random=dec, z_data=z, nside=nside,
                              mangle_npy=None, contiguous=True)
    cover = fp.observed_cover > 0
    # acceptance: every target_mask pixel is covered OR scheduled to be filled
    unfilled = fp.target_mask & ~cover & ~(fp.fill_weight > 0)
    assert int(unfilled.sum()) == 0
    # and the footprint itself has no interior holes
    assert np.array_equal(_fill_interior_holes(fp.target_mask, nside), fp.target_mask)
    # the punched veto hole is in the fill region
    assert (fp.fill_weight > 0).sum() > 0
