"""Local-neighborhood true-3D catalog (WIP, branch data/local-neighborhood).

A ``LocalCatalog`` conforming to the ``SurveyCatalog`` protocol but in TRUE 3D: galaxies sit
at their real comoving positions (peculiar-velocity distances), not in redshift space. The
first instance is built from Cosmicflows-4 (direct distance moduli → distances); the 2M++
anchor with field-corrected distances is the next increment (see docs/local_neighborhood.md).
Positions are carried in the **supergalactic** comoving frame (the local-universe convention,
CF4-native), so they align with the CF4 / Manticore field cubes for conditioning.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

C_KMS = 299792.458


def equatorial_to_supergalactic_xyz(ra, dec, dist_mpc):
    """(RA, Dec, distance) -> supergalactic comoving Cartesian (SGX, SGY, SGZ) [Mpc]."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    sg = SkyCoord(np.asarray(ra) * u.deg, np.asarray(dec) * u.deg, frame="icrs").supergalactic
    sgl = sg.sgl.rad; sgb = sg.sgb.rad
    d = np.asarray(dist_mpc, float); cb = np.cos(sgb)
    return np.column_stack([d * cb * np.cos(sgl), d * cb * np.sin(sgl), d * np.sin(sgb)]).astype(np.float32)


@dataclass
class LocalCatalog:
    # --- SurveyCatalog protocol ---
    ra_data: np.ndarray
    dec_data: np.ndarray
    z_data: np.ndarray              # cz/c (CMB frame); the TRUE info is xyz_data/dist_mpc
    ra_random: np.ndarray
    dec_random: np.ndarray
    z_random: np.ndarray
    sel_map: np.ndarray             # HEALPix angular selection (ZoA-masked)
    nside: int
    # --- true-3D extension ---
    xyz_data: np.ndarray            # (N,3) supergalactic comoving [Mpc] at the REAL distance
    dist_mpc: np.ndarray            # (N,) real distance [Mpc]
    # --- completion weights (uniform: no fibre collisions in a distance catalogue) ---
    w_sys_data: np.ndarray = None
    w_cp_data: np.ndarray = None
    w_noz_data: np.ndarray = None
    source: str = "cf4"

    @property
    def N_data(self):
        return len(self.ra_data)


def load_local_cf4(nside=32, zoa_deg=5.0, n_random_mult=4, dmax_mpc=None, seed=0):
    """Build a true-3D ``LocalCatalog`` from the CF4 distance catalogue.

    Positions are supergalactic comoving at the real (distance-modulus) distance. ``sel_map`` is
    the angular coverage of the catalogue (HEALPix pixels containing galaxies) minus the Zone of
    Avoidance (``|b_gal| < zoa_deg``) — the ZoA being the flagship 3D inpaint region. Randoms are
    uniform over ``sel_map`` with the data's radial distribution."""
    import healpy as hp
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from .cf4 import read_cf4_catalog

    c = read_cf4_catalog()
    keep = np.isfinite(c.dist_mpc) & (c.dist_mpc > 0)
    if dmax_mpc:
        keep &= c.dist_mpc <= dmax_mpc
    ra, dec, dist = c.ra[keep], c.dec[keep], c.dist_mpc[keep]
    z = (c.vcmb[keep] / C_KMS).astype(np.float32)
    xyz = equatorial_to_supergalactic_xyz(ra, dec, dist)

    # angular selection: pixels with galaxies, minus the Zone of Avoidance (|b|<zoa)
    theta = np.radians(90.0 - dec); phi = np.radians(ra % 360.0)
    pix = hp.ang2pix(nside, theta, phi)
    sel = np.zeros(hp.nside2npix(nside), np.float32)
    sel[np.unique(pix)] = 1.0
    gl, gb = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
    bgal = SkyCoord(gl * u.deg, gb * u.deg, frame="icrs").galactic.b.deg
    sel[np.abs(bgal) < zoa_deg] = 0.0

    # randoms: uniform over sel_map, radial distribution resampled from the data distances
    rng = np.random.default_rng(seed)
    nr = n_random_mult * len(ra)
    okpix = np.where(sel > 0)[0]
    rp = rng.choice(okpix, nr)
    rth, rph = hp.pix2ang(nside, rp)
    # jitter within the pixel
    rth = np.clip(rth + (rng.random(nr) - 0.5) * hp.nside2resol(nside), 1e-6, np.pi - 1e-6)
    rdec = 90.0 - np.degrees(rth); rra = np.degrees(rph) % 360.0
    rdist = rng.choice(dist, nr)
    rz = (rng.choice(z, nr)).astype(np.float32)

    n = len(ra)
    return LocalCatalog(
        ra_data=ra.astype(np.float32), dec_data=dec.astype(np.float32), z_data=z,
        ra_random=rra.astype(np.float32), dec_random=rdec.astype(np.float32), z_random=rz,
        sel_map=sel, nside=nside, xyz_data=xyz, dist_mpc=dist.astype(np.float32),
        w_sys_data=np.ones(n, np.float32), w_cp_data=np.ones(n, np.float32),
        w_noz_data=np.ones(n, np.float32), source="cf4")
