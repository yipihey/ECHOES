"""Cosmicflows-4 ingest for the true-3D local-neighborhood line (WIP, branch
``data/local-neighborhood``).

Two readers, deliberately thin — they turn the fetched CF4 products (``data/fetch_cf4.py``)
into arrays the ECHOES 3D machinery already understands (comoving positions; a gridded
field). The completion/posterior layer is built on top in later phases (see
``docs/local_neighborhood.md``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

CF4_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "local", "cf4")
VELOCITY_SCALE = 52.0           # IP2I velocity cubes are stored x1/52; multiply -> km/s
BOX_MPC_H = 1000.0              # public 64^3 cube is a 1000 Mpc/h box
NVOX = 64


@dataclass
class CF4Field:
    delta: np.ndarray           # (64,64,64) over-density
    velocity: np.ndarray        # (3,64,64,64) peculiar velocity [km/s] (x52 applied)
    box_mpc_h: float            # box side [Mpc/h]
    nvox: int                   # voxels per side
    axis_order: str = "SGZ,SGY,SGX"     # supergalactic, per the IP2I convention

    @property
    def voxel_mpc_h(self):
        return self.box_mpc_h / self.nvox


def read_cf4_field(cf4_dir=CF4_DIR, tag="CF4_new_64-z008"):
    """Reconstructed δ + 3D peculiar-velocity cubes (Courtois et al. 2023), velocity in km/s."""
    from astropy.io import fits
    delta = np.asarray(fits.getdata(os.path.join(cf4_dir, f"{tag}_delta.fits")), np.float32)
    vel = np.asarray(fits.getdata(os.path.join(cf4_dir, f"{tag}_velocity.fits")), np.float32)
    return CF4Field(delta=delta, velocity=vel * VELOCITY_SCALE,
                    box_mpc_h=BOX_MPC_H, nvox=delta.shape[0])


def _dm_to_mpc(dm):
    """Distance modulus -> distance [Mpc].  DM = 5 log10(d/Mpc) + 25."""
    return 10.0 ** ((np.asarray(dm, float) - 25.0) / 5.0)


@dataclass
class CF4Catalog:
    pgc: np.ndarray             # PGC id
    ra: np.ndarray              # RA J2000 [deg]
    dec: np.ndarray             # Dec J2000 [deg]
    dist_mpc: np.ndarray        # REAL distance from the combined distance modulus [Mpc]
    vcmb: np.ndarray            # CMB-frame velocity [km/s]
    xyz: np.ndarray             # (N,3) equatorial comoving Cartesian [Mpc] at the real distance


def read_cf4_catalog(cf4_dir=CF4_DIR, table="cf4_table2.fits"):
    """Individual CF4 distances → real comoving positions (true 3D, peculiar-velocity-corrected).

    Distance is from the combined distance modulus ``DM`` (not from redshift), so positions are
    physical/true-3D rather than redshift-space. Rows without a finite ``DM`` are dropped."""
    from astropy.io import fits
    d = fits.getdata(os.path.join(cf4_dir, table))
    dm = np.asarray(d["DM"], float)
    ok = np.isfinite(dm) & (dm > 0)
    ra = np.asarray(d["RAJ2000"], float)[ok]
    dec = np.asarray(d["DEJ2000"], float)[ok]
    dist = _dm_to_mpc(dm[ok])
    r = np.radians(ra); dd = np.radians(dec); cd = np.cos(dd)
    xyz = np.column_stack([dist * cd * np.cos(r), dist * cd * np.sin(r), dist * np.sin(dd)])
    return CF4Catalog(pgc=np.asarray(d["PGC"])[ok], ra=ra, dec=dec, dist_mpc=dist,
                      vcmb=np.asarray(d["Vcmb"], float)[ok], xyz=xyz.astype(np.float32))


if __name__ == "__main__":          # quick smoke check on the fetched data
    f = read_cf4_field()
    c = read_cf4_catalog()
    print(f"field: delta {f.delta.shape} in [{f.delta.min():.2f},{f.delta.max():.2f}], "
          f"|v| up to {np.linalg.norm(f.velocity, axis=0).max():.0f} km/s, voxel {f.voxel_mpc_h:.1f} Mpc/h")
    print(f"catalog: {len(c.ra):,} galaxies with real distances, "
          f"median {np.median(c.dist_mpc):.0f} Mpc, max {c.dist_mpc.max():.0f} Mpc")
