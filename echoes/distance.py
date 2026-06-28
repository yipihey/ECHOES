"""Cosmological distance: (RA, Dec, z) -> comoving (x, y, z), pure NumPy.

For a flat (w0, wa) cosmology with massless neutrinos, the dimensionless
Hubble parameter is

    E^2(z) = Om (1+z)^3 + (1-Om) * (1+z)^{3(1+w0+wa)} * exp(-3 wa z/(1+z))

(Linder 2003 parameterisation). The comoving distance is

    D_C(z) = c/H0 integral_0^z dz' / E(z'),  c/H0 = 2997.92458 / h Mpc

evaluated via cumulative-trapezoid on a dense, fixed redshift grid; the
per-galaxy distance is then a 1D linear interpolation at each sample's z.
The (RA, Dec) angular-position part is the standard spherical-to-Cartesian
transform.

Output convention: ``c/H0`` is divided by ``h`` so distances are in Mpc/h,
the same units our pair-counting and basis machinery use.
"""

from __future__ import annotations

from typing import NamedTuple, Tuple

import numpy as np

C_OVER_H100_MPCH = 2997.92458   # c/(100 km/s/Mpc) [Mpc/h];   D_C[Mpc/h] = (c/H100) * integ


class DistanceCosmo(NamedTuple):
    """Minimal cosmological-distance parameters.

    Notes
    -----
    Massless neutrinos and pure radiation contribution are ignored
    (negligible at z << 100). For survey use cases the dominant
    parameters are ``Om``, ``w0``, ``wa``. ``h`` is carried for
    completeness but distances are returned in Mpc/h, which are
    independent of ``h``.
    """
    Om: float = 0.31
    h: float = 0.68
    w0: float = -1.0
    wa: float = 0.0


def E_of_z(z: np.ndarray, cosmo: DistanceCosmo) -> np.ndarray:
    """Dimensionless Hubble parameter ``E(z) = H(z)/H0``."""
    z = np.asarray(z, np.float64)
    Om, w0, wa = cosmo.Om, cosmo.w0, cosmo.wa
    matter = Om * (1.0 + z) ** 3
    de_exp = 3.0 * (1.0 + w0 + wa)
    de = (1.0 - Om) * (1.0 + z) ** de_exp * np.exp(-3.0 * wa * z / (1.0 + z))
    return np.sqrt(matter + de)


def comoving_distance(
    z: np.ndarray,
    cosmo: DistanceCosmo,
    z_max: float = 4.0,
    n_grid: int = 4000,
) -> np.ndarray:
    """Comoving distance D_C(z) [Mpc/h].

    Builds a cumulative-trapezoid table of ``int_0^z dz'/E(z')`` on a
    fixed ``[0, z_max]`` grid with ``n_grid`` points, then linearly
    interpolates at the sample redshifts. ``c/H0/h`` sets the Mpc/h
    amplitude.
    """
    z_grid = np.linspace(0.0, z_max, n_grid)
    integrand = 1.0 / E_of_z(z_grid, cosmo)
    # Cumulative trapezoid: D[k] = sum_{j<=k} 0.5 (f[j] + f[j+1]) dz
    dz = z_grid[1] - z_grid[0]
    cum = np.concatenate([
        np.array([0.0]),
        np.cumsum(0.5 * (integrand[:-1] + integrand[1:]) * dz),
    ])
    return C_OVER_H100_MPCH * np.interp(np.asarray(z, np.float64), z_grid, cum)


def radec_z_to_cartesian(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    z: np.ndarray,
    cosmo: DistanceCosmo,
    z_max: float = 4.0,
    n_grid: int = 4000,
) -> np.ndarray:
    """Convert (RA, Dec, z) to comoving Cartesian (x, y, z) in Mpc/h.

    Returns shape ``(N, 3)``. ``ra_deg``, ``dec_deg`` are in degrees.
    Right-handed (x, y, z): x is RA=0 / Dec=0, z is the celestial pole.
    """
    ra = np.deg2rad(np.asarray(ra_deg, np.float64))
    dec = np.deg2rad(np.asarray(dec_deg, np.float64))
    D = comoving_distance(z, cosmo, z_max=z_max, n_grid=n_grid)
    cos_dec = np.cos(dec)
    x = D * cos_dec * np.cos(ra)
    y = D * cos_dec * np.sin(ra)
    z_cart = D * np.sin(dec)
    return np.stack([x, y, z_cart], axis=-1)


def cartesian_to_radec_z(
    xyz: np.ndarray,
    cosmo: DistanceCosmo,
    z_max: float = 4.0,
    n_grid: int = 4000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse mapping: comoving (x, y, z) -> (RA[deg], Dec[deg], z).

    Numerically inverts ``D_C(z)`` via a 1D interpolation table.
    """
    xyz = np.asarray(xyz, np.float64)
    D = np.linalg.norm(xyz, axis=-1)
    z_grid = np.linspace(0.0, z_max, n_grid)
    D_grid = comoving_distance(z_grid, cosmo, z_max=z_max, n_grid=n_grid)
    z = np.interp(D, D_grid, z_grid)
    dec = np.arcsin(xyz[..., 2] / (D + 1e-30))
    ra = np.arctan2(xyz[..., 1], xyz[..., 0])
    ra = np.where(ra < 0, ra + 2 * np.pi, ra)
    return np.rad2deg(ra), np.rad2deg(dec), z
