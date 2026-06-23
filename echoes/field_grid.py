"""Gridded reconstructed field as an ECHOES conditioning context (true-3D local line).

The BOSS line conditions the completion on a field inferred from a measured ξ(r) kernel
(``echoes.fieldpost.FieldContext``). The local line instead conditions on an *externally
reconstructed* field cube — CF4 (Courtois et al. 2023) or a Manticore-Local posterior
realization (McAlpine et al. 2025) — sampled at any comoving point. ``GriddedFieldContext``
is that sampler: trilinear interpolation of the over-density ``1+δ`` and the 3D peculiar
velocity on an observer-centred cube.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GriddedFieldContext:
    """Observer-centred reconstructed field on a regular cube.

    ``delta`` is ``1+δ`` (mean 1); ``velocity`` is optional (3, n, n, n) in km/s. The cube
    spans ``[-box/2, +box/2]`` per axis about the observer. ``axis_order`` maps the cube's
    storage axes (0,1,2) to the physical (x,y,z) of the sample points — e.g. CF4 cubes are
    stored (SGZ, SGY, SGX) so ``axis_order=(2,1,0)`` when sampling in (SGX, SGY, SGZ)."""
    delta: np.ndarray                 # (n,n,n) 1+δ
    box_mpc: float                    # box side (same length units as the sample points)
    velocity: np.ndarray = None       # (3,n,n,n) km/s, optional
    axis_order: tuple = (0, 1, 2)     # storage-axis order for (x,y,z) samples

    @property
    def nvox(self):
        return self.delta.shape[0]

    def _vox_coords(self, xyz):
        """Physical (...,3) [Mpc, observer-centred] -> fractional cube indices in storage order."""
        xyz = np.atleast_2d(np.asarray(xyz, float))
        frac = (xyz + self.box_mpc / 2.0) / self.box_mpc * self.nvox - 0.5   # per physical axis
        return np.stack([frac[:, self.axis_order[0]], frac[:, self.axis_order[1]],
                         frac[:, self.axis_order[2]]], axis=0)               # (3, N) storage order

    def overdensity_at(self, xyz):
        """Trilinear ``1+δ`` at comoving points ``xyz`` (N,3); outside the box -> nearest edge."""
        from scipy.ndimage import map_coordinates
        return map_coordinates(self.delta, self._vox_coords(xyz), order=1, mode="nearest")

    def velocity_at(self, xyz):
        """Trilinear 3D peculiar velocity (N,3) [km/s] at ``xyz``; requires ``velocity``."""
        if self.velocity is None:
            raise ValueError("this field has no velocity cube")
        from scipy.ndimage import map_coordinates
        c = self._vox_coords(xyz)
        return np.stack([map_coordinates(self.velocity[a], c, order=1, mode="nearest")
                         for a in range(3)], axis=1)
