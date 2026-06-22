"""Manticore-Local posterior field ensemble reader (WIP, branch data/local-neighborhood).

Turns the fetched Manticore HDF5 fields (``data/fetch_manticore.py``) into the gridded
density + 3D velocity an ECHOES 3D completion conditions on. Each realization is one
posterior constrained twin of our neighbourhood; the ensemble (up to 80) carries the
reconstruction's uncertainty (see docs/local_neighborhood.md).
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np

MANTICORE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "local", "manticore")
BOX_MPC = 1000.0            # observer-centred box side [Mpc]
NVOX = 256                  # inference grid (3.9 Mpc/voxel)


@dataclass
class ManticoreField:
    mcmc: int                   # posterior realization index
    density: np.ndarray         # (256,256,256) over-density 1+δ = ρ/⟨ρ⟩  (mean 1)
    velocity: np.ndarray        # (3,256,256,256) peculiar velocity [km/s] = momentum/ρ
    n_particle: np.ndarray      # (256,256,256) N-body particle counts (the density tracer)
    box_mpc: float = BOX_MPC
    nvox: int = NVOX

    @property
    def voxel_mpc(self):
        return self.box_mpc / self.nvox


def available_realizations(manticore_dir=MANTICORE_DIR):
    out = []
    for p in sorted(glob.glob(os.path.join(manticore_dir, "mcmc*_velocity.h5"))):
        out.append(int(os.path.basename(p).split("mcmc")[1].split("_")[0]))
    return sorted(out)


def manticore_field_context(mcmc=0, manticore_dir=MANTICORE_DIR):
    """A :class:`echoes.field_grid.GriddedFieldContext` for one Manticore realization.

    Frame (validated by maximising 2M++ overdensity alignment, mean 1+δ≈4.5): **equatorial**
    Cartesian, identity axes, observer at the box centre — so positions fed to it must be
    equatorial comoving [Mpc]."""
    from ..field_grid import GriddedFieldContext
    f = read_manticore_field(mcmc, manticore_dir)
    return GriddedFieldContext(delta=f.density, box_mpc=f.box_mpc, velocity=f.velocity,
                               axis_order=(0, 1, 2))


def read_manticore_field(mcmc=0, manticore_dir=MANTICORE_DIR):
    """Density (1+δ) + 3D peculiar-velocity (km/s) of one Manticore-Local posterior realization.

    The HDF5 stores raw mass density ρ and momentum (p0,p1,p2)=ρv; we return the over-density
    ``1+δ = ρ/⟨ρ⟩`` and the velocity ``v = p/ρ`` (km/s) — the physical fields the completion
    conditions on."""
    import h5py
    path = os.path.join(manticore_dir, f"mcmc{mcmc}_velocity.h5")
    with h5py.File(path, "r") as h:
        rho = np.asarray(h["density"], np.float64)
        p = np.stack([np.asarray(h["p0"], np.float64), np.asarray(h["p1"], np.float64),
                      np.asarray(h["p2"], np.float64)], axis=0)
        nic = np.asarray(h["num_in_cell"], np.float32)
    opd = (rho / rho.mean()).astype(np.float32)                 # 1+δ
    vel = (p / np.maximum(rho[None], 1e-30)).astype(np.float32)  # km/s
    return ManticoreField(mcmc=mcmc, density=opd, velocity=vel, n_particle=nic)


if __name__ == "__main__":
    reals = available_realizations()
    print(f"available realizations: {reals[:10]}{' ...' if len(reals) > 10 else ''} ({len(reals)} total)")
    if reals:
        f = read_manticore_field(reals[0])
        print(f"mcmc{f.mcmc}: 1+δ {f.density.shape} mean {f.density.mean():.3f} "
              f"[{f.density.min():.2f},{f.density.max():.0f}], "
              f"|v| median {np.median(np.linalg.norm(f.velocity, axis=0)):.0f} "
              f"max {np.linalg.norm(f.velocity, axis=0).max():.0f} km/s, voxel {f.voxel_mpc:.2f} Mpc")
