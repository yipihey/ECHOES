"""The Survey interface that every ECHOES survey loader satisfies.

A survey loader returns a catalog object exposing, at minimum, the observed
galaxies (``ra_data``, ``dec_data``, ``z_data``), a matching random catalog
(``ra_random``, ``dec_random``, ``z_random``), and an angular
completeness/selection map (``sel_map`` with ``nside``). See
``echoes/surveys/boss.py`` for the reference (BOSS DR12 CMASS) implementation
and ``docs/adding_a_survey.md`` for the full contract.

Beyond the required fields below, loaders attach **optional** attributes that
consumers read defensively via ``getattr(cat, name, None)`` (so different
datasets carry only what they have). The conventional optional fields are:

- ``w_sys_data, w_cp_data, w_noz_data`` — completeness-weight components used to
  build the missing-target list (``restore_missing`` mode; BOSS, DESI).
- ``colors_data, mags_data, colors_finite`` — photometric features for the
  colour-space photo-z (BOSS). Absent => field-only redshift posterior.
- ``z_data_err`` — per-object redshift error, the photo-z 1σ width that the
  ``resample_redshift`` mode draws from (Quaia's ``redshift_quaia_err``).
- ``xyz_data, xyz_random, fid_cosmo`` — Cartesian coordinates + fiducial cosmology.
- ``imatch_data, icollided_data`` — provenance bookkeeping.

Optional fields are intentionally NOT declared on the Protocol so that
``runtime_checkable`` ``isinstance`` checks remain structural on the required
fields only.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SurveyCatalog(Protocol):
    ra_data: np.ndarray
    dec_data: np.ndarray
    z_data: np.ndarray
    ra_random: np.ndarray
    dec_random: np.ndarray
    z_random: np.ndarray
    sel_map: np.ndarray
    nside: int
