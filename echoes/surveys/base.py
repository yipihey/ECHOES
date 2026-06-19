"""The Survey interface that every ECHOES survey loader satisfies.

A survey loader returns a catalog object exposing, at minimum, the observed
galaxies (``ra_data``, ``dec_data``, ``z_data``), a matching random catalog
(``ra_random``, ``dec_random``, ``z_random``), an angular completeness/selection
map (``sel_map`` with ``nside``), and the completeness-weight components used to
build the missing-target list (``w_sys_data``, ``w_cp_data``, ``w_noz_data``).
See ``echoes/surveys/boss.py`` for the reference (BOSS DR12 CMASS) implementation
and ``docs/adding_a_survey.md`` for the full contract.
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
