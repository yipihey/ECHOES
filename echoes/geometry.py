"""Angular geometry helpers shared across the ECHOES pipeline."""
from __future__ import annotations

import numpy as np


def _radec_to_nhat(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """Unit sky vectors (N, 3) from RA, Dec in degrees."""
    ra = np.radians(np.asarray(ra_deg, dtype=np.float64))
    dec = np.radians(np.asarray(dec_deg, dtype=np.float64))
    cd = np.cos(dec)
    return np.stack([cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)], axis=1)
