"""Generic completion targets for the ``resample_redshift`` mode.

In ``restore_missing`` mode each dataset supplies a bespoke missing-target list
(``CMASSTargets``, ``DESITargets``). In ``resample_redshift`` mode (Quaia) there
are no *missing* objects — every observed object is present with a fixed sky
position but an uncertain redshift that we resample. The "target list" is then
just *every* object, each self-hosted, carrying the per-object photo-z posterior
parameters ``(photoz_mu, photoz_sigma)``.

This is dataset-agnostic: any catalog exposing ``ra_data``, ``dec_data``,
``z_data`` and a per-object redshift error ``z_data_err`` can be turned into a
resample target set with :func:`synth_resample_targets`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ResampleTargets:
    """Every object as its own completion target (positions fixed, z resampled)."""
    ra: np.ndarray            # (N,)
    dec: np.ndarray           # (N,)
    colors: np.ndarray        # (N, 0) — no colour-space photo-z in resample mode
    mags: Optional[np.ndarray]    # None
    miss_kind: np.ndarray     # (N,) — all "photoz"
    host_index: np.ndarray    # (N,) — self-host: arange(N)
    photoz_mu: np.ndarray     # (N,) — per-object photo-z central redshift
    photoz_sigma: np.ndarray  # (N,) — per-object photo-z 1σ width

    @property
    def N(self) -> int:
        return len(self.ra)


def synth_resample_targets(catalog) -> ResampleTargets:
    """Build a resample-mode target set: every object in ``catalog``, self-hosted.

    ``catalog`` must expose ``ra_data``, ``dec_data``, ``z_data`` and a per-object
    redshift error ``z_data_err`` (the photo-z 1σ width).
    """
    ra = np.asarray(catalog.ra_data, np.float64)
    dec = np.asarray(catalog.dec_data, np.float64)
    mu = np.asarray(catalog.z_data, np.float64)
    sig = getattr(catalog, "z_data_err", None)
    if sig is None:
        raise ValueError(
            "resample_redshift mode requires a per-object redshift error "
            "`z_data_err` on the catalog (e.g. Quaia's redshift_quaia_err)"
        )
    sig = np.asarray(sig, np.float64)
    n = len(ra)
    return ResampleTargets(
        ra=ra, dec=dec,
        colors=np.empty((n, 0), np.float64), mags=None,
        miss_kind=np.full(n, "photoz"),
        host_index=np.arange(n),
        photoz_mu=mu, photoz_sigma=sig,
    )
