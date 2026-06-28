"""Dataset registry — the seam between reusable ECHOES infrastructure and
dataset-specific tooling.

ECHOES infrastructure (``echoes/posterior.py`` ``build_package``/``draw``, the
field engines, the copula, the released ``.npz`` format) is survey-agnostic.
Everything that *is* survey-specific — how the catalogue is loaded, how the
missing-target list is built, which completion mode applies, which photometric
features feed the photo-z — is captured declaratively in one :class:`DatasetSpec`
per dataset. A dataset module (``echoes/surveys/<name>.py``) ``register()``\\ s its
spec at import time; the pipeline then drives any dataset uniformly via
``registry.get(name)``.

This keeps adding "more and more datasets" to: write a loader + (optionally) a
target builder, then register one spec. No infrastructure changes per dataset.

Two completion modes are supported (see ``echoes/posterior.py``):

- ``"restore_missing"`` — there are spectroscopically-missing objects to restore
  at their imaging positions, with a data-driven redshift (BOSS, DESI).
- ``"resample_redshift"`` — every object is present with a fixed position but an
  uncertain redshift, which is resampled from a per-object posterior (Quaia).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Literal, Optional, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:  # keep the registry import lightweight (defer the distance module)
    from ..distance import DistanceCosmo
    from .base import SurveyCatalog

CompletionMode = Literal["restore_missing", "resample_redshift"]


@runtime_checkable
class CompletionTargets(Protocol):
    """The per-object completion-target contract consumed by
    :func:`echoes.posterior.build_package` and
    :func:`echoes.completion.complete_catalog_photoz`.

    ``CMASSTargets`` (``echoes/surveys/boss_targets.py``) is the reference
    implementation; ``DESITargets`` and the generic resample targets
    (``targets_generic.synth_resample_targets``) satisfy the same shape.

    - ``colors`` may be a ``(M, 0)`` array (no photometry) — the photo-z colour
      factor is then dropped and the redshift posterior is field-only.
    - ``mags`` may be ``None``.
    - ``miss_kind`` values must be keys in ``echoes.completion.PROV``.
    """
    ra: np.ndarray            # (M,)
    dec: np.ndarray           # (M,)
    colors: np.ndarray        # (M, C)  C may be 0
    mags: Optional[np.ndarray]    # (M, B) or None
    miss_kind: np.ndarray     # (M,) str labels (PROV keys)
    host_index: np.ndarray    # (M,) int, -1 if none

    @property
    def N(self) -> int: ...


@dataclass(frozen=True)
class DatasetSpec:
    """Declarative description of one dataset for the ECHOES pipeline.

    The dataset-specific *weight formula* deliberately lives inside the loader
    (BOSS computes ``w_*_data``; DESI's reader combines ``WEIGHT``), not here —
    the spec only names the loader and the completion behaviour.
    """
    name: str                                       # registry key, e.g. "boss_cmass_south"
    load_catalog: Callable[..., "SurveyCatalog"]    # the loader (e.g. load_boss)
    load_catalog_kwargs: dict = field(default_factory=dict)
    build_targets: Optional[Callable[..., CompletionTargets]] = None
        # restore_missing: required (e.g. load_cmass_targets);
        # resample_redshift: None -> targets synthesised generically from the catalog
    completion_mode: CompletionMode = "restore_missing"
    footprint_inpaint: bool = False                 # restore objects in undersampled sel-map regions
    photoz_features: Optional[Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]] = None
        # None => no colour-space photo-z (field-only posterior)
    photoz_gaussian: bool = False                   # use per-object N(z, z_err) factor (Quaia)
    field_pool: Literal["observed", "anchors"] = "observed"
        # which objects seed the line-of-sight density prior; "anchors" = lowest-z_err subset
    anchor_frac: float = 0.30                       # field_pool="anchors": fraction kept as anchors
    fid_cosmo: Optional["DistanceCosmo"] = None     # None => loader default
    nside: int = 256
    nq: int = 65                                     # inverse-CDF quantile levels per object
    product_prefix: str = ""                        # output file stem; defaults to name
    default_engine: Literal["field", "generative"] = "field"
    copula: bool = True
    uniform_randoms: bool = False                   # ship uniform-footprint randoms (DESI/Quaia)
                                                    # instead of completeness-imprinted survey randoms

    @property
    def prefix(self) -> str:
        return self.product_prefix or self.name


_REGISTRY: dict[str, DatasetSpec] = {}


def register(spec: DatasetSpec) -> DatasetSpec:
    """Register ``spec`` under ``spec.name`` (idempotent if identical)."""
    existing = _REGISTRY.get(spec.name)
    if existing is not None and existing is not spec:
        raise ValueError(f"dataset {spec.name!r} already registered")
    _REGISTRY[spec.name] = spec
    return spec


def get(name: str) -> DatasetSpec:
    """Return the registered spec for ``name`` (KeyError lists what is available)."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown dataset {name!r}; registered: {available()}"
        ) from None


def available() -> list[str]:
    """Sorted list of registered dataset names."""
    return sorted(_REGISTRY)
