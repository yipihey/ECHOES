"""Survey loaders for ECHOES. Add a new survey by implementing the
:class:`~echoes.surveys.base.SurveyCatalog` interface (see ``boss.py``) and
registering a :class:`~echoes.surveys.registry.DatasetSpec` (see ``registry.py``
and ``docs/adding_a_survey.md``)."""
from .base import SurveyCatalog
from .registry import (DatasetSpec, CompletionTargets, CompletionMode,
                       register, get, available)
from .targets_generic import synth_resample_targets, ResampleTargets
from .boss import load_boss, BOSSCatalog
from .boss_targets import load_cmass_targets, CMASSTargets

__all__ = ["SurveyCatalog", "load_boss", "BOSSCatalog",
           "load_cmass_targets", "CMASSTargets",
           "DatasetSpec", "CompletionTargets", "CompletionMode",
           "register", "get", "available",
           "synth_resample_targets", "ResampleTargets"]
