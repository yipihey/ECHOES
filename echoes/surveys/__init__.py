"""Survey loaders for ECHOES. Add a new survey by implementing the
:class:`~echoes.surveys.base.SurveyCatalog` interface (see ``boss.py``)."""
from .base import SurveyCatalog
from .boss import load_boss, BOSSCatalog
from .boss_targets import load_cmass_targets, CMASSTargets

__all__ = ["SurveyCatalog", "load_boss", "BOSSCatalog",
           "load_cmass_targets", "CMASSTargets"]
