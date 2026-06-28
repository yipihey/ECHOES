"""Import side-effect module: populate the dataset registry.

Importing :mod:`echoes.datasets` imports every dataset module, firing each one's
``register(DatasetSpec(...))`` so :func:`echoes.surveys.registry.available` lists
them. The pipeline (``pipeline/build_release.py``) imports this once, then drives
any dataset uniformly by name.

Add a new dataset by writing ``echoes/surveys/<name>.py`` (loader + ``register``)
and importing it here.
"""
from __future__ import annotations

from .surveys import boss          # noqa: F401  -> registers "boss_cmass_south"

for _mod in ("desi", "quaia"):     # registered as they are added (Phase 5/6)
    try:
        __import__(f"echoes.surveys.{_mod}")
    except ImportError:
        pass

from .surveys.registry import available, get  # noqa: F401, E402

__all__ = ["available", "get"]
