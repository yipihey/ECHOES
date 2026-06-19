"""Angular 2D kNN-CDF estimators (Banerjee & Abel 2021; Yuan, Abel &
Wechsler 2024) for the experimental ECHOES kNN2D redshift-completion engine.

Pure-observable nearest-neighbour statistics in ``(theta, z)`` space — no
fiducial cosmology, no comoving distances. The joint angular kNN-CDF
``P_{>=k}(theta; z_q, z_n)`` is a single hierarchical pass over the catalog;
every standard clustering observable (mean count, counts-in-cells PMF,
Davis-Peebles / Landy-Szalay xi, sigma^2_clust, higher moments) is a thin
reduction over its cube, collected in :mod:`echoes.knn.derived`.

Ported (with import surgery only) from the graphGP-cosmology research code so
ECHOES can both *drive* the kNN2D completion engine and *close* it — re-measure
the same statistic on the completed catalog and confirm recovery.

Requires ``numba`` (the per-cap kernel) and ``healpy`` (the neighbour lookup).
"""

from __future__ import annotations

from ._kernels import (
    _NUMBA_OK,
    _per_cap_count_kernel,
    _per_cap_count_kernel_per_region,
)
from .cdf import KnnCdfResult, joint_knn_cdf
from . import derived
from . import analytic_rr

__all__ = [
    "joint_knn_cdf",
    "KnnCdfResult",
    "derived",
    "analytic_rr",
    "_per_cap_count_kernel",
    "_per_cap_count_kernel_per_region",
    "_NUMBA_OK",
]
