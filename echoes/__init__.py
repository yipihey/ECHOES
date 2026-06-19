"""ECHOES: Equal-weight Completed Hypothetical Observation Ensembles.

Survey-ready posterior samples of completed galaxy catalogs. The default engine
is a fast local-density (KNN) redshift completion; an optional graphGP engine
provides a correlated conditional-field posterior (``pip install echoes[graphgp]``).
"""
__version__ = "0.1.0"

from .completion import complete_catalog_photoz, build_gp_field, PROV, PROV_NAME
from .posterior import build_package, write_package, load_package, draw

__all__ = ["complete_catalog_photoz", "build_gp_field", "PROV", "PROV_NAME",
           "build_package", "write_package", "load_package", "draw", "__version__"]
