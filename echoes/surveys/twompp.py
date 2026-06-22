"""2M++ galaxy catalogue ingest for the true-3D local line (WIP).

2M++ (Lavaux & Hudson 2011, MNRAS 416, 2840) is the near-full-sky redshift compilation
(2MASS + SDSS + 6dF) that Manticore-Local was inferred from — so it is the natural anchor
catalogue, and 2M++ galaxies are self-consistent with the Manticore density field. 2M++
gives positions + K-band magnitude + redshift (CMB-frame velocity); true distances come from
correcting the redshift with the reconstructed peculiar-velocity field (Manticore), see
``echoes.surveys.local``. Fetched from VizieR ``J/MNRAS/416/2840`` into ``data/local/2mpp/``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

C_KMS = 299792.458
TWOMPP_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "local", "2mpp")
_CACHE = os.path.join(TWOMPP_DIR, "2mpp_catalog.fits")


def fetch_2mpp(force=False):
    """Download the 2M++ galaxy table from VizieR -> data/local/2mpp/2mpp_catalog.fits."""
    if os.path.exists(_CACHE) and not force:
        return _CACHE
    from astroquery.vizier import Vizier
    os.makedirs(TWOMPP_DIR, exist_ok=True)
    v = Vizier(columns=["**"]); v.ROW_LIMIT = -1
    cats = v.get_catalogs("J/MNRAS/416/2840")
    tab = None
    for t in cats:
        if "catalog" in str(t.meta.get("name", "")):
            tab = t
    if tab is None:
        tab = cats[0]
    tab.write(_CACHE, overwrite=True)
    return _CACHE


@dataclass
class TwoMPPCatalog:
    ra: np.ndarray              # RA J2000 [deg]
    dec: np.ndarray             # Dec J2000 [deg]
    vcmb: np.ndarray            # CMB-frame velocity [km/s]
    ksmag: np.ndarray           # 2MASS K_s apparent magnitude
    gid: np.ndarray             # group id


def read_2mpp(path=None):
    from astropy.io import fits
    path = path or fetch_2mpp()
    d = fits.getdata(path)
    cols = {c.upper(): c for c in d.columns.names}
    ra = np.asarray(d[cols.get("_RA", "_RA")], float)
    dec = np.asarray(d[cols.get("_DE", "_DE")], float)
    vcmb = np.asarray(d[cols["VCMB"]], float)
    ks = np.asarray(d[cols["KSMAG"]], float) if "KSMAG" in cols else np.full(len(ra), np.nan)
    gid = np.asarray(d[cols["GID"]]) if "GID" in cols else np.full(len(ra), -1)
    ok = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(vcmb) & (vcmb > 0)
    return TwoMPPCatalog(ra=ra[ok], dec=dec[ok], vcmb=vcmb[ok], ksmag=ks[ok], gid=gid[ok])


if __name__ == "__main__":
    c = read_2mpp()
    print(f"2M++: {len(c.ra):,} galaxies | cz_cmb median {np.median(c.vcmb):.0f} "
          f"max {c.vcmb.max():.0f} km/s (d~{c.vcmb.max()/68.1:.0f} Mpc) | "
          f"Ksmag {np.nanmedian(c.ksmag):.1f}")
