"""Shared SDSS large-scale-structure I/O: clustering-FITS reader and
angular completeness map. Survey-agnostic helpers reused by survey loaders
(extracted from the original DESI/BOSS loaders).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _read_clustering_fits(
    path: str,
    with_weight_fkp: bool = True,
    with_photsys: bool = True,
    with_nx: bool = False,
):
    """Read RA/DEC/Z and combined weight from a DESI clustering FITS.

    Tries the canonical column names in order:
      WEIGHT * WEIGHT_FKP (preferred)
      WEIGHT_SYS * WEIGHT_NOZ * WEIGHT_COMP_TILE * WEIGHT_FKP
      fall back to 1.0

    Returns
    -------
    ra, dec, z, w, photsys, nx
        photsys is a length-N ``'<U1'`` array of 'N'/'S' chars (empty
        if PHOTSYS column absent or with_photsys=False). nx is None
        unless ``with_nx`` is set and the column exists.
    """
    from astropy.io import fits

    with fits.open(path, memmap=True) as hdul:
        t = hdul[1].data
        cols = [c.upper() for c in t.columns.names]

        def col(name):
            return np.asarray(t[name], dtype=np.float64) if name.upper() in cols else None

        ra = col("RA"); dec = col("DEC")
        z = col("Z")
        if ra is None or dec is None or z is None:
            raise ValueError(f"{path}: required columns RA/DEC/Z missing")

        w = col("WEIGHT")
        if w is None:
            ws = col("WEIGHT_SYS"); wn = col("WEIGHT_NOZ"); wc = col("WEIGHT_COMP_TILE")
            w = (ws if ws is not None else 1.0) * \
                (wn if wn is not None else 1.0) * \
                (wc if wc is not None else 1.0)
            if isinstance(w, float):
                w = np.full(len(ra), w)
        if with_weight_fkp:
            w_fkp = col("WEIGHT_FKP")
            if w_fkp is not None:
                w = w * w_fkp

        if with_photsys and "PHOTSYS" in cols:
            raw = np.asarray(t["PHOTSYS"])
            # FITS char columns may come back as bytes ('|S1') or str
            # depending on astropy version; normalise to '<U1'.
            if raw.dtype.kind == "S":
                photsys = np.char.decode(raw, "ascii").astype("U1")
            else:
                photsys = raw.astype("U1")
        else:
            photsys = np.empty(0, dtype="U1")

        nx = col("NX") if with_nx else None

        return ra, dec, z, np.asarray(w, dtype=np.float64), photsys, nx




def angular_completeness_from_randoms(
    ra_random: np.ndarray, dec_random: np.ndarray,
    nside: int = 256, w_random: Optional[np.ndarray] = None,
):
    """Build a HEALPix angular completeness map by binning the random
    catalog at the given NSIDE.

    DESI randoms Poisson-sample the angular survey footprint with
    completeness corrections already imprinted (the random density on
    the sky IS the completeness function). Histogramming randoms at
    HEALPix NSIDE gives a continuous mask normalised to peak = 1.

    If per-random weights are provided, the binned counts are weighted
    sums (this captures z-dependent FKP weight, but for the angular
    mask the WEIGHT product is what matters).
    """
    import healpy as hp

    npix = 12 * nside ** 2
    pix = hp.ang2pix(nside, np.deg2rad(90.0 - dec_random),
                       np.deg2rad(ra_random))
    if w_random is None:
        counts = np.bincount(pix, minlength=npix).astype(np.float64)
    else:
        counts = np.bincount(pix, weights=np.asarray(w_random, dtype=np.float64),
                              minlength=npix).astype(np.float64)
    # normalise to [0, 1] by the median of the populated pixels
    populated = counts[counts > 0]
    if populated.size == 0:
        raise ValueError("no random objects in any pixel")
    rho = np.median(populated)
    mask = np.clip(counts / rho, 0.0, 1.0)
    return mask
