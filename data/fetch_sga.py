"""Fetch Siena Galaxy Atlas 2020 (SGA-2020) ellipse geometry for the cross-match (optional).

SGA-2020 (Moustakas+ 2023) is built on the DESI Legacy Surveys and provides, for ~380k resolved
galaxies, accurate elliptical-aperture geometry (D(26) diameter, axis ratio b/a, position angle)
plus precomputed color mosaics. It is the *deep* geometry source where the Legacy footprint covers
a galaxy; HyperLEDA (``data/fetch_hyperleda.py``) is the all-sky fallback incl. the Zone of
Avoidance. SGA is distributed as a parent FITS catalog from NERSC (it is NOT on VizieR), so this
fetcher is **optional and heavier** than the HyperLEDA one — the pipeline runs fine on HyperLEDA
alone, and SGA simply refines the geometry of Legacy-covered galaxies.

Output: ``data/local/sga/sga_geometry.npz`` (keys ``ra, dec, d25_arcmin, b_a, pa_deg, morph,
pgc``), in the same convention as the HyperLEDA table so ``galaxy_geometry.enrich_geometry`` can
take it as the preferred (``sga=``) cross-match. Gitignored; regenerable.

    ~/.venv/k3d/bin/python3 data/fetch_sga.py --url <SGA-2020 ellipse FITS URL>

The default URL points at the SGA-2020 data release; override ``--url`` if the host path changes.
"""
from __future__ import annotations

import argparse
import os
import ssl
import urllib.request

import numpy as np

OUT_DIR = os.path.join("data", "local", "sga")
# SGA-2020 parent catalog (ellipse table). Large (~GB); see https://www.legacysurvey.org/sga/sga2020/
DEFAULT_URL = "https://portal.nersc.gov/project/cosmo/data/sga/2020/SGA-2020.fits"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")


def fetch(url=DEFAULT_URL, verbose=True):
    from astropy.io import fits
    from astropy.table import Table

    os.makedirs(OUT_DIR, exist_ok=True)
    raw = os.path.join(OUT_DIR, os.path.basename(url) or "SGA-2020.fits")
    if not os.path.exists(raw) or os.path.getsize(raw) < 1_000_000:
        if verbose:
            print(f"[sga] downloading {url} → {raw} (large; one-time) …")
        ctx = ssl.create_default_context(); ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE                                # NERSC chain can be partial
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=600, context=ctx) as r, open(raw, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)

    # SGA-2020 columns: RA, DEC, D26 (arcmin), BA, PA, and a morphology/type if present.
    t = Table.read(raw)
    cols = {c.lower(): c for c in t.colnames}

    def col(*names, default=None):
        for n in names:
            if n.lower() in cols:
                return np.asarray(t[cols[n.lower()]])
        return default

    ra = col("RA", "ra"); dec = col("DEC", "dec")
    d26 = col("D26", "DIAM", "d25")                                    # arcmin
    ba = col("BA", "ba", default=np.ones(len(t)))
    pa = col("PA", "pa", default=np.zeros(len(t)))
    pgc = col("PGC", "pgc", default=np.full(len(t), -1))
    morph = col("MORPHTYPE", "TYPE", default=np.array([""] * len(t), dtype=object))
    ra = np.asarray(ra, float); dec = np.asarray(dec, float); d26 = np.asarray(d26, float)
    keep = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(d26) & (d26 > 0)

    out = os.path.join(OUT_DIR, "sga_geometry.npz")
    np.savez(out, ra=ra[keep].astype(np.float32), dec=dec[keep].astype(np.float32),
             d25_arcmin=d26[keep].astype(np.float32),
             b_a=np.clip(np.asarray(ba, float)[keep], 0.05, 1.0).astype(np.float32),
             pa_deg=np.asarray(pa, float)[keep].astype(np.float32),
             morph=np.asarray(morph, dtype=object)[keep],
             pgc=np.asarray(pgc)[keep].astype(np.int64))
    if verbose:
        print(f"[sga] {keep.sum():,} galaxies → {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL, help="SGA-2020 parent FITS URL")
    fetch(ap.parse_args().url)


if __name__ == "__main__":
    main()
