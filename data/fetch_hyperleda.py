"""Fetch HyperLEDA galaxy geometry (D25, axis ratio, position angle, morphology) for the
local-neighborhood textured viewer's cross-match.

HyperLEDA (Paturel+ 2003, VizieR ``VII/237/pgc``) is the all-sky reference — crucially it covers
the Zone of Avoidance, where SGA-2020 (Legacy-footprint only) does not. We pull the resolved
galaxies (apparent diameter above a floor) and convert the LEDA log quantities to the convention
``echoes.surveys.galaxy_geometry`` expects:

    D25 [arcmin] = 0.1 · 10^logD25        (logD25 = log of D25 in 0.1-arcmin units)
    b/a          = 10^(−logR25)           (logR25 = log10(a/b))
    PA  [deg]    = PA                       morph = MType

Output: ``data/local/hyperleda/leda_geometry.npz`` (keys ``ra, dec, d25_arcmin, b_a, pa_deg,
morph, pgc``), consumed by ``galaxy_geometry.enrich_geometry(leda=...)`` and
``pipeline/build_texture_atlas.py``. Gitignored (under ``data/local/``); regenerable.

    ~/.venv/k3d/bin/python3 data/fetch_hyperleda.py [--d25-min-arcmin 0.3]
"""
from __future__ import annotations

import argparse
import os

import numpy as np

OUT_DIR = os.path.join("data", "local", "hyperleda")


def fetch(d25_min_arcmin=0.3, verbose=True):
    from astroquery.vizier import Vizier
    import warnings
    warnings.filterwarnings("ignore")

    # logD25 floor for the requested D25 (0.1-arcmin units): logD25 > log10(D25_arcmin/0.1)
    logd25_min = np.log10(max(d25_min_arcmin, 1e-3) / 0.1)
    v = Vizier(catalog="VII/237/pgc", row_limit=-1,
               columns=["_RAJ2000", "_DEJ2000", "PGC", "MType", "logD25", "logR25", "PA"],
               column_filters={"logD25": f">{logd25_min:.3f}"})
    if verbose:
        print(f"[hyperleda] querying VizieR VII/237/pgc (D25 > {d25_min_arcmin}' "
              f"⇒ logD25 > {logd25_min:.2f}) …")
    res = v.query_constraints()
    if not len(res):
        raise RuntimeError("HyperLEDA query returned no tables")
    t = res[0]
    ra = np.asarray(t["_RAJ2000"], float)
    dec = np.asarray(t["_DEJ2000"], float)
    logd25 = np.asarray(t["logD25"], float)
    logr25 = np.ma.filled(np.ma.asarray(t["logR25"], float), 0.0)      # 0 (round) where missing
    pa = np.ma.filled(np.ma.asarray(t["PA"], float), 0.0)
    pgc = np.ma.filled(np.ma.asarray(t["PGC"]), -1).astype(np.int64)
    morph = np.array([str(s).strip() for s in np.ma.filled(t["MType"], "")], dtype=object)

    d25 = 0.1 * 10.0 ** logd25                                          # arcmin
    b_a = np.clip(10.0 ** (-np.abs(logr25)), 0.05, 1.0)
    keep = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(d25) & (d25 > 0)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "leda_geometry.npz")
    np.savez(out, ra=ra[keep].astype(np.float32), dec=dec[keep].astype(np.float32),
             d25_arcmin=d25[keep].astype(np.float32), b_a=b_a[keep].astype(np.float32),
             pa_deg=pa[keep].astype(np.float32), morph=morph[keep], pgc=pgc[keep])
    if verbose:
        print(f"[hyperleda] {keep.sum():,} galaxies → {out} "
              f"(D25 med {np.median(d25[keep]):.2f}', b/a med {np.median(b_a[keep]):.2f})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d25-min-arcmin", type=float, default=0.3,
                    help="only fetch galaxies larger than this (the cross-match reference floor)")
    fetch(ap.parse_args().d25_min_arcmin)


if __name__ == "__main__":
    main()
