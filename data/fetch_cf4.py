"""Fetch Cosmicflows-4 (CF4) data for the local-neighborhood true-3D ECHOES line.

CF4 gives REAL distances + peculiar velocities of ~56,000 galaxies and a reconstructed
3D density + velocity field of the local universe — the ingredients to place galaxies at
their true comoving positions (not redshift space) and to condition a 3D completion.

Two products are fetched into ``data/local/cf4/``:

  1. Reconstructed field cubes (IP2I Lyon portal) — Courtois et al. 2023, A&A 670, L15
     (arXiv:2211.16390). 64^3 voxels on a 1000 Mpc/h box (~15.6 Mpc/h/voxel), supergalactic
     axis order (SGZ, SGY, SGX). delta = over-density; velocity = 3D peculiar velocity.
     IMPORTANT: the stored velocity values (and their errors) must be MULTIPLIED BY 52
     to recover km/s (a portal scaling convention). Error cubes are the per-voxel std.
  2. The CF4 distance/velocity catalog via VizieR (J/ApJ/944/94) — Tully et al. 2023,
     ApJ 944, 94 (arXiv:2209.11238): individual distances (table2) + groups (peculiar
     velocities). astroquery.Vizier; no row cap.

Cite: Courtois et al. 2023 (fields); Tully et al. 2023 (catalog). See
docs/local_neighborhood.md for the data landscape (CF4 / BORG / Manticore) and the plan.

    python data/fetch_cf4.py                # cubes + catalog
    python data/fetch_cf4.py --cubes-only   # just the reconstructed fields
"""
import argparse, os, ssl, urllib.request

IP2I = "https://projets.ip2i.in2p3.fr/cosmicflows/"
OUT = os.path.join(os.path.dirname(__file__), "local", "cf4")
VELOCITY_SCALE = 52.0                 # multiply the stored velocity cubes by this -> km/s

# ungrouped reconstruction at z~0.08 (depth ~240-300 Mpc/h); "CF4gp_" = grouped variant.
CUBES = [
    "CF4_new_64-z008_delta.fits",
    "CF4_new_64-z008_delta_error.fits",
    "CF4_new_64-z008_velocity.fits",
    "CF4_new_64-z008_velocity_cube_error.fits",
]


def _ctx():
    c = ssl.create_default_context()      # IP2I's cert chain is occasionally incomplete
    c.check_hostname = False
    c.verify_mode = ssl.CERT_NONE
    return c


def fetch_cubes():
    os.makedirs(OUT, exist_ok=True)
    ctx = _ctx()
    for f in CUBES:
        dst = os.path.join(OUT, f)
        if os.path.exists(dst):
            print(f"  have {f}"); continue
        print(f"  downloading {f} ...", flush=True)
        with urllib.request.urlopen(urllib.request.Request(IP2I + f), timeout=120, context=ctx) as r, \
             open(dst, "wb") as out:
            out.write(r.read())
        print(f"    -> {dst} ({os.path.getsize(dst)/1e6:.1f} MB)")
    print(f"  NOTE: multiply the velocity cubes by VELOCITY_SCALE={VELOCITY_SCALE:.0f} to get km/s.")


def fetch_catalog():
    os.makedirs(OUT, exist_ok=True)
    from astroquery.vizier import Vizier
    v = Vizier(columns=["**"]); v.ROW_LIMIT = -1
    print("  querying VizieR J/ApJ/944/94 (Tully+ 2023) ...", flush=True)
    cats = v.get_catalogs("J/ApJ/944/94")
    for t in cats:
        name = t.meta.get("name", "table").split("/")[-1].replace(".", "_")
        dst = os.path.join(OUT, f"cf4_{name}.fits")
        t.write(dst, overwrite=True)
        print(f"    {name}: {len(t):,} rows -> {os.path.basename(dst)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cubes-only", action="store_true")
    ap.add_argument("--catalog-only", action="store_true")
    args = ap.parse_args()
    print(f"fetching CF4 into {OUT}/")
    if not args.catalog_only:
        fetch_cubes()
    if not args.cubes_only:
        try:
            fetch_catalog()
        except Exception as e:
            print(f"  catalog fetch failed ({type(e).__name__}: {e}); "
                  f"VizieR J/ApJ/944/94 or EDD (http://edd.ifa.hawaii.edu) as fallback")
    print("done.")


if __name__ == "__main__":
    main()
