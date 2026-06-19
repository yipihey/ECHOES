"""Command-line entry points for ECHOES."""
import argparse

from .posterior import load_package, draw


def draw_main(argv=None):
    """``echoes-draw`` — draw completed catalogs from a released posterior package."""
    p = argparse.ArgumentParser(
        description="Draw equal-weight completed catalogs from an ECHOES posterior package.")
    p.add_argument("--package", default="data_release/cmass_south_posterior.npz",
                   help="path to the .npz posterior package")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=1, help="number of realizations (seeds seed..seed+n-1)")
    p.add_argument("--out", default="catalog.fits", help="output path for a single realization")
    p.add_argument("--out-prefix", default="catalog_", help="prefix for --n>1 (<prefix><seed>.fits)")
    p.add_argument("--no-systot", action="store_true")
    args = p.parse_args(argv)
    pkg = load_package(args.package)

    def write(cat, path):
        if path.endswith(".fits"):
            from astropy.table import Table
            Table({"RA": cat["ra"], "DEC": cat["dec"], "Z": cat["z"], "PROV": cat["prov"]}).write(
                path, overwrite=True)
        else:
            import numpy as np
            np.savez_compressed(path, **{k: cat[k] for k in ("ra", "dec", "z", "prov")})

    seeds = range(args.seed, args.seed + args.n)
    for s in seeds:
        cat = draw(pkg, seed=s, systot=not args.no_systot)
        out = args.out if args.n == 1 else f"{args.out_prefix}{s}.fits"
        write(cat, out)
        print(f"seed {s}: {cat['N']:,} galaxies -> {out}")


if __name__ == "__main__":
    draw_main()
