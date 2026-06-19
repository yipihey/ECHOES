"""Command-line entry points for ECHOES."""
import argparse
import hashlib
import os
from pathlib import Path
from urllib.request import urlopen

from .posterior import load_package, draw

DEFAULT_PACKAGE = "cmass_south_posterior.npz"
DEFAULT_SHA256 = "de637ade725404db3b9c711c853c6f77dc502efab0e58b28a6803ecd0b95910e"
DEFAULT_URL = (
    "https://raw.githubusercontent.com/yipihey/ECHOES/main/"
    f"data_release/{DEFAULT_PACKAGE}"
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _default_cache_dir() -> Path:
    return Path(os.environ.get("ECHOES_DATA", Path.home() / ".cache" / "echoes")).expanduser()


def _download_default_package(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    print(f"downloading {DEFAULT_PACKAGE} -> {dest}")
    with urlopen(DEFAULT_URL, timeout=30) as r, tmp.open("wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    got = _sha256(tmp)
    if got != DEFAULT_SHA256:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"downloaded {DEFAULT_PACKAGE} has SHA256 {got}, expected {DEFAULT_SHA256}"
        )
    tmp.replace(dest)
    return dest


def _resolve_package(path_arg: str | None) -> Path:
    """Resolve a posterior package path, downloading the default if needed."""
    if path_arg:
        path = Path(path_arg).expanduser()
        if not str(path).endswith(".npz"):
            path = path.with_suffix(path.suffix + ".npz")
        if not path.exists():
            raise SystemExit(f"posterior package not found: {path}")
        return path

    candidates = [
        Path("data_release") / DEFAULT_PACKAGE,
        Path(DEFAULT_PACKAGE),
        _default_cache_dir() / DEFAULT_PACKAGE,
    ]
    for path in candidates:
        if path.exists():
            return path
    try:
        return _download_default_package(candidates[-1])
    except Exception as exc:
        raise SystemExit(
            "could not find or download the default ECHOES posterior package. "
            "Run from a clone of the repository, pass --package PATH, or set "
            "ECHOES_DATA to a directory containing cmass_south_posterior.npz. "
            f"Original error: {exc}"
        ) from exc


def draw_main(argv=None):
    """``echoes-draw`` — draw completed catalogs from a released posterior package."""
    p = argparse.ArgumentParser(
        description="Draw equal-weight completed catalogs from an ECHOES posterior package.")
    p.add_argument("--package", default=None,
                   help="path to the .npz posterior package; defaults to the released CMASS-South package")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=1, help="number of realizations (seeds seed..seed+n-1)")
    p.add_argument("--out", default="catalog.npz", help="output path for a single realization")
    p.add_argument("--out-prefix", default="catalog_", help="prefix for --n>1 (<prefix><seed>.npz)")
    p.add_argument("--no-systot", action="store_true")
    args = p.parse_args(argv)
    pkg = load_package(_resolve_package(args.package))

    def write(cat, path):
        if path.endswith(".fits"):
            try:
                from astropy.table import Table
            except ImportError as exc:
                raise SystemExit(
                    "writing FITS requires astropy; install echoes[fits] or use a .npz output path"
                ) from exc
            Table({"RA": cat["ra"], "DEC": cat["dec"], "Z": cat["z"], "PROV": cat["prov"]}).write(
                path, overwrite=True)
        else:
            import numpy as np
            np.savez_compressed(path, **{k: cat[k] for k in ("ra", "dec", "z", "prov")})

    seeds = range(args.seed, args.seed + args.n)
    for s in seeds:
        cat = draw(pkg, seed=s, systot=not args.no_systot)
        out = args.out if args.n == 1 else f"{args.out_prefix}{s}.npz"
        write(cat, out)
        print(f"seed {s}: {cat['N']:,} galaxies -> {out}")


if __name__ == "__main__":
    draw_main()
