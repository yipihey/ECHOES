"""Fetch the Manticore-Local posterior field ensemble for the true-3D ECHOES line.

Manticore-Local (McAlpine et al. 2025, MNRAS 540, 716; arXiv:2505.10682) is a Bayesian
field-level reconstruction of our cosmic neighbourhood from 2M++: an 80-member POSTERIOR
ENSEMBLE of constrained density + 3D velocity fields on a 256^3 grid in a 1000 Mpc
observer-centred box (~3.9 Mpc/voxel). It is the recommended conditioning field for the
local-neighborhood completion — each realization is a self-consistent constrained twin of
our actual neighbourhood, and the ensemble carries the reconstruction's uncertainty.

Each ``velocity_fields`` HDF5 (~335 MB) bundles, on the 256^3 grid:
  density (delta+1 over-density), p0/p1/p2 (3 velocity components), num_in_cell (galaxy counts).
Downloaded via the ``manticore_data`` package (open, S3-backed: cosmictwin.org) into
``data/local/manticore/mcmc<i>_velocity.h5``.

    python data/fetch_manticore.py                # default: first 5 realizations (~1.7 GB)
    python data/fetch_manticore.py --n 1          # one (bootstrap)
    python data/fetch_manticore.py --all          # full 80-member ensemble (~27 GB)

Cite: McAlpine et al. 2025 (MNRAS 540, 716). See docs/local_neighborhood.md.
"""
import argparse, os

OUT = os.path.join(os.path.dirname(__file__), "local", "manticore")


def _mcmc_index(f):
    return int(f.key.split("/mcmc_")[1].split("/")[0])


def fetch_velocity_fields(n=5, all_realizations=False):
    import manticore_data as m
    os.makedirs(OUT, exist_ok=True)
    L = m.ManticoreDataLoader()
    files = list(L.get_product_files(m.ManticoreGeneration.Local,
                                     "velocity_fields/R1024", as_file_objects=True))
    files = sorted(files, key=_mcmc_index)
    sel = files if all_realizations else files[:n]
    print(f"manticore-local velocity_fields: {len(files)} realizations available; fetching {len(sel)}")
    for f in sel:
        i = _mcmc_index(f)
        dst = os.path.join(OUT, f"mcmc{i}_velocity.h5")
        if os.path.exists(dst):
            print(f"  have mcmc{i}"); continue
        print(f"  downloading mcmc{i} ({f.key}) ...", flush=True)
        f.download(dst)
        print(f"    -> {dst} ({os.path.getsize(dst)/1e6:.0f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="number of posterior realizations to fetch")
    ap.add_argument("--all", action="store_true", help="fetch the full 80-member ensemble (~27 GB)")
    args = ap.parse_args()
    fetch_velocity_fields(n=args.n, all_realizations=args.all)
    print("done.")


if __name__ == "__main__":
    main()
