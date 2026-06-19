#!/usr/bin/env python3
"""Download the MultiDark-Patchy DR12 SGC COMPSAM mocks used for ECHOES
truth-recovery validation (Section: truth-known validation).

Source (public, ~hundreds of MB):
    https://data.sdss.org/sas/dr12/boss/lss/dr12_multidark_patchy_mocks/
        Patchy-Mocks-DR12SGC-COMPSAM_V6C.tar.gz            (the mock galaxy catalogs)
        Patchy-Mocks-Randoms-DR12SGC-COMPSAM_V6C_x10.tar.gz (matching randoms, ~187 MB)

These mocks are NOT redistributed in this repository (too large for GitHub).

    python data/fetch_patchy.py --n-mocks 10 --out data/boss/mocks
"""
import argparse, os, subprocess, urllib.request

BASE = "https://data.sdss.org/sas/dr12/boss/lss/dr12_multidark_patchy_mocks/"
MOCKS = "Patchy-Mocks-DR12SGC-COMPSAM_V6C.tar.gz"
RANDS = "Patchy-Mocks-Randoms-DR12SGC-COMPSAM_V6C_x10.tar.gz"


def _download(name, out):
    dest = os.path.join(out, name)
    if os.path.exists(dest):
        print(f"  have {name}")
        return dest
    print(f"  downloading {name} ...")
    urllib.request.urlretrieve(BASE + name, dest)
    return dest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/boss/mocks")
    p.add_argument("--n-mocks", type=int, default=10,
                   help="extract only the first N mock realizations (full tar has 2048)")
    p.add_argument("--randoms", action="store_true", help="also fetch the x10 randoms (~187 MB)")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tar = _download(MOCKS, args.out)
    # extract the first N mock files only
    members = subprocess.run(["tar", "tzf", tar], stdout=subprocess.PIPE, text=True).stdout.split()
    members = [m for m in members if m.endswith(".dat")][: args.n_mocks]
    subprocess.run(["tar", "xzf", tar, "-C", args.out] + members)
    print(f"extracted {len(members)} mock realizations to {args.out}")
    if args.randoms:
        rtar = _download(RANDS, args.out)
        subprocess.run(["tar", "xzf", rtar, "-C", args.out])
        print("extracted Patchy randoms")
    print("\nNote: ECHOES validation uses ~6-10 mocks; the full set is large.")


if __name__ == "__main__":
    main()
