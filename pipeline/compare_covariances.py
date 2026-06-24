"""Compare clustering-covariance diagonals across LGCP variants and the external Patchy reference.

Prints, per statistic (wp/xi0/xi2), the per-bin σ ratio of each LGCP covariance to Patchy
(σ_lgcp/σ_patchy; >1 = LGCP over-dispersed), and the median. Patchy is the external truth, so this
is the trust metric the angular jackknife cannot give.

  python pipeline/compare_covariances.py --patchy covariance_patchy_N600.npz \
      covariance_poisson_N1000_cf20.npz covariance_poisson_lognorm_N1000_cf20_sets20.npz
"""
import argparse
import os
import sys

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "data_release", "boss_lgcp_julia", "covariance")


def _load(name):
    p = name if os.path.isabs(name) or os.path.exists(name) else os.path.join(OUT, name)
    return np.load(p), os.path.basename(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patchy", required=True, help="Patchy covariance NPZ (the reference)")
    ap.add_argument("covs", nargs="+", help="LGCP covariance NPZ(s) to compare vs Patchy")
    args = ap.parse_args()

    P, pname = _load(args.patchy)
    print(f"reference = {pname}\n")
    for name in args.covs:
        L, lname = _load(name)
        print(f"=== {lname}   vs   {pname} ===")
        for stat in ("wp", "xi0", "xi2"):
            key = f"cov_{stat}"
            if key not in L or key not in P:
                continue
            sl = np.sqrt(np.clip(np.diag(L[key]), 0, None))
            sp = np.sqrt(np.clip(np.diag(P[key]), 0, None))
            r = np.where(sp > 0, sl / sp, np.nan)               # σ_lgcp / σ_patchy  (>1 = over-dispersed)
            x = L["rp"] if stat == "wp" else L["s"]
            print(f"  {stat:>3}: median σ_lgcp/σ_patchy = {np.nanmedian(r):.2f}  "
                  f"[range {np.nanmin(r):.2f}-{np.nanmax(r):.2f}]")
            with np.printoptions(precision=2, suppress=True, linewidth=200):
                print(f"        bin = {np.asarray(x)}")
                print(f"        ratio = {r}")
        print()


if __name__ == "__main__":
    main()
