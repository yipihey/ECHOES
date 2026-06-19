"""Residual imaging-systematic (WEIGHT_SYSTOT) bias in the completed catalog.

The equal-weight completed catalog must reproduce the w_systot-**weighted**
density (the standard BOSS systematic correction): in a region of weight
w_systot, E[completed count per observed galaxy] should be w_systot. The legacy
add-only analog step gives max(w_systot,1) instead — it restores the deficit
where w_systot>1 but leaves the w_systot<1 majority (64% of CMASS-South)
un-thinned, so those regions stay over-dense by a factor 1/w_systot relative to
the weighted target, imprinting a degree-scale gradient. The mean-preserving fix
(``systot_thin``) also thins the w_systot<1 regions.

This isolates the systot handling on real CMASS-South: complete with collisions
and redshift failures switched OFF (only the imaging-systematic step acts), bin
the observed galaxies by w_systot, and compare the completed count per bin to the
weighted target Σw_systot. A correct catalog gives completed/weighted = 1 in
every bin; add-only rises above 1 where w_systot<1.

    JAX_PLATFORMS=cpu OMP_NUM_THREADS=16 ~/.venv/k3d/bin/python3 \
        validation/systot_gradient.py
"""
import argparse, dataclasses, os, sys, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from echoes.surveys.boss import load_boss
from echoes.completion import complete_catalog_photoz
from echoes.geometry import _radec_to_nhat

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"


def _empty_targets():
    return types.SimpleNamespace(
        ra=np.zeros(0), dec=np.zeros(0), N=0, host_index=np.zeros(0, int),
        miss_kind=np.array([], dtype="<U8"),
        colors=np.zeros((0, 4)), mags=np.zeros((0, 5)))


class _NoPZ:
    def posterior(self, feat):
        return np.zeros((len(feat), 1)), np.zeros((len(feat), 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-real", type=int, default=6)
    p.add_argument("--out", default="output/systot_gradient.png")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    wsys = np.asarray(cat.w_sys_data)
    N = len(wsys)
    # isolate systot: neutralise the collision/failure weights so w_c == w_systot
    # and pass an empty target list (no restored collisions/failures).
    cat_s = dataclasses.replace(cat, w_cp_data=np.ones(N), w_noz_data=np.ones(N))
    tg = _empty_targets()
    print(f"w_systot: mean={wsys.mean():.4f} median={np.median(wsys):.4f} "
          f"frac<1={100*np.mean(wsys<1):.0f}%  Σw_systot/N={wsys.sum()/N:.4f}")

    otree = cKDTree(_radec_to_nhat(np.asarray(cat.ra_data), np.asarray(cat.dec_data)))
    def wsys_nearest(ra_q, dec_q):
        _, j = otree.query(_radec_to_nhat(np.asarray(ra_q), np.asarray(dec_q)), workers=-1)
        return wsys[j]

    edges = np.quantile(wsys, np.linspace(0, 1, 9)); edges[0] -= 1e-6; edges[-1] += 1e-6
    cen = 0.5 * (edges[1:] + edges[:-1])
    # weighted target per bin: Σ w_systot over observed galaxies in the bin.
    which = np.clip(np.digitize(wsys, edges) - 1, 0, len(cen) - 1)
    weighted = np.bincount(which, weights=wsys, minlength=len(cen))

    def completed_over_weighted(thin):
        out = []
        for s in range(args.n_real):
            c = complete_catalog_photoz(cat_s, tg, _NoPZ(), seed=s, systot_thin=thin)
            wg = wsys_nearest(c["ra"], c["dec"])
            ng = np.histogram(wg, edges)[0].astype(float)
            out.append(ng / weighted)
        return np.array(out)

    r_thin = completed_over_weighted(True)
    r_add = completed_over_weighted(False)

    def report(name, r):
        m, e = r.mean(0), r.std(0) / np.sqrt(len(r))
        lo = m[cen < 1].mean(); hi = m[cen >= 1].mean()
        print(f"\n{name}  (completed / weighted target; 1.0 = correct):")
        for i in range(len(cen)):
            print(f"  w_systot~{cen[i]:.3f}: {m[i]:.4f} ± {e[i]:.4f}")
        print(f"  <ratio | w_systot<1> = {lo:.4f}   | w_systot>=1> = {hi:.4f}   "
              f"max|dev| = {np.max(np.abs(m-1)):.4f}")
        return m, e

    print("\n=== completed / w_systot-weighted density per bin (1.0 = correct) ===")
    m_thin, e_thin = report("systot_thin=True  (mean-preserving fix)", r_thin)
    m_add, e_add = report("systot_thin=False (legacy add-only)", r_add)

    fig, a = plt.subplots(figsize=(7.4, 5.0))
    a.axhline(1.0, color="k", lw=1, ls=":"); a.axvline(1.0, color="gray", lw=0.8)
    a.errorbar(cen, m_add, e_add, fmt="s--", color="#c0392b",
               label="add-only (legacy): over-dense where w_systot<1")
    a.errorbar(cen, m_thin, e_thin, fmt="o-", color="#3a6ea8", lw=2,
               label="mean-preserving (systot_thin): matches weighted")
    a.set_xlabel("WEIGHT_SYSTOT"); a.set_ylabel("completed / w_systot-weighted density")
    a.set_ylim(0.97, 1.12); a.legend(); a.set_title("Imaging-systematic correction in the completed catalog")
    fig.tight_layout(); fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
