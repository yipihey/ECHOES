"""graphGP vs KNN-KDE on DESI Y1 QSO — the prior-dominated / uneven-coverage head-to-head.

The audit flagged DESI as the seam where the cheap KNN-KDE local-density prior is most
suspect: its coverage-missing objects live in spatially-extended *under-tiled* regions
where the K nearest observed neighbours are far/sparse, so the KNN estimate degrades to a
noisy global n(z). graphGP's conditional field instead reverts to the kernel prior with
honest uncertainty there. This test measures whether that matters.

Inject-and-recover with REAL spectroscopic truth: the full observed DESI QSO sample is
TRUTH. We HIDE a coverage-mimicking subset (preferentially in low-NTILE / high-deficit
regions — exactly the prior-dominated regime), keep the hidden objects at their true sky
positions, and re-assign their redshifts with BOTH engines from the *visible* field:

  * KNN-KDE   — z_mode="field"   (the released DESI engine; field-only, no colours)
  * graphGP   — z_mode="graphgp" (conditional Matheron field over the visible galaxies)

then compare the recovered wp(rp) to TRUTH, with an oracle (hidden at true z) as the floor.

    JAX_PLATFORMS=cpu OMP_NUM_THREADS=16 python validation/graphgp_vs_knn_desi.py
"""
import argparse, glob, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy.spatial import cKDTree

from echoes.surveys.desi import load_desi_qso
from echoes.completion import complete_catalog_photoz, build_gp_field, PROV
from echoes.geometry import _radec_to_nhat
from echoes.clustering import wp_rp


class _ExpCat:
    """Minimal SurveyCatalog for the experiment: conditioning set for both engines."""
    def __init__(self, ra, dec, z, xyz, xyz_r, sel_map, nside, fid_cosmo):
        self.ra_data = ra; self.dec_data = dec; self.z_data = z
        self.xyz_data = xyz; self.xyz_random = xyz_r
        self.w_data = np.ones(len(ra)); self.w_sys_data = None
        self.sel_map = sel_map; self.nside = nside; self.fid_cosmo = fid_cosmo
        self.N_data = len(ra)

    def shift_to_positive(self, margin=100.0):
        a = np.concatenate([self.xyz_data, self.xyz_random], 0)
        s = -a.min(0) + margin
        p, r = self.xyz_data + s, self.xyz_random + s
        return p, r, float(np.max(np.concatenate([p, r]).max(0)) + margin)


class _Targets:
    """Coverage-missing hidden objects (true positions, redshift to be re-assigned)."""
    def __init__(self, ra, dec, host):
        self.ra = ra; self.dec = dec
        self.colors = np.empty((len(ra), 0)); self.mags = None
        self.miss_kind = np.full(len(ra), "coverage"); self.host_index = host

    @property
    def N(self):
        return len(self.ra)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-sub", type=int, default=250_000, help="subsample for a tractable field solve")
    p.add_argument("--hide-frac", type=float, default=0.15)
    p.add_argument("--n-real", type=int, default=4)
    p.add_argument("--nside", type=int, default=64)
    p.add_argument("--nz", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    dat = ["data/desi/QSO_NGC_clustering.dat.fits", "data/desi/QSO_SGC_clustering.dat.fits"]
    ran = sorted(glob.glob("data/desi/QSO_*_clustering.ran.fits"))
    cat = load_desi_qso(dat, ran, nside=args.nside)
    N = cat.N_data
    sub = rng.choice(N, min(args.n_sub, N), replace=False)
    ra, dec, z = cat.ra_data[sub], cat.dec_data[sub], cat.z_data[sub]
    xyz = cat.xyz_data[sub]
    wcomp = cat.w_comp_data[sub]; ntile = cat.ntile_data[sub]
    n = len(ra)
    print(f"DESI truth N={n:,} (subsampled from {N:,})")

    # ---- hide a coverage-mimicking subset: P(hide) ∝ deficit (1 − 1/w_comp) → low-NTILE preferential
    deficit = np.clip(1.0 - 1.0 / np.maximum(wcomp, 1.0), 0.0, None) + 1e-3
    ph = deficit / deficit.sum()
    n_hide = int(args.hide_frac * n)
    hidden = rng.choice(n, n_hide, replace=False, p=ph)
    is_hidden = np.zeros(n, bool); is_hidden[hidden] = True
    vis = ~is_hidden
    print(f"hidden N={n_hide:,} ({100*n_hide/n:.0f}%); NTILE=1 share of hidden={100*np.mean(ntile[hidden]==1):.0f}%"
          f" vs all={100*np.mean(ntile==1):.0f}%  (>1 ⇒ preferentially low-coverage)")

    # randoms for both the field engine and the wp estimator (uniform-ish DESI randoms, subsampled)
    rr = rng.choice(len(cat.ra_random), min(4 * n, len(cat.ra_random)), replace=False)
    rar, decr, zr = cat.ra_random[rr], cat.dec_random[rr], cat.z_random[rr]
    xyz_r = cat.xyz_random[rr]

    mock = _ExpCat(ra[vis], dec[vis], z[vis], xyz[vis], xyz_r, cat.sel_map, args.nside, cat.fid_cosmo)

    # hidden objects: keep TRUE positions, host = nearest visible (redshift fallback only)
    _, host = cKDTree(_radec_to_nhat(ra[vis], dec[vis])).query(_radec_to_nhat(ra[hidden], dec[hidden]))
    tg = _Targets(ra[hidden], dec[hidden], host.astype(int))

    n_vis = int(vis.sum())

    def completed(c):
        d = {"ra": np.asarray(c["ra"]), "dec": np.asarray(c["dec"]), "z": np.asarray(c["z"])}
        d["z_hidden"] = d["z"][n_vis:n_vis + n_hide]      # the re-assigned redshifts of the hidden set
        return d

    # ---- KNN-KDE ensemble (z_mode="field", field-only) ----
    print("[knn] completing (field-only KNN-KDE) ...")
    knn = [completed(complete_catalog_photoz(mock, tg, None, seed=s, dz_pool=np.zeros(1), z_mode="field"))
           for s in range(args.n_real)]

    # ---- graphGP conditional-field ensemble ----
    print(f"[graphgp] building conditional field (n={args.n_real}, nside={args.nside}, nz={args.nz}) ...")
    res = build_gp_field(mock, n_samples=args.n_real, nside=args.nside, n_z_bins=args.nz,
                         r_edges=np.logspace(np.log10(2.0), np.log10(150.0), 28), seed=0, verbose=False)
    ggp = [completed(complete_catalog_photoz(mock, tg, None, seed=s, dz_pool=np.zeros(1),
                                             z_mode="graphgp", gp_field=res)) for s in range(args.n_real)]

    # ---- direct test: did the engine put the HIDDEN objects at the right redshifts? ----
    # The hidden objects have known true z. Compare the ensemble-pooled assigned-z distribution to the
    # true hidden n(z), via KS — overall and in the most prior-dominated regime (NTILE=1). This isolates
    # redshift-assignment quality from the visible-galaxy dilution that wp suffers.
    from scipy.stats import ks_2samp
    z_true_h = z[hidden]; nt1 = ntile[hidden] == 1
    print("\n=== hidden-object n(z) recovery vs TRUTH (KS distance; lower = better) ===")
    for label, mask in [("all hidden", np.ones(n_hide, bool)), (f"NTILE=1 ({nt1.sum():,})", nt1)]:
        zk = np.concatenate([c["z_hidden"][mask] for c in knn])
        zg = np.concatenate([c["z_hidden"][mask] for c in ggp])
        ks_k = ks_2samp(zk, np.tile(z_true_h[mask], args.n_real)).statistic
        ks_g = ks_2samp(zg, np.tile(z_true_h[mask], args.n_real)).statistic
        win = "graphGP" if ks_g < 0.95 * ks_k else "KNN" if ks_k < 0.95 * ks_g else "tie"
        print(f"  {label:18s}  KNN KS={ks_k:.4f}   graphGP KS={ks_g:.4f}   → {win}")

    # ---- references: truth / observed(incomplete) / oracle(true z) ----
    truth = {"ra": ra, "dec": dec, "z": z}
    observed = {"ra": ra[vis], "dec": dec[vis], "z": z[vis]}
    oracle = {"ra": np.concatenate([ra[vis], ra[hidden]]), "dec": np.concatenate([dec[vis], dec[hidden]]),
              "z": np.concatenate([z[vis], z[hidden]])}

    rp_edges = np.logspace(np.log10(0.8), np.log10(40.0), 13); rpc = np.sqrt(rp_edges[1:] * rp_edges[:-1])
    wp_t, RR = wp_rp(truth["ra"], truth["dec"], truth["z"], rar, decr, zr,
                     rp_edges=rp_edges, pimax=60.0, nthreads=16, return_RR=True)

    def wp1(c):
        return wp_rp(c["ra"], c["dec"], c["z"], rar, decr, zr,
                     rp_edges=rp_edges, pimax=60.0, nthreads=16, precomp_RR=RR)

    def wpe(cc):
        return np.mean([wp1(c) for c in cc], 0)

    wp_o, wp_or, wp_k, wp_g = wp1(observed), wp1(oracle), wpe(knn), wpe(ggp)

    ok = np.isfinite(wp_t) & (wp_t > 0) & (rpc > 1.5)     # stable bins (QSO small-rp wp is shot-noisy)

    def err(w):
        r = (w / wp_t)[ok]
        return float(np.median(np.abs(r - 1.0))), float(np.median(r))

    print("\n=== wp(rp) recovered / TRUTH  (closest to 1 = most faithful) ===")
    for name, w in [("observed (incomplete)", wp_o), ("oracle (true z) — floor", wp_or),
                    ("KNN-KDE completion", wp_k), ("graphGP completion", wp_g)]:
        mad, med = err(w)
        print(f"  {name:26s} median ratio {med:.3f}   median |ratio−1| {mad:.4f}")
    print("\n  rp      truth    obs/t   oracle/t   KNN/t    GP/t")
    for i in range(len(rpc)):
        print(f"  {rpc[i]:6.2f}  {wp_t[i]:7.2f}  {wp_o[i]/wp_t[i]:6.3f}  {wp_or[i]/wp_t[i]:7.3f}  "
              f"{wp_k[i]/wp_t[i]:6.3f}  {wp_g[i]/wp_t[i]:6.3f}")
    kmad = err(wp_k)[0]; gmad = err(wp_g)[0]
    verdict = ("graphGP MORE faithful" if gmad < 0.9 * kmad else
               "KNN MORE faithful" if kmad < 0.9 * gmad else "TIE (within 10%)")
    print(f"\nVERDICT (median |wp/truth−1|): KNN={kmad:.4f}  graphGP={gmad:.4f}  → {verdict}")


if __name__ == "__main__":
    main()
