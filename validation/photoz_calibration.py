"""Is the per-galaxy calibration limit the photo-z posterior itself?

The field-conditional completion samples z ∝ field(z)·p_photoz(z)·[close-pair], so
if the photo-z posterior p_photoz is mis-calibrated (over- or under-confident), the
completion inherits it — and our diagnostics traced the residual per-galaxy
miscalibration to the photo-z, not the field. This test checks the photo-z
directly: on held-out CMASS galaxies with known spec-z, the PIT of the true
redshift within the PhotoZKNN posterior should be uniform. If it is not, we find
the single width-recalibration factor s that makes it uniform (a standard photo-z
recalibration: scale each posterior's deviations from its mean by s), which is the
quantity to fold back into the completion's photo-z likelihood.

    JAX_PLATFORMS=cpu OMP_NUM_THREADS=16 ~/.venv/k3d/bin/python3 \
        validation/photoz_calibration.py
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from echoes.surveys.boss import load_boss
from echoes.photoz import PhotoZKNN, photoz_features
from echoes.pit import pit_uniformity, format_pit

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"


def pit_of(zk, wk, ztrue, scale=1.0):
    """Weighted-CDF PIT of the true z within each kNN photo-z posterior, with the
    posterior optionally widened/narrowed about its mean by ``scale``."""
    pit = np.empty(len(ztrue))
    for i in range(len(ztrue)):
        z = zk[i]; w = wk[i]
        ok = np.isfinite(w) & (w > 0) & np.isfinite(z)
        if not ok.any():
            pit[i] = np.nan; continue
        z = z[ok]; w = w[ok] / w[ok].sum()
        mu = np.sum(w * z)
        zs = mu + scale * (z - mu)                       # width-rescaled posterior
        pit[i] = float(np.sum(w[zs < ztrue[i]]))
    return pit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--n-test", type=int, default=20000)
    args = p.parse_args()

    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    z = np.asarray(cat.z_data)
    feat = photoz_features(cat.colors_data, cat.mags_data)
    good = np.isfinite(feat).all(1) & (cat.imatch_data == 1)
    idx = np.flatnonzero(good)
    rng = np.random.default_rng(0); rng.shuffle(idx)
    n_test = min(args.n_test, len(idx) // 3)
    test, train = idx[:n_test], idx[n_test:]
    pz = PhotoZKNN(k=args.k).fit(feat[train], z[train])
    zk, wk = pz.posterior(feat[test]); ztrue = z[test]
    print(f"photo-z trained on {len(train):,}, tested on {len(test):,}")

    pit1 = pit_of(zk, wk, ztrue, scale=1.0)
    pu1 = pit_uniformity(pit1)
    print(f"\nphoto-z PIT (as used):  {format_pit(pu1)}")
    std = pu1["std"]
    verdict = ("OVER-confident (U-shaped; posteriors too narrow)" if std > 0.30 else
               "UNDER-confident (∩-shaped; posteriors too wide)" if std < 0.27 else
               "≈ calibrated")
    print(f"  PIT std {std:.3f} (0.289 ideal) -> {verdict}")

    # find the width-recalibration that maximises uniformity (min KS to flat)
    scales = np.linspace(0.6, 2.0, 29)
    ks = []
    for s in scales:
        ps = pit_uniformity(pit_of(zk, wk, ztrue, scale=s))
        ks.append(ps["ks"])
    ks = np.array(ks); s_best = scales[ks.argmin()]
    pu_best = pit_uniformity(pit_of(zk, wk, ztrue, scale=s_best))
    print(f"\nbest width recalibration s = {s_best:.2f}  (s>1 widens, s<1 narrows)")
    print(f"recalibrated photo-z PIT: {format_pit(pu_best)}")
    print(f"\n(if s≠1 and the recalibrated PIT is uniform, the per-galaxy completion miscalibration "
          f"IS the photo-z width; fold s into p_photoz — bw_p and the kNN spread — in the completion.)")


if __name__ == "__main__":
    main()
