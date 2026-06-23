"""Does the log-Gaussian completion give the ZoA/faint fills the OBSERVED density contrast?

The fills used to be too smooth (the `intensity='bias'` power-law matched only the mean). The
`intensity='transform'` path treats log(1+δ) as the Gaussian field and maps it through a transform
fit to the OBSERVED galaxy counts-in-cells PDF, so the fill reproduces the observed variance/skew.
This A/B test measures the counts-in-cells PDF (var/mean², skew, shot-noise-free) at the Manticore
voxel scale for: the OBSERVED galaxies, and the painted galaxies under `bias` vs `transform` — in a
nearby complete volume. PASS: transform ≫ bias and approaches observed.

    JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 ~/.venv/k3d/bin/python3 validation/local_contrast.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from echoes.surveys.local import load_local_2mpp
from echoes.surveys.manticore import manticore_field_context, available_realizations
from echoes.local_completion import complete_local_zoa, galactic_b
from echoes.density_transform import field_moments_from_counts

N, L = 256, 1000.0


def to_voxel(ra, dec, dist):
    r = np.radians(ra); d = np.radians(dec); cd = np.cos(d)
    xyz = np.asarray(dist)[:, None] * np.column_stack([cd * np.cos(r), cd * np.sin(r), np.sin(d)])
    idx = np.floor((xyz + L / 2) / L * N).astype(int)
    idx = idx[((idx >= 0) & (idx < N)).all(1)]
    return np.bincount((idx[:, 0] * N + idx[:, 1]) * N + idx[:, 2], minlength=N ** 3).reshape(N, N, N)


def main():
    reals = available_realizations()
    if not reals:
        raise SystemExit("no Manticore realizations fetched — run data/fetch_manticore.py")
    cat = load_local_2mpp(field_mcmc=reals[0], dmax_mpc=300)
    fc = manticore_field_context(reals[0])

    ax = ((np.arange(N) + 0.5) / N * L - L / 2).astype(np.float32)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    d = np.sqrt(X * X + Y * Y + Z * Z)
    ra = np.degrees(np.arctan2(Y, X)) % 360.0
    dec = np.degrees(np.arcsin(np.clip(Z / np.maximum(d, 1e-6), -1.0, 1.0)))
    near = (d > 5) & (d < 100); zoa = np.abs(galactic_b(ra, dec)) < 5

    Nobs = to_voxel(cat.ra_data, cat.dec_data, cat.dist_mpc)
    _, vo, so = field_moments_from_counts(Nobs[near & ~zoa])
    print(f"\ncounts-in-cells at the Manticore voxel scale (3.9 Mpc), d<100, shot-noise-free:")
    print(f"  {'OBSERVED 2M++':22s} var/mean² {vo:6.2f}   skew {so:6.2f}")
    rows = {}
    for it in ("bias", "transform"):
        ip = complete_local_zoa(cat, fc, dmax=300, intensity=it, seed=0)
        Np = to_voxel(ip["ra"], ip["dec"], ip["dist_mpc"])
        _, v, s = field_moments_from_counts(Np[near & zoa])
        rows[it] = (v, s)
        print(f"  {'painted ('+it+')':22s} var/mean² {v:6.2f}   skew {s:6.2f}   (+{len(ip['ra']):,})")
    vt, st = rows["transform"]; vb, sb = rows["bias"]
    ok = vt > 2.0 * vb and vt > 0.4 * vo
    print(f"\n{'PASS' if ok else 'CHECK'}: transform var {vt:.2f} vs bias {vb:.2f} "
          f"({vt/max(vb,1e-9):.1f}× sharper), reaching {100*vt/vo:.0f}% of observed.")


if __name__ == "__main__":
    main()
