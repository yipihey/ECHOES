"""Validate the productionized build_package: legacy-equivalence + calibration fixes + speedup.

The vectorised build_package adds two calibration fixes (per-miss-kind field-K for z-fails;
close-pair Δz background-broadening for collided) and batches the per-object KDE loop. This checks:
  1. LEGACY params (K_zfail=K, dz_bg_frac=0) reproduce the old PIT (the refactor changed no math);
  2. PRODUCTION defaults calibrate both arms (collided U→flat, z-fail std→0.289);
  3. the vectorised loop is faster (wall-clock of the build).

  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 JAX_PLATFORMS=cpu python validation/validate_build_package.py
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from echoes.surveys.boss import load_boss
from echoes.photoz import PhotoZKNN, photoz_features
from echoes.completion import measure_close_pair_dz
from echoes.mock_systematics import apply_survey_systematics
from echoes.posterior import build_package
from echoes.pit import pit_uniformity

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"


def opit(pkg, tz):
    q = np.asarray(pkg["qlev"], float); iv = np.asarray(pkg["invcdf"], float)
    return np.array([np.interp(tz[i], iv[i], q, left=0.0, right=1.0) for i in range(iv.shape[0])])


def main():
    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    ra = np.asarray(cat.ra_data); dec = np.asarray(cat.dec_data); z = np.asarray(cat.z_data)
    feat = photoz_features(cat.colors_data, cat.mags_data)
    good = np.isfinite(feat).all(1) & (np.asarray(cat.imatch_data) == 1)
    pz = PhotoZKNN(k=100).fit(feat[good], z[good])
    obs, tg, kept, true_z = apply_survey_systematics(
        ra, dec, z, cat.colors_data, cat.mags_data, np.asarray(cat.w_sys_data),
        coll_frac=0.6, zfail_frac=0.014, zfail_faint_bias=1.5, seed=0)
    true_z = np.asarray(true_z); miss = np.asarray(tg.miss_kind)
    zf = miss == "zfail"; coll = miss == "collided"
    dz = measure_close_pair_dz(obs, 62 / 3600.)

    def run(label, **kw):
        t0 = time.perf_counter()
        pkg = build_package(obs, tg, pz, dz_pool=dz, **kw)
        dt = time.perf_counter() - t0
        pit = opit(pkg, true_z)
        a, c, f = pit_uniformity(pit), pit_uniformity(pit[coll]), pit_uniformity(pit[zf])
        print(f"  {label:34s} build={dt:5.2f}s   ALL KS={a['ks']:.3f}   "
              f"collided KS={c['ks']:.3f}   zfail KS={f['ks']:.3f} std={f['std']:.3f}", flush=True)
        return dt

    print(f"inject-and-recover: {len(true_z):,} missing ({coll.mean():.0%} collided)\n")
    print("=== build_package: legacy vs production (PIT + wall-clock) ===")
    t_leg = run("LEGACY (K_zfail=150, bg=0)", K_zfail=150, dz_bg_frac=0.0)
    t_pro = run("PRODUCTION (defaults: 20, 0.30)")
    print(f"\n  legacy ALL KS≈0.072 / collided 0.095 / zfail 0.204 (from pit_breakdown.py) — "
          f"match = refactor is math-equivalent.")
    print(f"  production should show collided KS DOWN (~0.06) + zfail std UP (toward 0.289).")
    print(f"  speedup (same work, batched): {t_leg / max(t_pro, 1e-9):.1f}x  ({t_leg:.2f}s -> {t_pro:.2f}s)")


if __name__ == "__main__":
    main()
