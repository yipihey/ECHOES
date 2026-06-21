"""Build the shareable release bundle into data_release/ (small enough for GitHub +
served by GitHub Pages; the report HTML links to these files for download).

Writes:
  data_release/cmass_south_posterior.npz   the compact posterior (observed base once +
                                        per-missing inverse-CDF); draw unlimited samples
  data_release/cmass_south_randoms.npz     uniform-footprint randoms (downsampled survey
                                        randoms; CMASS COMP~0.99 so they are uniform to ~1%)
  data_release/draw_samples.py             standalone numpy-only sampler (shipped separately)
  data_release/README.md                   quickstart

    OMP_NUM_THREADS=16 JAX_PLATFORMS=cpu python pipeline/build_release.py
    # Tier-A non-Gaussian generative engine (behind the gates):
    OMP_NUM_THREADS=16 JAX_PLATFORMS=cpu python pipeline/build_release.py --engine generative
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from echoes.surveys.boss import load_boss
from echoes.photoz import PhotoZKNN, photoz_features
from echoes.surveys.boss_targets import load_cmass_targets
from echoes.completion import measure_close_pair_dz
from echoes import posterior as PS

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"
TARGETS = "data/boss/cmass_targets_South.fits"
OUT = "data_release"
N_RAND_MULT = 4          # randoms = 4x N_data (uniform-footprint; users can draw more)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["field", "generative"], default="field",
                    help="'field' (default, KNN-KDE local density) or 'generative' "
                         "(Tier-A data-driven non-Gaussian field; behind the gates)")
    ap.add_argument("--transform", default="lognormal",
                    help="generative transform: lognormal (shot-noise-deconvolved, the "
                         "calibration-friendly default) | empirical (max raw kNN, over-skews) | identity")
    ap.add_argument("--cic-R", type=float, default=8.0, help="CiC scale [Mpc/h] for T")
    ap.add_argument("--no-copula", action="store_true",
                    help="ship the legacy IID sampler (no field-correlation copula modes)")
    ap.add_argument("--copula-modes", type=int, default=128, help="low-rank copula mode count")
    args = ap.parse_args()
    copula = not args.no_copula

    os.makedirs(OUT, exist_ok=True)
    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    z = np.asarray(cat.z_data); feat = photoz_features(cat.colors_data, cat.mags_data)
    good = np.isfinite(feat).all(1) & (cat.imatch_data == 1)
    pz = PhotoZKNN(k=100).fit(feat[good], z[good]); dz = measure_close_pair_dz(cat, 62/3600.)
    tg = load_cmass_targets(cat, path=TARGETS, seed=0)

    # ---- posterior package ----
    # The field-correlation copula adds the coherent cross-object dependence the IID
    # sampler under-disperses (recovers the ~19% large-scale completion-variance deficit;
    # validation/copula_covariance_check.py), with per-object marginals/PIT unchanged.
    if args.engine == "generative":
        from echoes.generative import build_generative_model
        gm = build_generative_model(cat, transform=args.transform, cic_R=args.cic_R, verbose=True)
        pkg = PS.build_package_generative(cat, tg, pz, gm, dz_pool=dz, verbose=True,
                                          copula=copula, copula_modes=args.copula_modes)
    else:
        fctx = None
        if copula:
            from echoes.fieldpost import build_field_context
            fctx = build_field_context(cat, sel_map=cat.sel_map, nside=cat.nside, verbose=True)
        pkg = PS.build_package(cat, tg, pz, dz_pool=dz, verbose=True,
                               copula=copula, field_ctx=fctx, copula_modes=args.copula_modes)
    ppath = os.path.join(OUT, "cmass_south_posterior.npz")
    PS.write_package(pkg, ppath)
    psz = os.path.getsize(ppath)

    # ---- uniform-footprint randoms (downsampled survey randoms; COMP~0.99 => uniform) ----
    rng = np.random.default_rng(0)
    nr = min(N_RAND_MULT * cat.N_data, len(cat.ra_random))
    idx = rng.choice(len(cat.ra_random), nr, replace=False)
    rpath = os.path.join(OUT, "cmass_south_randoms.npz")
    np.savez_compressed(rpath,
                        ra=np.asarray(cat.ra_random)[idx].astype(np.float32),
                        dec=np.asarray(cat.dec_random)[idx].astype(np.float32),
                        z=np.asarray(cat.z_random)[idx].astype(np.float32))
    rsz = os.path.getsize(rpath)

    # ---- README ----
    readme = f"""# BOSS CMASS-South completed-catalog posterior (release bundle)

Equal-weight, cosmology-free completed catalogs of BOSS DR12 CMASS-South. Every
realization keeps the {pkg['n_obs']:,} observed galaxies (fixed) and adds the
{pkg['n_miss']:,} spectroscopically-missing galaxies (fiber collisions + redshift
failures) at their real imaging positions with a GP/local-density redshift. The
posterior is stored compactly so you draw as many samples as you like locally.

## Files
- `cmass_south_posterior.npz`  ({psz/1e6:.2f} MB) the posterior (observed base once +
  each missing galaxy's redshift inverse-CDF). 1 file = the whole ensemble.
- `cmass_south_randoms.npz`     ({rsz/1e6:.2f} MB) uniform-footprint randoms (RA, DEC, Z),
  {nr:,} points. CMASS-South is ~99% complete (COMP~0.99) so these are uniform to ~1%.
- `draw_samples.py`             standalone numpy-only sampler.

## Quickstart
```bash
pip install numpy
python draw_samples.py --seed 0 --out catalog_0.npz         # one realization
python draw_samples.py --seed 0 --n 100 --out-prefix cat_   # 100 realizations
```
```python
from echoes.posterior import load_package, draw
pkg = load_package("cmass_south_posterior.npz")
cat = draw(pkg, seed=0)            # dict(ra, dec, z, prov, N); ~120k galaxies
```
A fixed, reproducible ensemble of K catalogs is just K seeds (0..K-1) — no need to
store K copies (the observed galaxies are shared). Pair the catalogs with
`cmass_south_randoms.npz` and use equal weights (no completeness weights needed):
the completion reproduces the official w_c-weighted clustering to ~1-2%.

{"This package carries a field-correlation **copula**: the missing redshifts are drawn"
 " with the coherent cross-object dependence of the measured xi(r), not independently,"
 " so the large-scale completion covariance is honest (the per-object marginals are"
 " identical to the independent draw). `draw(pkg, seed, copula=False)` recovers the"
 " independent draw." if copula else
 "This package uses the independent (IID) missing-redshift draw."}

## Columns
RA, DEC [deg]; Z (redshift); PROV (per-object provenance, int8):
  0 observed   — real spec-z, the fixed base catalogue;
  1 collided   — COMPLETED: galaxy lost to a fiber collision, restored at its
                 imaging position with a close-pair-anchored redshift;
  2 zfail      — COMPLETED: galaxy whose spectrum yielded no redshift, restored
                 from imaging with a local-density / photo-z redshift;
  4 zhost      — COMPLETED (fiber-collision), redshift fell back to the host;
  3 systot     — INPAINTED: a synthetic point added to undo an imaging-systematic
                 density deficit. NOT a real missing galaxy (no imaging counterpart
                 of its own); drop these for any imaging-position-level analysis.
Roll-up groups (observed / completed:fiber-collision / completed:redshift-failure
/ inpainted) and matching display colours are in echoes.completion.PROV_GROUP /
PROV_COLOR; tools/viz_provenance.py renders the catalog coloured by them.

See DATA.md and docs/method.md (repository root) for product conventions, scope,
and the systematics budget.
"""
    with open(os.path.join(OUT, "README.md"), "w") as f:
        f.write(readme)

    total = psz + rsz + os.path.getsize(os.path.join(OUT, "draw_samples.py"))
    print(f"\nwrote release bundle to {OUT}/:")
    print(f"  cmass_south_posterior.npz  {psz/1e6:6.2f} MB")
    print(f"  cmass_south_randoms.npz    {rsz/1e6:6.2f} MB  ({nr:,} randoms)")
    print(f"  draw_samples.py + README.md")
    print(f"  total ~ {total/1e6:.2f} MB  (fits in the repo + GitHub Pages; <100 MB/file, <1 GB/repo)")


if __name__ == "__main__":
    main()
