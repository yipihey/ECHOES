"""Build a shareable release bundle for any registered dataset into data_release/.

Registry-driven: ``--dataset <name>`` selects a :class:`echoes.surveys.registry.DatasetSpec`
(``boss_cmass_south``, ``desi_qso``, ``quaia``, …) and the same flow produces the compact
posterior package + uniform-footprint randoms + README. BOSS keeps its historical flat
location (``data_release/cmass_south_*.npz``); new datasets get a clean per-dataset
subdirectory (``data_release/<name>/``).

    OMP_NUM_THREADS=16 JAX_PLATFORMS=cpu python pipeline/build_release.py --dataset boss_cmass_south
    OMP_NUM_THREADS=16 JAX_PLATFORMS=cpu python pipeline/build_release.py --dataset desi_qso
    OMP_NUM_THREADS=16 JAX_PLATFORMS=cpu python pipeline/build_release.py --dataset quaia
    # BOSS Tier-A non-Gaussian generative engine (behind the gates):
    OMP_NUM_THREADS=16 JAX_PLATFORMS=cpu python pipeline/build_release.py --dataset boss_cmass_south --engine generative
"""
import argparse, glob, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

import echoes.datasets                          # noqa: F401  (fires dataset registrations)
from echoes.surveys import registry
from echoes import posterior as PS
from echoes.completion import measure_close_pair_dz

N_RAND_MULT = 4          # randoms = 4x N_data (uniform-footprint; users can draw more)

# Deployment paths (filesystem layout is a deployment concern — kept out of the library).
DATA_PATHS = {
    "boss_cmass_south": dict(
        data=["data/boss/galaxy_DR12v5_CMASS_South.fits.gz"],
        random=["data/boss/random0_DR12v5_CMASS_South.fits.gz"],
        targets="data/boss/cmass_targets_South.fits"),
    "desi_qso": dict(
        data=["data/desi/QSO_NGC_clustering.dat.fits", "data/desi/QSO_SGC_clustering.dat.fits"],
        random=sorted(glob.glob("data/desi/QSO_*_clustering.ran.fits"))),
    "quaia": dict(
        data="data/quaia/quaia_G20.0.fits",
        random="data/quaia/random_G20.0_10x.fits",
        selection="data/quaia/selection_function_NSIDE64_G20.0.fits"),
}


def _out_dir(name):
    """BOSS keeps the historical flat layout; new datasets get a per-dataset subdir."""
    return "data_release" if name == "boss_cmass_south" else os.path.join("data_release", name)


def _load_catalog(spec, paths):
    if spec.name == "boss_cmass_south":
        return spec.load_catalog(paths["data"], paths["random"], **spec.load_catalog_kwargs)
    if spec.name == "quaia":
        return spec.load_catalog(paths["data"], paths["random"],
                                 selection_path=paths["selection"], **spec.load_catalog_kwargs)
    if spec.name == "desi_qso":
        return spec.load_catalog(paths["data"], paths["random"], nside=spec.nside,
                                 **spec.load_catalog_kwargs)
    return spec.load_catalog(**spec.load_catalog_kwargs)


def _build_targets(spec, cat, paths):
    if spec.completion_mode == "resample_redshift":
        return spec.build_targets(cat, footprint_inpaint=spec.footprint_inpaint, seed=0)
    if paths.get("targets"):
        return spec.build_targets(cat, path=paths["targets"], seed=0)
    return spec.build_targets(cat, seed=0)


def _fit_photoz(spec, cat):
    """Colour-space photo-z (BOSS only). None when the dataset has no imaging colours."""
    if spec.photoz_features is None or getattr(cat, "colors_data", None) is None:
        return None
    from echoes.photoz import PhotoZKNN
    z = np.asarray(cat.z_data)
    feat = spec.photoz_features(cat.colors_data, cat.mags_data)
    good = np.isfinite(feat).all(1)
    imatch = getattr(cat, "imatch_data", None)
    if imatch is not None:
        good = good & (np.asarray(imatch) == 1)
    return PhotoZKNN(k=100).fit(feat[good], z[good])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=registry.available(),
                    help="registered dataset to build")
    ap.add_argument("--engine", choices=["field", "generative"], default=None,
                    help="'field' (KNN-KDE local density) or 'generative' (Tier-A non-Gaussian; "
                         "BOSS only). Default: the dataset spec's default_engine.")
    ap.add_argument("--transform", default="lognormal", help="generative transform (BOSS generative)")
    ap.add_argument("--cic-R", type=float, default=8.0, help="CiC scale [Mpc/h] for T")
    ap.add_argument("--no-copula", action="store_true", help="ship the legacy IID sampler")
    ap.add_argument("--copula-modes", type=int, default=128, help="low-rank copula mode count")
    args = ap.parse_args()

    spec = registry.get(args.dataset)
    paths = DATA_PATHS[args.dataset]
    engine = args.engine or spec.default_engine
    copula = spec.copula and not args.no_copula
    out = _out_dir(spec.name)
    os.makedirs(out, exist_ok=True)

    print(f"[build_release] dataset={spec.name} mode={spec.completion_mode} "
          f"engine={engine} copula={copula} -> {out}/")
    cat = _load_catalog(spec, paths)
    tg = _build_targets(spec, cat, paths)
    pz = _fit_photoz(spec, cat)

    # ---- posterior package ----
    if engine == "generative":
        from echoes.generative import build_generative_model
        dz = measure_close_pair_dz(cat, 62 / 3600.)
        gm = build_generative_model(cat, transform=args.transform, cic_R=args.cic_R, verbose=True)
        pkg = PS.build_package_generative(cat, tg, pz, gm, features_fn=spec.photoz_features,
                                          nq=spec.nq, dz_pool=dz, verbose=True, copula=copula,
                                          copula_modes=args.copula_modes)
    else:
        fctx = None
        if copula:
            from echoes.fieldpost import build_field_context
            fctx = build_field_context(cat, sel_map=cat.sel_map, nside=cat.nside, verbose=True)
        pkg = PS.build_package(cat, tg, pz, mode=spec.completion_mode, nq=spec.nq,
                               features_fn=spec.photoz_features, anchor_frac=spec.anchor_frac,
                               verbose=True, copula=copula, field_ctx=fctx,
                               copula_modes=args.copula_modes)

    ppath = os.path.join(out, f"{spec.prefix}_posterior.npz")
    PS.write_package(pkg, ppath)
    psz = os.path.getsize(ppath)

    # ---- uniform-footprint randoms ----
    # A completed catalog has the angular selection removed, so it must pair with
    # UNIFORM-footprint randoms. For datasets whose survey randoms trace completeness
    # (DESI NTILE, Quaia selection function) generate uniform randoms from sel_map;
    # for BOSS (COMP~0.99) the survey randoms are already uniform, so subsample them.
    rng = np.random.default_rng(0)
    n_data = len(np.asarray(cat.ra_data))
    nr = N_RAND_MULT * n_data
    if spec.uniform_randoms:
        from echoes.surveys.randoms import uniform_footprint_randoms
        ra_r, dec_r, z_r = uniform_footprint_randoms(cat, nr, rng=rng)
    else:
        nr = min(nr, len(np.asarray(cat.ra_random)))
        idx = rng.choice(len(np.asarray(cat.ra_random)), nr, replace=False)
        ra_r = np.asarray(cat.ra_random)[idx]
        dec_r = np.asarray(cat.dec_random)[idx]
        z_r = np.asarray(cat.z_random)[idx]
    rpath = os.path.join(out, f"{spec.prefix}_randoms.npz")
    np.savez_compressed(rpath, ra=ra_r.astype(np.float32), dec=dec_r.astype(np.float32),
                        z=z_r.astype(np.float32))
    rsz = os.path.getsize(rpath)

    _write_readme(spec, pkg, out, psz, rsz, nr, copula, engine)

    print(f"\nwrote release bundle to {out}/:")
    print(f"  {spec.prefix}_posterior.npz  {psz/1e6:6.2f} MB  "
          f"(n_obs={pkg['n_obs']:,} n_miss={pkg['n_miss']:,} mode={pkg.get('package_mode')})")
    print(f"  {spec.prefix}_randoms.npz    {rsz/1e6:6.2f} MB  ({nr:,} randoms)")
    print(f"  README.md")


def _write_readme(spec, pkg, out, psz, rsz, nr, copula, engine):
    from echoes.completion import prov_registry
    mode = pkg.get("package_mode", "restore_missing")
    if mode == "resample_redshift":
        what = (f"Equal-weight completed catalogs of **{spec.name}**. Every realization keeps "
                f"all {pkg['n_miss']:,} object positions FIXED and **resamples each redshift** "
                f"from its photo-z posterior sharpened by the local density field "
                f"(plus selection-function footprint inpaint). The positions are data; only the "
                f"uncertain redshifts vary between realizations.")
    else:
        what = (f"Equal-weight, cosmology-free completed catalogs of **{spec.name}**. Every "
                f"realization keeps the {pkg['n_obs']:,} observed objects (fixed) and restores the "
                f"{pkg['n_miss']:,} spectroscopically-missing objects at their positions with a "
                f"data-driven redshift. The posterior is stored compactly — draw unlimited samples.")
    codes = "\n".join(f"  {c} {info['short_label']:9s} — {info['group']}"
                      for c, info in sorted(prov_registry().items())
                      if c in np.unique(pkg["base_prov"]))
    readme = f"""# {spec.name} completed-catalog posterior (release bundle)

{what}

## Files
- `{spec.prefix}_posterior.npz`  ({psz/1e6:.2f} MB) the posterior (shared base once +
  each varying-redshift object's inverse-CDF). 1 file = the whole ensemble. Engine: {engine}.
- `{spec.prefix}_randoms.npz`     ({rsz/1e6:.2f} MB) uniform-footprint randoms (RA, DEC, Z), {nr:,} points.
- `draw_samples.py`              standalone numpy-only sampler (shared, in ../_shared/ or repo root).

## Quickstart
```python
from echoes.posterior import load_package, draw
pkg = load_package("{spec.prefix}_posterior.npz")
cat = draw(pkg, seed=0)            # dict(ra, dec, z, prov, N)
```
A reproducible ensemble of K catalogs is just K seeds (0..K-1) — the shared base is stored once.
Use **equal weights** (no completeness weights): the completion has absorbed the correction.
{"This package carries a field-correlation **copula** (coherent cross-object redshift dependence; per-object marginals unchanged). `draw(pkg, seed, copula=False)` recovers the independent draw." if copula else "This package uses the independent (IID) per-object redshift draw."}

## Provenance codes present (PROV, int8)
{codes}

See DATA.md and docs/method.md (repository root) for product conventions and the full
provenance registry (echoes.completion.PROV_GROUP / PROV_COLOR).
"""
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(readme)


if __name__ == "__main__":
    main()
