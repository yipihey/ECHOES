# Adding a new dataset

ECHOES is structured so a new dataset is a new module under
[`echoes/surveys/`](../echoes/surveys/) plus one `register(DatasetSpec(...))` call —
never a fork of the pipeline. The reusable infrastructure (`echoes/posterior.py`
`build_package`/`draw`, the field engines, the copula, the released `.npz` format)
stays survey-agnostic; everything dataset-specific is captured declaratively in one
[`DatasetSpec`](../echoes/surveys/registry.py). Three datasets ship today:

| dataset | mode | what is completed |
|---|---|---|
| `boss_cmass_south` | `restore_missing` | fiber-collision + redshift-failure galaxies, restored at imaging positions with a colour photo-z × local-density redshift |
| `desi_qso` | `restore_missing` | uneven-coverage (NTILE) incompleteness, restored across the footprint by the completeness deficit with a field-only redshift |
| `quaia` | `resample_redshift` | every object's large-error photo-z resampled from N(z, σ_z) × local field, + selection-function footprint inpaint |

## The two completion modes

- **`restore_missing`** — there are spectroscopically-missing objects to restore.
  The observed catalogue is the fixed backbone; only the restored redshifts vary.
  Build a missing-target list (`CompletionTargets`) and a `DatasetSpec` with
  `completion_mode="restore_missing"`.
- **`resample_redshift`** — every object is present with a fixed position but an
  uncertain redshift (a photo-z). No backbone (`n_obs=0`); every object's redshift is
  resampled. `build_targets=None` → the pipeline synthesises targets with
  [`synth_resample_targets`](../echoes/surveys/targets_generic.py), or supply a builder
  that also adds footprint inpaint (see `quaia.build_quaia_targets`). Set
  `completion_mode="resample_redshift"` and `photoz_gaussian=True`.

## The catalog contract

A loader returns a **catalog object** satisfying
[`echoes.surveys.base.SurveyCatalog`](../echoes/surveys/base.py). Required:

| attribute | meaning |
|---|---|
| `ra_data, dec_data, z_data` | observed objects (deg, deg, redshift; `z_data` is the photo-z point estimate in resample mode) |
| `ra_random, dec_random, z_random` | a survey random catalog (defines the footprint) |
| `sel_map`, `nside` | angular completeness/selection HEALPix map |

Optional (read defensively via `getattr`), by mode:

| attribute | used by |
|---|---|
| `w_sys_data, w_cp_data, w_noz_data` | BOSS restore_missing weight→target list |
| `w_comp_data` | DESI restore_missing completeness deficit |
| `colors_data, mags_data` | colour-space photo-z (BOSS); absent ⇒ field-only redshift |
| `z_data_err` | resample_redshift per-object photo-z 1σ (Quaia `redshift_quaia_err`) |

## Steps

1. **Loader** `echoes/surveys/<name>.py`: read the catalog + randoms + `sel_map`.
   Reuse `echoes.distance` for coordinates and `echoes.surveys.sdss_io` for SDSS FITS.
2. **Targets** (restore_missing only): build the missing list as a `CompletionTargets`
   (`ra, dec, colors[(M,C) or (M,0)], mags|None, miss_kind[PROV keys], host_index`).
   See `boss_targets.py` (real colours) and `desi.py::load_desi_targets` (no colours,
   footprint-deficit placement). resample_redshift uses `synth_resample_targets`.
3. **Register** at the bottom of the module:
   ```python
   from .registry import register, DatasetSpec
   register(DatasetSpec(name="<name>", load_catalog=load_<name>,
                        completion_mode="restore_missing"|"resample_redshift",
                        build_targets=<builder or None>, photoz_features=<fn or None>,
                        photoz_gaussian=<bool>, field_pool="observed"|"anchors",
                        uniform_randoms=<bool>, nq=<int>, product_prefix="<name>"))
   ```
   and add `from .surveys import <name>` to `echoes/datasets.py`.
4. **Provenance**: if the dataset introduces a new origin, add a code to `PROV` in
   `echoes/completion.py` (+ the parallel `PROV_GROUP/COLOR/LABEL/DESCRIPTION` dicts) —
   it then propagates to every visualizer via `prov_registry()`.
5. **Uniform-footprint randoms**: if the survey randoms trace completeness (DESI NTILE,
   Quaia selection function), set `uniform_randoms=True` — `build_release.py` then ships
   `1/completeness`-flattened randoms (a completed catalog has the selection removed and
   must pair with uniform randoms; see `echoes/surveys/randoms.py`).
6. **Build**: `python pipeline/build_release.py --dataset <name>` writes the compact
   posterior + randoms + README into `data_release/<name>/` (BOSS keeps the historical
   flat `data_release/`).
7. **Validate**: `python validation/validate_new_datasets.py` (count, coverage,
   clustering for restore_missing; n(z), resample width, PIT for resample_redshift).

Each new dataset is intended to be its own short paper in the ECHOES series.
