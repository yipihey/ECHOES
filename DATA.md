# Data: products, inputs, and how to get them

ECHOES separates **released products** (small, shipped here / on Zenodo) from the
**public input data** (large, downloaded from the survey archives). Nothing in
this repository exceeds GitHub's size limits; the heavy inputs are fetched on
demand with the scripts in [`data/`](data/).

---

## 1. Released ECHOES products (use these to draw catalogs)

Shipped in [`data_release/`](data_release/) and ready for Zenodo archival:

| file | size | contents |
|---|---|---|
| `cmass_south_posterior.npz` | 3.1 MB | the full completion posterior — the fixed observed catalog, an inverse-CDF redshift posterior for each missing target, and the field-correlation **copula** modes |
| `cmass_south_randoms.npz` | 4.6 MB | uniform-footprint random catalog (RA, DEC, Z) |
| `draw_samples.py` | — | standalone NumPy-only sampler |

The missing redshifts are drawn through a **field-correlation copula** (low-rank modes
of the measured ξ(r) correlation, stored in the package): the draw carries the coherent
cross-object dependence of the density field, so the large-scale **completion covariance**
is honest rather than ~19% under-dispersed, while every per-object marginal — hence the
per-object PIT calibration — is identical to the independent draw (the copula changes only
the joint law). `draw(pkg, seed, copula=False)` recovers the legacy independent draw
bit-for-bit. See `validation/completion_covariance_shape.py` (the deficit) and
`validation/copula_covariance_check.py` (the copula closing it; ×1.21 total variance).

The GitHub Pages viewer at
[`docs/visualizer/`](docs/visualizer/) is generated from the same posterior. Its
browser bundle stores the observed/base catalog once and stores only
realization-specific missing-redshift draws plus imaging-systematic analogs.

Draw completed catalogs with no large downloads:
```bash
pip install "echoes @ git+https://github.com/yipihey/ECHOES.git"
echoes-draw --seed 0 --out catalog_0.npz           # one realization (~120k galaxies)
echoes-draw --seed 0 --n 100 --out-prefix cat_      # a 100-member ensemble
```
or in Python:
```python
from echoes import load_package, draw
pkg = load_package("data_release/cmass_south_posterior.npz")
cat = draw(pkg, seed=0)        # dict(ra, dec, z, prov, N)
```
`echoes-draw` uses the in-repo posterior when run from a clone. From a package
install it downloads the same ~3 MB file once into `~/.cache/echoes` (or
`$ECHOES_DATA`) and verifies the SHA256 hash. FITS output is available with
`pip install "echoes[fits] @ git+https://github.com/yipihey/ECHOES.git"` and
`--out catalog.fits`.

**Zenodo archive status:** the data products are staged for Zenodo, but the
public DOI is not minted in this checkout. Until the DOI is minted, cite the
repository commit and verify any product copy against the SHA256 manifest below.
After publishing, run `python tools/set_doi.py 10.5281/zenodo.NNNNNNN` to
propagate the DOI into the public files.

---

## 1b. The generative (non-Gaussian) field engine — optional, data-driven

The shipped completion above uses the default redshift engine (`z_mode='field'`).
An **additional, opt-in** engine (`z_mode='generative'`, `pipeline/build_release.py
--engine generative`) reproduces the survey's **non-Gaussian cosmic-web structure**
— the skewed 1-point density PDF, empty voids, and the small-scale clustering that
the `kNN`-CDF / counts-in-cells statistics detect — which the default and the
stationary GraphGP field (a maximum-entropy, two-point-only Gaussian field) cannot.

**How (purely data-driven, no simulations).** A measured **monotonic transform**
`1+δ = T(g)` is applied to the calibrated Gaussian field draw `g`. `T` is fit from
the survey's *own* counts-in-cells PDF (shot-noise-deconvolved via factorial
moments), so no external simulation or cosmology enters. Because `T` is monotonic
it is rank-preserving — the calibrated posterior coverage is untouched. The engine
builds on GraphGP/fieldpost (never replaces them) and stays in the JAX/CUDA stack.

**What it achieves (validated on real CMASS-South + Patchy mocks).**
- **+63%** reduction of the `kNN`-CDF gap to the data (shot-noise-clean; a small-
  scale, R≈8 Mpc/h effect) — a real, large non-Gaussian gain over the Gaussian field.
- Generated catalogs are **systematics-clean**: the imaging seeing/depth imprint
  (cross-correlation 5σ in the raw catalog) is removed to <2σ by the
  `WEIGHT_SYSTOT`-weighted field plus optional ISD residual decontamination
  (`echoes.sp_maps`); the transform itself is SP-blind and does not re-leak.
- The data-driven ISD decontamination is competitive with a Rezaie-style NN-weight
  baseline (`validation/sp_weight_baseline.py`).

**Scope and limits (read before use).**
- A local 1-point transform reshapes the density PDF and the induced clustering; it
  **cannot synthesize genuine filament phase coherence** (the connectivity of
  non-linear collapse is a phase correlation, not a 1- or 2-point property). That is
  the ceiling of this engine; a field-level forward fit (the disco-dj lightcone
  pipeline) would be required to exceed it, at the cost of a fixed cosmology.
- For the **per-object redshift** completion the default `field`/`fieldpost`
  posterior is the recommended product; the generative transform's benefit is in
  the **field/inpaint** (empty-region) generation. The shot-noise-deconvolved
  transform preserves object-level redshift PIT; the raw transform mildly over-
  sharpens it.
- **Residual-systematic caveat:** the data-driven decontamination (ISD /
  `WEIGHT_SYSTOT`) flattens only the **known, mapped** SP templates (sky, depth,
  seeing, airmass, extinction). Any unmapped or unknown systematic that does not
  correlate with a provided template is invisible to both the weighting and the
  null-test battery and **survives in the product**. The gates certify cleanliness
  against the tested templates, not the universe of systematics. The GraphGP/`field`
  modes remain the cosmology-free default; the generative field adds a data-measured
  non-Gaussian texture on top.

Gate scripts (all under `validation/`): `transform_probe.py` (kNN-CDF / CiC),
`sp_cross_power.py` (g×SP), `object_pit.py` and `calibration.py --engine`
(calibration), `sp_weight_baseline.py` (ISD vs NN). The generative engine stays
**off by default** until adopted for a release.

### Fully-contiguous product (every interior hole painted in)

For consumers of **hole-sensitive** statistics (topology, kNN, field-level) the
default masked footprint is unusable — those statistics break at interior holes. A
**contiguous** product fills *every* interior veto hole **and the striped partial-
completeness regions** with the data-driven non-Gaussian field so the catalog has only
the survey's **outer boundary, no inner holes and no completeness striping**
(`pipeline/build_contiguous_release.py`; built on a topological footprint,
`echoes.fill_footprint.build_fill_footprint(contiguous=True)`). The fill is
**completeness-proportional** — it adds the deficit `(1 − completeness)` everywhere, so
the veto/badfield striping is brought to uniform survey density, not just the
zero-coverage pixels.

The completeness itself is the **exact BOSS angular selection** — `completeness_mask ×
Π(veto_masks)` evaluated directly from the source mangle maps (`pipeline/boss_selection.py`;
cached shot-noise-free at nside=2048 in `data/boss/boss_selection_2048.npz`, `ud_grade`-d to
the fill resolution by `fill_footprint.load_analytic_completeness`). This makes the product
**independent of the shipped LSS randoms**, whose finite count makes a random-derived
completeness shot-noise-limited (split-half cover correlation 0.89 → 0.49 → 0.06 at nside
256 → 512 → 1024). With the analytic completeness the real striping (≈7.5% below 0.8, *half*
the random estimate) is captured exactly and the fill deficit *converges* with resolution
(≈260 deg²) instead of growing with noise — so the fill is clean at nside 512 and 1024. The
veto mangle masks (~1 GB, SDSS `lss/geometry/`) are not redistributed; the 4.4 MB cache is.

**Why the *outer boundary* still comes from the randoms, not the masks.** The interior
completeness is analytic (above), but the footprint **boundary** is taken from where the
survey randoms exist (`cover_bool = counts > 0` in `fill_footprint.build_fill_footprint`),
and this is deliberate. The final BOSS LSS window applies the **Anderson et al. (2014, §3.5)
/ Reid et al. (2016) sector cuts** on top of `completeness × veto`: (1) `c > 0.7` (already
baked into our mangle mask — min polygon weight 0.778), (2) a **2°-isolation cut** (drop
sectors not surrounded within 2° by spectroscopically-observed sectors), and (3) a **minimum
sector size 0.1273 deg²**. Cuts (2)–(3) need the spectroscopic-tiling bookkeeping (which
sectors were observed + sector adjacency), which is **not in the mangle mask files** — it
lives in the tiling/spAll products — so the exact window is *not analytically reproducible*
from the masks we ship. Empirically, evaluating our `completeness × veto` at the survey-random
footprint pixel centres returns 0 for ~7–9% of them (≈185–265 deg²): ~207 deg² from our vetoes
(mostly the badfield-seeing/extinction striping mask) flagging sub-pixel regions where randoms
still survive, plus ~58 deg² of completeness-mask-edge pixelization. The robust split is
therefore: **boundary from the randoms** (a *binary* in/out, **not** shot-noise-limited —
median ≈228 randoms/pixel at nside 256), **interior completeness from the analytic mask**
(shot-noise-free). Full random-independence for the *clustering window* is thus not attainable
without the tiling bookkeeping; the survey randoms remain the gold-standard window, while the
analytic masks remain the right tool for the interior. (See `data/make_mangle_randoms.py`.)

- `data_release/contiguous/inpaint_seed_*.npz` — the per-seed inpaint galaxies
  (PROV=5; ~23,900 each, filling 1010 deg² of holes + partial-completeness stripes to
  uniform density at nside=512, with cosmic-web texture). The only seed-varying part.
- `data_release/cmass_south_randoms_contiguous.npz` — uniform randoms over the
  **filled** footprint. **The contiguous catalog must be paired with these randoms**
  (not the masked ones) so data and randoms treat the filled regions identically.

The contiguous catalog for a seed = `draw(cmass_south_posterior.npz, seed)` + that
seed's `inpaint_seed_*.npz`. In the **visualizer**, switch the method dropdown to
**"Contiguous (no inner holes)"** (the default view) to see the gap-free interior;
inpaint galaxies render in teal (PROV=5).

**Tradeoff (honest):** masking a small hole and masking the randoms there cancels it
*exactly* — optimal for 2-point clustering. Filling it is necessary for hole-sensitive
statistics but adds a small 2-point penalty (bounded; controlled by the matching
filled randoms). The masked completion product remains available for pure 2-point
work. Prior-dominated fills (deep holes where the field reverts to the mean) carry a
high `uncert`/`IS_PRIOR_FILL` flag — down-weight `uncert ≥ 0.5` for conservative use.

---

## 2. Public input data (large; download from the archives)

These are the published survey products the pipeline consumes. They are **not**
redistributed here. All paths default to `data/boss/`.

### BOSS DR12 large-scale-structure catalogs — SDSS Science Archive
Base: `https://data.sdss.org/sas/dr12/boss/lss/`

| file | size | what |
|---|---|---|
| `galaxy_DR12v5_CMASS_South.fits.gz` | ~50 MB | observed CMASS-South spectroscopic galaxies + completeness weights |
| `random0_DR12v5_CMASS_South.fits.gz` | ~1.2 GB | survey random catalog (one of 18 realizations) |
| `mask_DR12v5_CMASS_South.ply` | ~18 MB | mangle footprint/completeness mask |

```bash
python data/fetch_boss.py          # galaxy + random0 + mask
```

### CMASS imaging targets — SDSS SkyServer SQL
The spectroscopically-missing galaxies are real DR8 imaging detections. We query
the CMASS colour-selected targets (joined to `SpecObjAll`) from
`https://skyserver.sdss.org/dr12/SkyServerWS/SearchTools/SqlSearch`:
```bash
python data/fetch_cmass_targets.py   # -> data/boss/cmass_targets_South.fits (~48 MB)
```

### Uniform-footprint randoms from the mangle mask (generated locally)
```bash
pip install "echoes[mask] @ git+https://github.com/yipihey/ECHOES.git"
python data/make_mangle_randoms.py   # -> data/boss/mangle_uniform_radec.npy (~6 MB)
```

### MultiDark-Patchy DR12 SGC mocks — truth-recovery validation
Base: `https://data.sdss.org/sas/dr12/boss/lss/dr12_multidark_patchy_mocks/`
```bash
python data/fetch_patchy.py --n-mocks 10            # mock galaxy catalogs
python data/fetch_patchy.py --n-mocks 10 --randoms  # + matching x10 randoms (~187 MB)
```

---

## 3. Reproducing the released products
With the inputs in `data/boss/`:
```bash
python pipeline/build_release.py     # rebuilds cmass_south_posterior.npz + cmass_south_randoms.npz
```
The seed-0 census is deterministic: 109,636 observed + 5,272 fiber-collided +
1,505 redshift-failure + 3,472 imaging-systematic analogs = **119,885** galaxies.

Build the static WebGPU visualizer bundle from the released posterior:
```bash
python pipeline/build_viewer_bundle.py --seeds 0 1 2 3
```
This writes `docs/visualizer/data/viewer_manifest.json` and typed-array chunks
under `docs/visualizer/data/`. Enriched builds can add raw BOSS columns,
computed weights, or method diagnostics through the same manifest:
```bash
python pipeline/build_viewer_bundle.py --enriched-npz enriched_columns.npz
```
The enriched NPZ should contain numeric one-dimensional arrays aligned either to
the observed catalog (`n_obs`) or to the fixed base catalog (`n_base`).

## 4. Integrity (SHA256)
Verify any download against `data_release/SHA256SUMS`:
```bash
cd data_release && sha256sum -c SHA256SUMS
```

---

## 5. Minting the Zenodo DOI (maintainers)

The data products are deposited as a Zenodo **dataset**; the repository itself is
archived as **software** on each GitHub Release via the GitHub–Zenodo integration
(metadata in [`.zenodo.json`](.zenodo.json)).

To deposit the data products and reserve the DOI:
```bash
export ZENODO_TOKEN=...                       # token with deposit:write scope
python pipeline/deposit_zenodo.py --sandbox   # dry run on sandbox.zenodo.org first
python pipeline/deposit_zenodo.py             # real deposit (leaves a DRAFT to review)
```
This uploads `data_release/*` with the metadata in
[`data_release/zenodo_metadata.json`](data_release/zenodo_metadata.json) and prints
the reserved DOI and a review URL. Review and click **Publish** in the Zenodo UI,
then propagate the DOI everywhere:
```bash
python tools/set_doi.py 10.5281/zenodo.NNNNNNN   # updates DATA.md, README, CITATION.cff, data_release/README.md
```
