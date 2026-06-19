# ECHOES

**Equal-weight Completed Hypothetical Observation Ensembles** — survey-ready
posterior samples of completed galaxy catalogs.

ECHOES turns a spectroscopic survey into an **ensemble of equal-weight,
cosmology-free completed catalogs**: every observed galaxy is kept at its measured
(RA, Dec, z), and every spectroscopically-missing galaxy is restored at its real
imaging position with a redshift drawn from a data-driven posterior. The result
lets you run *any* clustering statistic — two-point, higher-order, marked,
nearest-neighbor, topological, field-level — directly on completed point catalogs
and propagate the observational-completion uncertainty by resampling, instead of
re-deriving the survey weights for each new statistic.

The first release completes **BOSS DR12 CMASS-South**. The structure is built to
add more surveys over time.

- 📄 **Paper:** [`paper/`](paper/) (ECHOES for BOSS DR12 CMASS-South)
- 📊 **Interactive report:** https://yipihey.github.io/ECHOES/report.html
- 📚 **Method walkthrough:** [`docs/method.md`](docs/method.md) — more pedagogical than the paper
- 📦 **Data:** [`DATA.md`](DATA.md) — products, inputs, and how to get them

## Install

```bash
pip install git+https://github.com/yipihey/ECHOES        # core (numpy-only sampling works)
pip install "echoes[clustering] @ git+https://github.com/yipihey/ECHOES"   # + Corrfunc measurements
pip install "echoes[graphgp]   @ git+https://github.com/yipihey/ECHOES"    # + graphGP field engine
```
or, for full reproducibility:
```bash
conda env create -f environment.yml && conda activate echoes && pip install -e .[all]
```

## Quickstart — draw completed catalogs (no large downloads)

```bash
echoes-draw --seed 0 --out catalog_0.fits          # one ~120k-galaxy realization
echoes-draw --seed 0 --n 100 --out-prefix cat_      # a 100-member ensemble
```
```python
from echoes import load_package, draw
pkg = load_package("data_release/cmass_south_posterior.npz")
cat = draw(pkg, seed=0)        # dict(ra, dec, z, prov, N) — equal-weight, cosmology-free
```
A reproducible ensemble is just the set of integer seeds. Pair the catalogs with
`data_release/cmass_south_randoms.npz` and use **equal weights** (no completeness
weights): the completion reproduces the official weighted BOSS clustering to ~1–2%.

Provenance flags (`PROV`): `0` observed-specz · `1` fiber-collided · `2`
redshift-failure · `3` imaging-systematic analog · `4` zhost-fallback.

## Two redshift engines
- **KNN-field (default):** a fast local-density posterior along each sightline.
  Cosmology-free, compresses to the 2 MB released posterior. Used for the release.
- **graphGP (optional):** a conditional anisotropic Gaussian-process posterior over
  the density field (Matheron sampling on a sparse graph), giving correlated
  redshift draws. `pip install echoes[graphgp]`. See [`docs/method.md`](docs/method.md).

## Repository layout
```
echoes/        core package: completion, photo-z, posterior, graphGP field, clustering
  surveys/     per-survey loaders (boss.py) behind a small Survey interface
pipeline/      build the release products and the HTML report
validation/    the truth-recovery / calibration / engine-comparison scripts (paper figures)
data/          fetch scripts for the public input data (see DATA.md)
data_release/  the shipped ECHOES products + standalone sampler
docs/          the interactive report, method walkthrough, tutorials, "adding a survey"
paper/         the manuscript (emulateapj)
```

## Reproduce the paper
With inputs fetched (see [`DATA.md`](DATA.md)):
```bash
python pipeline/build_release.py            # rebuild the posterior + randoms
python validation/truth_recovery.py         # truth-known recovery
python validation/graphgp_vs_knn.py         # engine comparison
python pipeline/build_report.py             # rebuild docs/report.html
```

## Adding a survey
Implement the `echoes.surveys.base.SurveyCatalog` interface (a loader returning
observed galaxies, randoms, a completeness map, and the weight components). See
[`docs/adding_a_survey.md`](docs/adding_a_survey.md).

## Citation
See [`CITATION.cff`](CITATION.cff). Please cite the ECHOES paper and the data DOI.

## License
MIT — see [`LICENSE`](LICENSE).
