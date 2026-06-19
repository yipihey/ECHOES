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
| `cmass_south_posterior.npz` | 2.0 MB | the full completion posterior — the fixed observed catalog plus an inverse-CDF redshift posterior for each missing target |
| `cmass_south_randoms.npz` | 4.6 MB | uniform-footprint random catalog (RA, DEC, Z) |
| `draw_samples.py` | — | standalone NumPy-only sampler |

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
install it downloads the same 2 MB file once into `~/.cache/echoes` (or
`$ECHOES_DATA`) and verifies the SHA256 hash. FITS output is available with
`pip install "echoes[fits] @ git+https://github.com/yipihey/ECHOES.git"` and
`--out catalog.fits`.

**Zenodo archive status:** the data products are staged for Zenodo, but the
public DOI is not minted in this checkout. Until the DOI is minted, cite the
repository commit and verify any product copy against the SHA256 manifest below.
After publishing, run `python tools/set_doi.py 10.5281/zenodo.NNNNNNN` to
propagate the DOI into the public files.

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
1,505 redshift-failure + 3,510 imaging-systematic analogs = **119,923** galaxies.

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
