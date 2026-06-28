# desi_qso completed-catalog posterior (release bundle)

Equal-weight, cosmology-free completed catalogs of **desi_qso**. Every realization keeps the 856,831 observed objects (fixed) and restores the 113,370 spectroscopically-missing objects at their positions with a data-driven redshift. The posterior is stored compactly — draw unlimited samples.

## Files
- `desi_qso_posterior.npz`  (22.54 MB) the posterior (shared base once +
  each varying-redshift object's inverse-CDF). 1 file = the whole ensemble. Engine: field.
- `desi_qso_randoms.npz`     (36.73 MB) uniform-footprint randoms (RA, DEC, Z), 3,427,324 points.
- `draw_samples.py`              standalone numpy-only sampler (shared, in ../_shared/ or repo root).

## Quickstart
```python
from echoes.posterior import load_package, draw
pkg = load_package("desi_qso_posterior.npz")
cat = draw(pkg, seed=0)            # dict(ra, dec, z, prov, N)
```
A reproducible ensemble of K catalogs is just K seeds (0..K-1) — the shared base is stored once.
Use **equal weights** (no completeness weights): the completion has absorbed the correction.
This package uses the independent (IID) per-object redshift draw.

## Provenance codes present (PROV, int8)
  0 observed  — observed
  6 coverage  — completed:coverage

See DATA.md and docs/method.md (repository root) for product conventions and the full
provenance registry (echoes.completion.PROV_GROUP / PROV_COLOR).
