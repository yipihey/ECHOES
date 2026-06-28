# quaia completed-catalog posterior (release bundle)

Equal-weight completed catalogs of **quaia**. Every realization keeps all 920,895 object positions FIXED and **resamples each redshift** from its photo-z posterior sharpened by the local density field (plus selection-function footprint inpaint). The positions are data; only the uncertain redshifts vary between realizations.

## Files
- `quaia_posterior.npz`  (57.67 MB) the posterior (shared base once +
  each varying-redshift object's inverse-CDF). 1 file = the whole ensemble. Engine: field.
- `quaia_randoms.npz`     (24.94 MB) uniform-footprint randoms (RA, DEC, Z), 2,979,336 points.
- `draw_samples.py`              standalone numpy-only sampler (shared, in ../_shared/ or repo root).

## Quickstart
```python
from echoes.posterior import load_package, draw
pkg = load_package("quaia_posterior.npz")
cat = draw(pkg, seed=0)            # dict(ra, dec, z, prov, N)
```
A reproducible ensemble of K catalogs is just K seeds (0..K-1) — the shared base is stored once.
Use **equal weights** (no completeness weights): the completion has absorbed the correction.
This package uses the independent (IID) per-object redshift draw.

## Provenance codes present (PROV, int8)
  7 photoz    — resampled:photo-z
  8 selfunc   — inpainted:selection-function

See DATA.md and docs/method.md (repository root) for product conventions and the full
provenance registry (echoes.completion.PROV_GROUP / PROV_COLOR).
