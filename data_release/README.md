# BOSS CMASS-South completed-catalog posterior (release bundle)

Equal-weight, cosmology-free completed catalogs of BOSS DR12 CMASS-South. Every
realization keeps the 109,636 observed galaxies (fixed) and adds the
6,777 spectroscopically-missing galaxies (fiber collisions + redshift
failures) at their real imaging positions with a GP/local-density redshift. The
posterior is stored compactly so you draw as many samples as you like locally.

## Files
- `cmass_south_posterior.npz`  (3.14 MB) the posterior (observed base once +
  each missing galaxy's redshift inverse-CDF). 1 file = the whole ensemble.
- `cmass_south_randoms.npz`     (4.63 MB) uniform-footprint randoms (RA, DEC, Z),
  438,544 points. CMASS-South is ~99% complete (COMP~0.99) so these are uniform to ~1%.
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

This package carries a field-correlation **copula**: the missing redshifts are drawn with the coherent cross-object dependence of the measured xi(r), not independently, so the large-scale completion covariance is honest (the per-object marginals are identical to the independent draw). `draw(pkg, seed, copula=False)` recovers the independent draw.

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
