# ECHOES data products — BOSS DR12 CMASS-South

| file | contents |
|---|---|
| `cmass_south_posterior.npz` | the completion posterior: the fixed 109,636 observed galaxies + an inverse-CDF redshift posterior for each of the 6,777 missing targets |
| `cmass_south_randoms.npz` | uniform-footprint randoms (RA, DEC, Z), 438,544 points |
| `draw_samples.py` | standalone NumPy-only sampler (no `echoes` install needed) |
| `SHA256SUMS` | integrity manifest |

```bash
python draw_samples.py --seed 0 --out catalog_0.npz
```
```python
from draw_samples import load_package, draw
pkg = load_package("cmass_south_posterior.npz")
cat = draw(pkg, seed=0)        # ra, dec, z, prov, N
```
Columns: `RA`, `DEC` [deg]; `Z` (redshift); `PROV` (0 observed, 1 collided,
2 zfail, 3 systot-analog, 4 zhost). Equal-weight and cosmology-free.

Zenodo DOI status: pending. Until a DOI is minted, cite the repository commit
and verify products with `SHA256SUMS`.
