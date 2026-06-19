# Local data layout

Input data are **not** stored in this repository (they are large and public). Fetch
them here with the scripts in this directory; see `../DATA.md` for the full guide.

```
data/boss/
  galaxy_DR12v5_CMASS_South.fits.gz      # observed CMASS-South spectroscopic galaxies  (~50 MB)
  random0_DR12v5_CMASS_South.fits.gz     # survey random catalog                        (~1.2 GB)
  mask_DR12v5_CMASS_South.ply            # mangle footprint mask                         (~18 MB)
  cmass_targets_South.fits               # SDSS imaging CMASS targets                    (~48 MB)
  mangle_uniform_radec.npy               # uniform-footprint randoms (generated locally) (~6 MB)
  mocks/
    Patchy-Mocks-DR12SGC-COMPSAM_V6C_000*.dat   # truth-recovery mocks (~22 MB each)
```

Fetch:
```
python data/fetch_boss.py                # galaxy + random(s) + mask from SDSS SAS
python data/fetch_cmass_targets.py       # CMASS imaging targets via SkyServer SQL
python data/make_mangle_randoms.py       # uniform randoms from the mask (needs pymangle)
python data/fetch_patchy.py --n-mocks 10 # MultiDark-Patchy mocks (validation)
```
