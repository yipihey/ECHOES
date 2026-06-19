# Adding a new survey

ECHOES is structured so a new survey is a new module under
[`echoes/surveys/`](../echoes/surveys/), not a fork of the pipeline. BOSS DR12
CMASS-South (`echoes/surveys/boss.py`) is the reference implementation.

## The contract

A survey loader returns a **catalog object** that satisfies
`echoes.surveys.base.SurveyCatalog`. At minimum it exposes:

| attribute | meaning |
|---|---|
| `ra_data, dec_data, z_data` | observed spectroscopic galaxies (deg, deg, redshift) |
| `ra_random, dec_random, z_random` | a matching survey random catalog |
| `sel_map`, `nside` | angular completeness/selection HEALPix map |
| `w_sys_data, w_cp_data, w_noz_data` | completeness-weight components (imaging, close-pair, redshift-failure) used to build the missing-target list |
| `colors_data, mags_data` | photometric features for the color-space photo-z |

A second function returns the **missing-target list** (the photometric detections
that were targeted but have no good spectroscopic redshift), tying them to the
weighted survivors — see `echoes/surveys/boss_targets.py`.

## Steps

1. **Loader** `echoes/surveys/<survey>.py`: read the survey's LSS catalog + randoms
   into the attributes above. Reuse `echoes.surveys.sdss_io` for SDSS-format FITS
   and the angular-completeness map; reuse `echoes.distance` for coordinates.
2. **Targets** `echoes/surveys/<survey>_targets.py`: build the missing list from the
   imaging target catalog and the survey weights (the BOSS version greedily ties
   no-redshift targets to `WEIGHT_CP`/`WEIGHT_NOZ` hosts within the relevant scale).
3. **Register** the survey in `echoes/surveys/__init__.py`.
4. **Parameters**: the completion takes the survey-specific scales as arguments
   (collision scale, redshift range, footprint cuts), so nothing in the core is
   hard-coded to BOSS.
5. **Validate**: run the `validation/` battery (truth-recovery, calibration,
   consistency) for the new survey, exactly as for BOSS.
6. **Release**: `pipeline/build_release.py` produces the compact posterior + randoms;
   `pipeline/build_report.py` produces the report; add the data to Zenodo and
   `DATA.md`.

Each new survey is intended to be its own short paper in the ECHOES series.
