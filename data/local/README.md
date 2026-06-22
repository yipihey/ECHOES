# data/local/ — local-neighborhood reconstructions (gitignored, fetched on demand)

External data for the true-3D ECHOES line (branch `data/local-neighborhood`). The data
files here are **gitignored** — fetch them with the scripts in `data/`. See
`docs/local_neighborhood.md` for the science plan and full references.

## Cosmicflows-4 — `data/local/cf4/` (`python data/fetch_cf4.py`)
- `CF4_new_64-z008_delta.fits` — over-density δ, 64³ on a 1000 Mpc/h box (~15.6 Mpc/h/voxel),
  supergalactic axis order **(SGZ, SGY, SGX)**.
- `CF4_new_64-z008_velocity.fits` — 3D peculiar velocity, shape **(3, 64, 64, 64)**.
  **Multiply by 52 to get km/s** (`fetch_cf4.VELOCITY_SCALE`). `*_error.fits` are per-voxel std.
- `cf4_table2.fits` — 55,877 individual galaxy distances (PGC, V_cmb, distance moduli per
  method: TF/FP/SBF/SNIa/SNII/TRGB/Cepheid/maser, RA/Dec). `cf4_groups.fits` — 38,053 groups
  with peculiar velocities.
- Source: IP2I Lyon portal + VizieR `J/ApJ/944/94`. Cite Tully+ 2023 (catalog) & Courtois+ 2023
  (fields). Gotcha: EDD (edd.ifa.hawaii.edu) has a broken TLS chain — use http or skip-verify.

## Manticore — `data/local/manticore/` (planned: `data/fetch_manticore.py`)
- 80-member posterior ensemble of δ + 3D velocity (256³ @ 3.9 Mpc, 1000 Mpc box) + halo/cluster
  catalogs, via the `manticore_data` Python package (cosmictwin.org / digitaltwin.fysik.su.se).
  `pip install git+https://git.aquila-consortium.org/Aquila-Consortium/manticore_data/`. Cluster
  catalog also on VizieR `J/MNRAS/540/716`. Cite McAlpine+ 2025 (MNRAS 540, 716).

## BORG — `data/local/borg/` (optional)
- BORG-SDSS DR7 derived products on Zenodo `10.5281/zenodo.1455729` (GPL). Full 2M++ BORG-PM
  chain is largely by-request from the Aquila Consortium. Cite Jasche & Wandelt 2013 + the
  specific reconstruction paper.
