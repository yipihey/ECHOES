# ECHOES — Local Neighborhood (true-3D) data line

**Branch:** `data/local-neighborhood`. **Status:** scaffolding (CF4 bootstrapped; design below).

A second ECHOES data line that leaves redshift space behind. For the local universe
(out to a few hundred Mpc) we have **real distances** (peculiar-velocity–corrected) and
**Bayesian reconstructions of the actual 3D density + velocity field**. ECHOES can therefore
deliver *true-3D* completed catalogs — galaxies at their real comoving positions, the
survey's holes (most importantly the **Zone of Avoidance** behind the Milky Way) filled in
by conditioning on a real reconstructed field — and an ensemble that propagates the
reconstruction's own uncertainty.

## Why this maps onto ECHOES so well

ECHOES (BOSS) ships an **ensemble of completed catalogs**: fixed observed galaxies + a
posterior draw of the unobserved part, repeated over seeds. The local-universe
reconstructions are *already* posterior ensembles — of the **field** rather than the
catalog:

- **BORG / Manticore** infer the initial conditions + evolved density/velocity field of our
  neighborhood and emit **~50–80 posterior realizations** (constrained N-body twins). They
  are "ECHOES of the field."
- **ECHOES** adds the **galaxy-completion layer in true 3D**: place observed galaxies at
  real comoving positions, then complete the unobserved volume (ZoA, flux limits, masks)
  conditioned on the field.

The two ensembles **compose**: each field realization → one true-3D completed-catalog
realization. The released product is an ensemble of completed local catalogs whose spread
carries *both* the completion uncertainty (as in BOSS ECHOES) *and* the reconstruction's
posterior uncertainty.

## The three data pillars

| Project | What it gives | Volume / resolution | Public access | Key refs |
|---|---|---|---|---|
| **Cosmicflows-4 (CF4)** | 55,877 galaxy **distances + peculiar velocities** (→ real comoving positions); reconstructed **δ + 3D velocity** field cubes | catalog to z≈0.1 (FP subsample ~300 Mpc/h; most ≲160 Mpc/h); public cubes 64³ on 1000 Mpc/h box (~15.6 Mpc/h/voxel) | IP2I Lyon portal (cubes, FITS); VizieR `J/ApJ/944/94` + EDD (catalog) | Tully+ 2023 (ApJ 944, 94; arXiv:2209.11238); Courtois+ 2023 (A&A 670, L15; arXiv:2211.16390); Hoffman+ 2024 WF+CR (MNRAS 527, 3788; arXiv:2311.01340) |
| **BORG** | Bayesian **posterior ensemble** of ICs + evolved density/velocity from 2M++ / SDSS / BOSS | BORG-PM 2M++ ~677 Mpc/h, 256³ (~2.65 Mpc/h); BORG-SDSS DR7 ~750 Mpc/h; BORG-BOSS 4 Gpc/h | Aquila Consortium code (CeCILL/GPL); BORG-SDSS derived products on Zenodo `10.5281/zenodo.1455729`; 2M++ chain largely on request | Jasche & Wandelt 2013 (arXiv:1203.3639); Jasche & Lavaux 2019 (A&A 625, A64; arXiv:1806.11117); Lavaux+ 2019 (arXiv:1909.06396) |
| **Manticore** (recommended field source) | State-of-the-art BORG application: **80 posterior** constrained N-body twins from 2M++; gridded δ+v cubes, halo/cluster/void catalogs, z=0 snapshots | 1000 Mpc box, **256³ @ 3.9 Mpc** inference, 1024³ particles; strongest within R≲200 Mpc | Open, no registration: cosmictwin.org / digitaltwin.fysik.su.se; `manticore_data` Python package (S3); VizieR `J/MNRAS/540/716` (clusters) | McAlpine+ 2025 Manticore I (MNRAS 540, 716; arXiv:2505.10682); Manticore-Deep II (arXiv:2606.10020, in prep) |

**How they relate.** Two lineages over the **same ~150–300 Mpc/h volume**, different inputs:
the **velocity/CF4** lineage (CF4 → CLUES/Hoffman Wiener-filter constrained sims) gives real
distances + a velocity field; the **density/2M++** lineage (BORG → CSiBORG → **Manticore**)
gives a full posterior ensemble of the density field + ICs. Manticore uses **2M++ only** and
treats CF4 as *independent validation* — no published inference jointly fuses them, so an
ECHOES product that **places CF4-distance galaxies in the Manticore field ensemble** is novel
and well-motivated. (Gotchas captured in `data/local/README.md`: IP2I velocity cubes are
stored ×1/52 — multiply by 52 for km/s; supergalactic axis order SGZ,SGY,SGX; EDD's TLS cert
chain is broken — use http or skip-verify.)

## The first products ("images and catalogs")

- **Catalogs** — true-3D completed local galaxy catalogs: observed galaxies at their real
  comoving positions (CF4 distances / Manticore-implied distances), plus completed galaxies
  in the unobserved volume, as a seed-indexed ensemble. PROV codes carry over (observed /
  completed / inpaint); the **Zone of Avoidance** (Galactic plane, ~10–20% of the sky) is the
  flagship inpaint region — the local analogue of BOSS veto holes, but filled in *true 3D*
  from a *real* reconstructed field rather than a measured ξ(r).
- **Images** — the reconstructed δ and 3D velocity field cubes themselves (CF4 now; the
  Manticore 80-member ensemble next), as the conditioning field and as a directly shippable
  volumetric product for the viewer (slices / volume render).

## Reuse of existing ECHOES infrastructure

The repo is already 3D-aware, so most pieces are reused, not rebuilt:
- `echoes/surveys/base.py` `SurveyCatalog` protocol + the `BOSSCatalog` dataclass already
  carry `xyz_data` comoving positions and a `fid_cosmo`; a `LocalCatalog` populates `xyz_data`
  **directly from real distances** (skip the redshift→cosmology step).
- `echoes/distance.py` (`radec_z_to_cartesian` / `cartesian_to_radec_z`) and
  `echoes/clustering.py::comoving_mpc_h` for coordinate transforms.
- `echoes/fieldpost.py` `FieldContext.x_obs` already conditions on comoving positions; the
  local line swaps the *measured ξ(r) kernel* for a **gridded reconstructed field** (CF4 /
  Manticore) as the conditioning prior — a `GriddedFieldContext` analogue.
- `echoes/posterior.py` package + `data_release/draw_samples.py` sampler schema reused for a
  `data_release/local_*_posterior.npz` (now true-3D positions instead of inverse-CDF z).
- `echoes/inpaint_field.py` constrained-realization fill (`cr` mode) generalizes to 3D voxels
  driven by the reconstructed field instead of the BOSS angular footprint.

## Phased plan

- **P0 — ingest (started).** `data/fetch_cf4.py` pulls the CF4 cubes + catalog into
  `data/local/cf4/`. Next: `data/fetch_manticore.py` via the `manticore_data` package (80-member
  δ+v field ensemble, ~3.9 Mpc) and optional BORG-SDSS Zenodo.
- **P1 — true-3D positions.** `echoes/surveys/local.py::load_local` → a `LocalCatalog` with
  `xyz_data` from real distances (DM→distance for CF4; or Manticore-implied), full-sky angular
  selection with the ZoA mask, conforming to `SurveyCatalog`.
- **P2 — field as prior.** A `GriddedFieldContext` that serves the CF4 (and then Manticore
  per-realization) δ/velocity at any comoving point, feeding the completion the way `fieldpost`
  feeds the BOSS field.
- **P3 — 3D completion.** ECHOES completion of the local catalog: complete flux-limited /
  ZoA-obscured regions in true 3D, conditioned on each field realization → ensemble of
  true-3D completed catalogs (PROV + uncertainty flags as in BOSS).
- **P4 — products + viewer.** `pipeline/build_release_local.py` writes the true-3D posterior
  package + the field cubes; the viewer renders the 3D volume + completed points; validation
  against the held-out reconstruction and CF4 velocities.

## P1 status & a key finding (frame/validation)

Shipped: `echoes/surveys/manticore.py` (80-member field reader; 1+δ=ρ/⟨ρ⟩, v=p/ρ),
`echoes/field_grid.py::GriddedFieldContext` (trilinear sampler of a reconstructed cube at
comoving points), `echoes/surveys/local.py::LocalCatalog`/`load_local_cf4` (true-3D catalog
in supergalactic Mpc, **conforms to `SurveyCatalog`**, ZoA-masked `sel_map`).

**Finding — the field-cube alignment must be validated with the self-consistent pairing.**
Galaxy positions correlate only weakly with the **CF4** δ cube (r≈0.07 even after brute-forcing
all axis/sign frames). This is expected: CF4 distance errors (~15–20%, ±40 Mpc at 200 Mpc) far
exceed the 15.6 Mpc/h voxel, so distance tracers scatter across voxels and the Wiener-filtered
field is intrinsically smooth. The decisive validation is **2M++ galaxies vs the Manticore field**
— self-consistent because Manticore was inferred *from* 2M++ (much denser, nonlinear).

**Resolved (P1b).** Brute-forcing the frame to maximise the 2M++ overdensity alignment gives a
**strong, unambiguous** signal: **mean 1+δ ≈ 4.5** at 2M++ galaxy positions (nearest-voxel), in
the **equatorial Cartesian** frame with **identity axes, observer-centred** — that is the
Manticore frame convention (`manticore_field_context` uses `axis_order=(0,1,2)`, equatorial Mpc).
With trilinear sampling on the field-corrected `load_local_2mpp` catalogue, mean 1+δ ≈ 2.1 (48%
of galaxies in overdensities) — galaxies trace the reconstruction, as they must. So: conditioning
field = **Manticore** (equatorial frame); 2M++ supplies the dense galaxy catalogue; CF4 supplies
direct distances for the sparse distance tracers + an independent velocity check.

## P2/P3 — the true-3D completion engine (shipped)

`echoes/local_completion.py` completes the local catalogue in true 3D by conditioning on the
Manticore field: in the unobserved volume (the **Zone of Avoidance**, |b|<5°) it Poisson-samples
galaxies from `λ ∝ n̄(d)·(1+δ)^b / ⟨(1+δ)^b⟩_shell` — **mass-conserving per distance shell** (the
fill reaches the all-sky mean density `n̄(d)` from the observed galaxies), modulated by the
reconstructed structure, with the galaxy-bias exponent `b` **auto-calibrated** so the fill traces
the field with the same mean over-density as the observed galaxies (faithful, not the
over-concentrated mass field). Filled galaxies carry PROV=5, true distances, a `cz` from
`H0·d + v·n̂`, and **distance-matched K-band magnitudes** (the flux-limited luminosity preserved).

`complete_local_ensemble` / `pipeline/build_local_release.py` runs one completion per Manticore
realization → the posterior product `data_release/local/`: the observed 2M++ base + a per-realization
ZoA completion. Demonstrator (3 realizations, d<300 Mpc): observed 67,966 galaxies + ~11,250–11,360
ZoA galaxies each (the spread across realizations is the reconstruction uncertainty); the fill
matches the observed luminosity (Ks≈11.7) and clustering (mean 1+δ ≈ 1.6 vs observed 2.2), in true
3D, tracing the structures behind the Milky Way (the Great Attractor region etc.). This **composes
the two posterior ensembles** — Manticore's field posterior with the catalogue completion.

**Refinements (shipped).** `echoes/local_completion.py::complete_local` now does the FULL
completion: it fills the ZoA AND restores the **faint galaxies below the flux limit everywhere**,
to a uniform volume-limited density to `m_faint` modulated by the field. The selection is a
**data-driven K-band luminosity function** (`estimate_lf`: `n̄0` + the LF sample from the nearby
complete volume, no Schechter fit); restored galaxies draw absolute mags from the LF fainter than
the local flux limit and carry `K = M + DM(d)`. Each completed galaxy carries a **per-galaxy
`uncert`** (`completion_uncert`): the principled measure is the **ensemble scatter of `1+δ`** across
the Manticore posterior realizations at that position (where the realizations disagree, the
completion is uncertain), with a distance heuristic fallback. `pipeline/build_local_release.py
--mode full` writes the product (volume-limited to `M_K=-22`: 67,966 observed + ~962k completed per
realization, ~half ZoA, half faint-end; large → gitignored, regenerable). A **true-3D interactive
viewer** (`pipeline/build_local_viewer.py` → `docs/local_viewer.html`, k3d) renders the observed +
completed galaxies in comoving 3D — the ZoA fills highlighted, reconstructed behind the Milky Way.

Further work: galaxy bias from the measured clustering (not just the mean-δ match); a Schechter
cross-check of the data-driven LF; SHA/manifest for a versioned release; the full 80-member ensemble.

## Open decisions (resolved 2026-06-22: 2M++ galaxies + CF4 distances; Manticore field; **equatorial** frame)

1. **Anchor galaxy catalog:** 2M++ (Manticore's own input; near-full-sky, ZoA-masked) vs the
   CF4 distance sample vs 2MRS. Recommendation: 2M++ for the catalog + CF4 for real distances,
   so the completion and the Manticore field share an input.
2. **Field source for conditioning:** CF4 Wiener-filter cube (simple, single map) vs the
   **Manticore 80-member posterior ensemble** (full uncertainty; recommended) vs BORG-SDSS.
3. **Coordinate frame:** supergalactic (CF4-native) vs equatorial comoving; pick one canonical
   frame for the product.
4. **Distance vs redshift positions:** use reconstructed real distances throughout (true 3D),
   keeping observed cz only as provenance.

## Citation

Any product built here must cite the sources used: **CF4** — Tully et al. 2023 (ApJ 944, 94)
for distances, Courtois et al. 2023 (A&A 670, L15) for the field cubes; **Manticore** —
McAlpine et al. 2025 (MNRAS 540, 716); **BORG** — Jasche & Wandelt 2013 + the specific
reconstruction paper and its Zenodo/VizieR DOI. No product redistributes the source data; the
fetch scripts pull from the original repositories.
