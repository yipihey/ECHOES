"""Observational layer for the BOSS CMASS-South LGCP mock catalogs.

The anisotropic-LGCP mocks produced by ``pipeline/build_boss_lgcp_catalogs.py``
are *bare* point sets — ``(ra, dec, z)`` only.  To be consumed by a standard
clustering pipeline (the second mock-catalog use case) they need the same
**observational layer** the real BOSS data carries:

  1. the survey **veto / angular mask** applied (drop points outside the
     CMASS-South footprint / inside veto holes), using the SAME mask the data
     uses;
  2. an FKP weight ``WEIGHT_FKP = 1 / (1 + n(z)·P0)`` with the CMASS standard
     ``P0 = 10⁴ h⁻³ Mpc³`` and the survey radial number density ``n(z)``;
  3. a completeness / systematics weight ``WEIGHT_SYSTOT`` from the angular
     completeness map; and
  4. the ``NZ`` column (``n(z)`` evaluated at each object's redshift).

It then writes a standard BOSS-style FITS file (RA, DEC, Z, WEIGHT_FKP,
WEIGHT_SYSTOT, NZ, WEIGHT) so the mock is a drop-in for the real clustering
catalog.

Conventions are taken straight from the real data and the existing ECHOES
helpers — nothing is reinvented:

* ``n(z)`` and ``WEIGHT_FKP(z)`` are derived from the REAL CMASS-South data.
  We reuse :func:`echoes.completion.fkp_weight_of_z` (which bins the data's own
  ``WEIGHT_FKP`` against z and interpolates) and invert the exact CMASS relation
  ``w_fkp = 1/(1 + n·P0)`` to recover ``n(z)`` — verified on the data to
  ``max|w_fkp − 1/(1+NZ·10⁴)| ≈ 8e-8``, i.e. P0 = 10⁴ exactly.

* The angular **mask + completeness** is the analytic mangle-rasterised
  selection ``boss_selection_2048.npz`` loaded exactly as
  :func:`echoes.fill_footprint.load_analytic_completeness` does (the shot-noise-
  free completeness × Π(1−veto) product, RING order).  We use **the healpix
  completeness map, NOT pymangle**: ``pymangle`` is installed but broken under
  NumPy 2.x in this environment (``_ARRAY_API not found`` / ImportError), and in
  any case the healpix selection map is what the rest of ECHOES uses as its
  ``sel_map``.  A point is vetoed where the completeness is ≤ ``veto_thresh``
  (default 0); the surviving completeness value becomes its ``WEIGHT_SYSTOT``.

CLI / demonstration (bottom of the file)::

    JAX_PLATFORMS=cpu python -m echoes.boss_mock_columns

processes ``data_release/boss_lgcp_julia/cmass_south_lgcp_julia_*.npz`` into
``data_release/boss_lgcp_julia/fits/`` and prints the validation comparison
(footprint fraction, WEIGHT_FKP mean mock-vs-data, n(z) match).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .completion import fkp_weight_of_z

# CMASS standard FKP P0 (h⁻³ Mpc³) — verified on the data: WEIGHT_FKP = 1/(1+NZ·P0).
P0_CMASS = 10000.0

# Repo-relative default inputs (this module lives in echoes/, repo root is its parent).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_FITS = os.path.join(_REPO_ROOT, "data", "boss",
                                 "galaxy_DR12v5_CMASS_South.fits.gz")
DEFAULT_SELECTION_NPZ = os.path.join(_REPO_ROOT, "data", "boss",
                                     "boss_selection_2048.npz")
DEFAULT_MASK_PLY = os.path.join(_REPO_ROOT, "data", "boss",
                                "mask_DR12v5_CMASS_South.ply")

# simBIG CMASS-South clustering cut (same as echoes.surveys.boss.SIMBIG_CUTS["CMASS"]
# + _in_sgc_footprint); used to derive the survey n(z)/FKP profile from the data.
_Z_MIN, _Z_MAX = 0.45, 0.60
_SGC_RA_HI, _SGC_RA_LO, _DEC_MIN = 28.0, 335.0, -6.0


# ──────────────────────────────────────────────────────────────────────
# Survey radial selection n(z) / FKP(z), learned from the real data
# ──────────────────────────────────────────────────────────────────────
@dataclass
class SurveyRadialSelection:
    """The survey's radial selection learned from the real CMASS-South data.

    Carries the data redshifts and per-object ``WEIGHT_FKP`` of the simBIG
    clustering subsample, from which ``WEIGHT_FKP(z)`` (via
    :func:`echoes.completion.fkp_weight_of_z`) and ``n(z)`` (inverting the FKP
    relation) are evaluated at arbitrary redshift.
    """
    z_data: np.ndarray            # data redshifts (simBIG CMASS-South cut)
    w_fkp_data: np.ndarray        # per-object WEIGHT_FKP of those galaxies
    P0: float = P0_CMASS

    def w_fkp_of_z(self, z: np.ndarray) -> np.ndarray:
        """Smooth ``WEIGHT_FKP(z)`` profile learned from the data (ECHOES helper)."""
        return fkp_weight_of_z(np.asarray(z, np.float64), self.z_data, self.w_fkp_data)

    def n_of_z(self, z: np.ndarray) -> np.ndarray:
        """Survey number density ``n(z)`` (h³ Mpc⁻³) by inverting w_fkp = 1/(1+n·P0).

        ``n(z) = (1/w_fkp(z) − 1) / P0`` — the exact CMASS relation, so the NZ
        column and the FKP weight are mutually consistent (and match the data's
        own NZ ↔ WEIGHT_FKP relation by construction).
        """
        w = np.clip(self.w_fkp_of_z(z), 1e-12, 1.0)
        return (1.0 / w - 1.0) / self.P0


def survey_selection_from_data(
    data_fits: str = DEFAULT_DATA_FITS,
    *,
    P0: float = P0_CMASS,
    simbig_cut: bool = True,
) -> SurveyRadialSelection:
    """Build :class:`SurveyRadialSelection` from the real CMASS-South FITS.

    Reads ``Z`` and ``WEIGHT_FKP`` from ``galaxy_DR12v5_CMASS_South.fits.gz`` and
    (by default) restricts to the simBIG CMASS-South clustering subsample
    (0.45<z<0.60, SGC footprint) so the learned ``n(z)``/FKP(z) is the radial
    selection of the SAME sample the mocks emulate.
    """
    from astropy.io import fits

    with fits.open(data_fits, memmap=True) as hdul:
        t = hdul[1].data
        ra = np.asarray(t["RA"], np.float64)
        dec = np.asarray(t["DEC"], np.float64)
        z = np.asarray(t["Z"], np.float64)
        w_fkp = np.asarray(t["WEIGHT_FKP"], np.float64)

    m = np.isfinite(w_fkp) & (w_fkp > 0)
    if simbig_cut:
        ra_ok = (ra < _SGC_RA_HI) | (ra > _SGC_RA_LO)
        m = m & ra_ok & (dec > _DEC_MIN) & (z >= _Z_MIN) & (z <= _Z_MAX)
    return SurveyRadialSelection(z_data=z[m], w_fkp_data=w_fkp[m], P0=float(P0))


# ──────────────────────────────────────────────────────────────────────
# Angular mask + completeness (analytic healpix selection map; no pymangle)
# ──────────────────────────────────────────────────────────────────────
@dataclass
class AngularCompleteness:
    """HEALPix angular completeness/selection map and its lookup.

    ``sel_map`` is the dense RING-order completeness × Π(1−veto) product loaded
    from the mangle-rasterised ``boss_selection_2048.npz`` (ud_grade'd to
    ``nside``) — identical to
    :func:`echoes.fill_footprint.load_analytic_completeness`.  ``value_at``
    returns the completeness at each (ra, dec); ``inside`` is the survey
    veto/mask (completeness above ``veto_thresh``).
    """
    sel_map: np.ndarray
    nside: int
    veto_thresh: float = 0.0

    def value_at(self, ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
        import healpy as hp
        pix = hp.ang2pix(self.nside, np.radians(90.0 - np.asarray(dec, np.float64)),
                         np.radians(np.asarray(ra, np.float64) % 360.0))
        return self.sel_map[pix]

    def inside(self, ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
        """Boolean survey mask: True where the object survives veto + footprint."""
        return self.value_at(ra, dec) > self.veto_thresh


def load_angular_completeness(
    nside: int = 512,
    *,
    selection_npz: str = DEFAULT_SELECTION_NPZ,
    veto_thresh: float = 0.0,
) -> AngularCompleteness:
    """Load the analytic CMASS-South completeness map at ``nside`` (RING order).

    Reconstructs the dense map from the sparse mangle-rasterised
    ``boss_selection_2048.npz`` and averages it to ``nside`` with ``ud_grade``
    (the exact fractional completeness) — exactly as
    :func:`echoes.fill_footprint.load_analytic_completeness`.  We default to
    ``nside=512`` (the resolution at which the mock candidates were drawn) so
    the veto edge matches the mock generation; the high-res map is averaged
    down, preserving the fractional completeness.
    """
    import healpy as hp

    d = np.load(selection_npz)
    n_hi = int(d["nside"])
    m = np.zeros(12 * n_hi ** 2, np.float64)
    m[d["ipix"]] = d["sel"]
    if nside != n_hi:
        m = hp.ud_grade(m, nside_out=nside, power=0)
    return AngularCompleteness(sel_map=m.astype(np.float64), nside=int(nside),
                               veto_thresh=float(veto_thresh))


@dataclass
class MangleCompleteness:
    """Exact mangle polygon mask + completeness (`value_at`/`inside` interface matching
    :class:`AngularCompleteness`). ``value_at`` returns the polygon weight (the per-point angular
    completeness) and 0 outside the footprint — the EXACT mask the data uses, not a healpix raster."""
    mng: object
    veto_thresh: float = 0.0

    def value_at(self, ra, dec):
        return np.asarray(self.mng.weight(np.asarray(ra, np.float64), np.asarray(dec, np.float64)),
                          np.float64)

    def inside(self, ra, dec):
        return self.value_at(ra, dec) > self.veto_thresh


def load_mangle_completeness(ply: str = DEFAULT_MASK_PLY, *, veto_thresh: float = 0.0):
    """Load the exact CMASS-South mangle mask via pymangle. Returns a :class:`MangleCompleteness`,
    or ``None`` if pymangle is unavailable/broken (caller falls back to the healpix map)."""
    try:
        import pymangle
        return MangleCompleteness(pymangle.Mangle(str(ply)), float(veto_thresh))
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────
# Main API: add the observational columns to a mock's (ra, dec, z)
# ──────────────────────────────────────────────────────────────────────
def add_observational_columns(
    ra: np.ndarray,
    dec: np.ndarray,
    z: np.ndarray,
    *,
    P0: float = P0_CMASS,
    selection: Optional[SurveyRadialSelection] = None,
    completeness: Optional[AngularCompleteness] = None,
    data_fits: str = DEFAULT_DATA_FITS,
    selection_npz: str = DEFAULT_SELECTION_NPZ,
    nside: int = 512,
    veto_thresh: float = 0.0,
    apply_mask: bool = True,
) -> dict:
    """Attach the BOSS observational layer to a mock's bare ``(ra, dec, z)``.

    Steps (all using the SAME conventions as the real data):

      1. **veto/mask**: drop points where the angular completeness is
         ≤ ``veto_thresh`` (i.e. outside the footprint / inside a veto hole),
         when ``apply_mask`` (default).  Uses the EXACT mangle mask via pymangle
         when available, else the healpix completeness map.
      2. ``WEIGHT_FKP = 1 / (1 + n(z)·P0)`` with ``n(z)`` the SURVEY radial
         density learned from the data (so the mock's FKP weights match the
         data's by construction).
      3. ``WEIGHT_SYSTOT`` = the angular completeness at each surviving object.
      4. ``NZ`` = ``n(z)`` at each object's redshift.
      5. ``WEIGHT`` = ``WEIGHT_FKP · WEIGHT_SYSTOT`` (the combined clustering
         weight; the LGCP mock has no fiber-collision / z-failure components, so
         WEIGHT_CP = WEIGHT_NOZ = 1 and are folded into WEIGHT_SYSTOT).

    Returns a ``dict`` of equal-length float64 arrays:
    ``ra, dec, z, WEIGHT_FKP, WEIGHT_SYSTOT, NZ, WEIGHT`` plus scalars
    ``n_in``/``n_total``/``mask_fraction``.
    """
    ra = np.asarray(ra, np.float64)
    dec = np.asarray(dec, np.float64)
    z = np.asarray(z, np.float64)

    if selection is None:
        selection = survey_selection_from_data(data_fits, P0=P0)
    if completeness is None:
        # prefer the EXACT mangle mask; fall back to the healpix completeness map if pymangle absent
        completeness = load_mangle_completeness(veto_thresh=veto_thresh) or \
            load_angular_completeness(nside, selection_npz=selection_npz, veto_thresh=veto_thresh)

    n_total = len(ra)
    compl = completeness.value_at(ra, dec)
    keep = (compl > veto_thresh) if apply_mask else np.ones(n_total, bool)

    ra, dec, z, compl = ra[keep], dec[keep], z[keep], compl[keep]

    n_z = selection.n_of_z(z)                       # survey n(z) at each object
    w_fkp = 1.0 / (1.0 + n_z * P0)                  # CMASS FKP weight
    w_systot = compl                                # completeness as systot weight
    weight = w_fkp * w_systot

    return {
        "ra": ra, "dec": dec, "z": z,
        "WEIGHT_FKP": w_fkp,
        "WEIGHT_SYSTOT": w_systot,
        "NZ": n_z,
        "WEIGHT": weight,
        "n_total": int(n_total),
        "n_in": int(keep.sum()),
        "mask_fraction": float(keep.mean()) if n_total else 0.0,
    }


def write_mock_fits(path: str, cols: dict, *, overwrite: bool = True) -> str:
    """Write the observational columns to a standard BOSS-style FITS file.

    Columns: ``RA, DEC, Z, WEIGHT_FKP, WEIGHT_SYSTOT, NZ, WEIGHT``.  ``cols`` is
    the dict returned by :func:`add_observational_columns` (extra scalar keys
    are ignored).  Returns ``path``.
    """
    from astropy.table import Table

    t = Table()
    t["RA"] = np.asarray(cols["ra"], np.float64)
    t["DEC"] = np.asarray(cols["dec"], np.float64)
    t["Z"] = np.asarray(cols["z"], np.float64)
    t["WEIGHT_FKP"] = np.asarray(cols["WEIGHT_FKP"], np.float64)
    t["WEIGHT_SYSTOT"] = np.asarray(cols["WEIGHT_SYSTOT"], np.float64)
    t["NZ"] = np.asarray(cols["NZ"], np.float64)
    t["WEIGHT"] = np.asarray(cols["WEIGHT"], np.float64)
    t.meta["P0_FKP"] = P0_CMASS
    t.meta["SAMPLE"] = "CMASS_South_LGCP_mock"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    t.write(path, format="fits", overwrite=overwrite)
    return path


def process_mock_npz(
    npz_path: str,
    fits_path: str,
    *,
    selection: Optional[SurveyRadialSelection] = None,
    completeness: Optional[AngularCompleteness] = None,
    **kwargs,
) -> dict:
    """Load a mock ``.npz`` (ra, dec, z), add the observational layer, write FITS.

    Returns the column dict (so callers can validate without re-reading).
    Pass a shared ``selection``/``completeness`` to amortise their construction
    across a batch of mocks.
    """
    d = np.load(npz_path)
    cols = add_observational_columns(
        d["ra"], d["dec"], d["z"],
        selection=selection, completeness=completeness, **kwargs)
    write_mock_fits(fits_path, cols)
    return cols


# ──────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────
def _data_reference(data_fits: str = DEFAULT_DATA_FITS) -> dict:
    """Real CMASS-South (simBIG cut) reference: RA/Dec ranges, WEIGHT_FKP mean,
    n(z) histogram — for the validation comparison."""
    from astropy.io import fits
    with fits.open(data_fits, memmap=True) as hdul:
        t = hdul[1].data
        ra = np.asarray(t["RA"], np.float64); dec = np.asarray(t["DEC"], np.float64)
        z = np.asarray(t["Z"], np.float64)
        w_fkp = np.asarray(t["WEIGHT_FKP"], np.float64)
        nz = np.asarray(t["NZ"], np.float64)
    ra_ok = (ra < _SGC_RA_HI) | (ra > _SGC_RA_LO)
    m = ra_ok & (dec > _DEC_MIN) & (z >= _Z_MIN) & (z <= _Z_MAX) & np.isfinite(w_fkp)
    return {
        "N": int(m.sum()),
        "ra": ra[m], "dec": dec[m], "z": z[m],
        "w_fkp_mean": float(w_fkp[m].mean()),
        "nz_mean": float(nz[m].mean()),
        "z_hist_edges": np.linspace(_Z_MIN, _Z_MAX, 11),
    }


def validate_against_data(cols: dict, *, data_fits: str = DEFAULT_DATA_FITS) -> str:
    """Compare a processed mock against the real CMASS-South data and return a
    printable report: footprint/mask fraction, RA/Dec/z ranges, WEIGHT_FKP mean
    (mock vs data), and normalised n(z) per z-bin (mock vs data)."""
    ref = _data_reference(data_fits)
    z_edges = ref["z_hist_edges"]

    mock_h, _ = np.histogram(cols["z"], bins=z_edges)
    data_h, _ = np.histogram(ref["z"], bins=z_edges)
    mock_n = mock_h / mock_h.sum()
    data_n = data_h / data_h.sum()
    zc = 0.5 * (z_edges[1:] + z_edges[:-1])

    lines = []
    L = lines.append
    L("=" * 70)
    L("VALIDATION — LGCP mock observational layer vs real CMASS-South data")
    L("=" * 70)
    L("(a) FOOTPRINT / MASK")
    L(f"    mock points: total={cols['n_total']:,}  kept(in mask)={cols['n_in']:,}  "
      f"mask_fraction={cols['mask_fraction']:.4f}")
    L(f"    mock  RA  range: [{cols['ra'].min():8.3f}, {cols['ra'].max():8.3f}]   "
      f"data RA  range: [{ref['ra'].min():8.3f}, {ref['ra'].max():8.3f}]")
    L(f"    mock  Dec range: [{cols['dec'].min():8.3f}, {cols['dec'].max():8.3f}]   "
      f"data Dec range: [{ref['dec'].min():8.3f}, {ref['dec'].max():8.3f}]")
    L(f"    mock  z   range: [{cols['z'].min():.4f}, {cols['z'].max():.4f}]   "
      f"data z   range: [{ref['z'].min():.4f}, {ref['z'].max():.4f}]")
    L(f"    angular completeness (WEIGHT_SYSTOT) mock mean={cols['WEIGHT_SYSTOT'].mean():.4f}  "
      f"(data COMP~0.988, sel-map mean is the analytic completeness)")
    L("(b) WEIGHT_FKP")
    L(f"    mock mean={cols['WEIGHT_FKP'].mean():.4f}   data mean={ref['w_fkp_mean']:.4f}   "
      f"ratio={cols['WEIGHT_FKP'].mean()/ref['w_fkp_mean']:.4f}")
    L(f"    mock [min,max]=[{cols['WEIGHT_FKP'].min():.4f},{cols['WEIGHT_FKP'].max():.4f}]")
    L("(c) n(z)  — normalised redshift distribution per z-bin (mock vs data)")
    L(f"    {'z_centre':>9} {'mock_frac':>10} {'data_frac':>10} {'ratio':>8}")
    for i in range(len(zc)):
        r = mock_n[i] / data_n[i] if data_n[i] > 0 else float("nan")
        L(f"    {zc[i]:9.4f} {mock_n[i]:10.4f} {data_n[i]:10.4f} {r:8.3f}")
    L(f"    sum|mock_frac - data_frac| = {np.abs(mock_n - data_n).sum():.4f}  "
      f"(0 = identical shape)")
    L("=" * 70)
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# CLI / demonstration
# ──────────────────────────────────────────────────────────────────────
def _main(argv=None) -> int:
    import argparse
    import glob

    ap = argparse.ArgumentParser(
        description="Add the BOSS observational layer (mask, WEIGHT_FKP, "
                    "WEIGHT_SYSTOT, NZ) to LGCP mock catalogs and write FITS.")
    ap.add_argument("--in-glob",
                    default=os.path.join(_REPO_ROOT, "data_release", "boss_lgcp_julia",
                                         "cmass_south_lgcp_julia_*.npz"),
                    help="glob of mock .npz files to process")
    ap.add_argument("--out-dir",
                    default=os.path.join(_REPO_ROOT, "data_release", "boss_lgcp_julia",
                                         "fits"),
                    help="output directory for FITS files")
    ap.add_argument("--data-fits", default=DEFAULT_DATA_FITS)
    ap.add_argument("--selection-npz", default=DEFAULT_SELECTION_NPZ)
    ap.add_argument("--nside", type=int, default=512)
    ap.add_argument("--P0", type=float, default=P0_CMASS)
    ap.add_argument("--veto-thresh", type=float, default=0.0)
    ap.add_argument("--no-mask", action="store_true",
                    help="skip the veto/mask step (keep all points)")
    ap.add_argument("--validate", action="store_true",
                    help="print the data comparison for the first processed mock")
    args = ap.parse_args(argv)

    paths = sorted(glob.glob(args.in_glob))
    if not paths:
        print(f"[boss-mock-columns] no inputs matched {args.in_glob}")
        return 1

    # Build the (expensive) data-derived selection + completeness ONCE, share across mocks.
    selection = survey_selection_from_data(args.data_fits, P0=args.P0)
    completeness = load_angular_completeness(
        args.nside, selection_npz=args.selection_npz, veto_thresh=args.veto_thresh)
    print(f"[boss-mock-columns] survey selection from {os.path.basename(args.data_fits)} "
          f"(N_data={len(selection.z_data):,}); completeness map nside={completeness.nside}")
    print(f"[boss-mock-columns] mask=healpix analytic completeness (pymangle unavailable "
          f"under NumPy 2.x); veto_thresh={args.veto_thresh}")

    os.makedirs(args.out_dir, exist_ok=True)
    first_cols = None
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        out = os.path.join(args.out_dir, stem + ".fits")
        cols = process_mock_npz(
            p, out, selection=selection, completeness=completeness,
            P0=args.P0, veto_thresh=args.veto_thresh, apply_mask=not args.no_mask)
        print(f"[boss-mock-columns] {os.path.basename(p)} -> {os.path.basename(out)}  "
              f"({cols['n_in']:,}/{cols['n_total']:,} kept, "
              f"mask_frac={cols['mask_fraction']:.4f}, "
              f"<w_fkp>={cols['WEIGHT_FKP'].mean():.4f})")
        if first_cols is None:
            first_cols = cols

    if args.validate and first_cols is not None:
        print()
        print(validate_against_data(first_cols, data_fits=args.data_fits))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
