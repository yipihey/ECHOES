"""DESI DR1 QSO — a ``restore_missing`` dataset with extreme uneven coverage.

DESI DR1 LSS QSO clustering catalogs (Ross et al. 2024) give spectroscopic
redshifts plus the standard weight decomposition. Unlike BOSS there is no
colour-bearing imaging target file shipped with the clustering catalog, so the
spectroscopically-missing objects cannot be restored at real imaging positions
with photo-z colours. Instead ECHOES restores them as **local analog clones** of
nearby observed quasars (no colours → a field-only redshift posterior), with the
per-object count set by the completeness deficit ``WEIGHT_COMP·WEIGHT_ZFAIL − 1``.

This makes DESI a stress test of the ``restore_missing`` path: the completeness
varies violently across the footprint (``WEIGHT_COMP`` up to ~6 in single-tile
regions, ~1 where many tiles overlap), so the restored-object density tracks the
local coverage automatically — many objects are restored where coverage is thin,
almost none where it is complete.

Public files (download into ``data/desi/``):
``QSO_{NGC,SGC}_clustering.dat.fits`` + ``QSO_{NGC,SGC}_{0..17}_clustering.ran.fits``
from ``data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from ..distance import DistanceCosmo, radec_z_to_cartesian

# DESI DR1 fiducial cosmology (matches the graphGP-cosmology research loader)
_DESI_COSMO = DistanceCosmo(Om=0.31, h=0.68, w0=-1.0, wa=0.0)


@dataclass
class DESICatalog:
    """DESI DR1 QSO catalog satisfying the ECHOES ``SurveyCatalog`` contract.

    The completeness deficit that drives ``restore_missing`` is
    ``w_comp_data = WEIGHT_COMP · WEIGHT_ZFAIL`` (fiber-assignment + redshift-failure
    incompleteness). ``WEIGHT_SYS`` (smooth imaging systematic) is kept for the
    record but is NOT restored as objects (it is a few-percent up/down-modulation,
    not missing quasars); ``w_sys_data`` is therefore ``None`` (systot no-op).
    """
    ra_data: np.ndarray
    dec_data: np.ndarray
    z_data: np.ndarray                 # spectroscopic redshift
    ra_random: np.ndarray
    dec_random: np.ndarray
    z_random: np.ndarray
    sel_map: np.ndarray                # HEALPix angular completeness, peak-normalised
    nside: int
    w_data: np.ndarray                 # WEIGHT · WEIGHT_FKP (clustering measurement)
    w_comp_data: np.ndarray            # WEIGHT_COMP · WEIGHT_ZFAIL (completeness deficit)
    w_imaging_data: np.ndarray         # WEIGHT_SYS (diagnostic only)
    ntile_data: np.ndarray             # number of overlapping tiles (uneven-coverage diagnostic)
    frac_tlobs_data: np.ndarray        # fraction of overlapping tiles actually observed
    photsys_data: np.ndarray
    fid_cosmo: DistanceCosmo
    xyz_data: Optional[np.ndarray] = None       # (N,3) comoving Mpc/h (for the graphGP field engine)
    xyz_random: Optional[np.ndarray] = None
    w_sys_data: Optional[np.ndarray] = None     # None => build_package systot is a no-op

    @property
    def N_data(self) -> int:
        return len(self.ra_data)

    def shift_to_positive(self, margin: float = 100.0):
        """(positions_data, positions_random, box_size) shifted so all coords ≥ margin
        (the graphGP density-field engine's cKDTree needs non-negative coordinates)."""
        all_xyz = np.concatenate([self.xyz_data, self.xyz_random], axis=0)
        shift = -all_xyz.min(axis=0) + margin
        pos = self.xyz_data + shift
        rnd = self.xyz_random + shift
        box = float(np.max(np.concatenate([pos, rnd]).max(axis=0)) + margin)
        return pos, rnd, box


def _read_clustering_fits(path):
    """Read RA/DEC/Z + weight components from one DESI clustering FITS.

    Returns ``(ra, dec, z, w_clust, w_comp, w_sys, ntile, frac_tlobs, photsys)``
    where ``w_clust = WEIGHT·WEIGHT_FKP`` (for clustering), ``w_comp =
    WEIGHT_COMP·WEIGHT_ZFAIL`` (the completeness deficit), ``w_sys = WEIGHT_SYS``."""
    from astropy.io import fits

    with fits.open(path, memmap=True) as hdul:
        t = hdul[1].data
        cols = {c.upper() for c in t.columns.names}

        def col(name, default=None):
            if name.upper() in cols:
                return np.asarray(t[name], np.float64)
            return default

        ra = col("RA"); dec = col("DEC")
        z = col("Z")
        if z is None:
            z = col("Z_QSO_HP")
        if ra is None or dec is None or z is None:
            raise ValueError(f"{path}: required columns RA/DEC/Z missing")
        n = len(ra)

        weight = col("WEIGHT", np.ones(n))
        w_fkp = col("WEIGHT_FKP", np.ones(n))
        w_comp = col("WEIGHT_COMP", np.ones(n)) * col("WEIGHT_ZFAIL", np.ones(n))
        w_sys = col("WEIGHT_SYS", np.ones(n))
        ntile = col("NTILE", np.ones(n))
        frac = col("FRAC_TLOBS_TILES", np.ones(n))

        if "PHOTSYS" in cols:
            raw = np.asarray(t["PHOTSYS"])
            photsys = (np.char.decode(raw, "ascii") if raw.dtype.kind == "S" else raw).astype("U1")
        else:
            photsys = np.empty(0, dtype="U1")

        return ra, dec, z, weight * w_fkp, w_comp, w_sys, ntile, frac, photsys


def angular_completeness_from_randoms(ra_random, dec_random, nside=256):
    """HEALPix angular completeness from the random catalog (random sky density IS
    the completeness function); peak-normalised to the median populated pixel."""
    import healpy as hp
    npix = 12 * nside ** 2
    pix = hp.ang2pix(nside, np.deg2rad(90.0 - np.asarray(dec_random, np.float64)),
                     np.deg2rad(np.asarray(ra_random, np.float64) % 360.0))
    counts = np.bincount(pix, minlength=npix).astype(np.float64)
    populated = counts[counts > 0]
    if populated.size == 0:
        raise ValueError("no random objects in any pixel")
    return np.clip(counts / np.median(populated), 0.0, 1.0)


def load_desi_qso(catalog_paths, random_paths=None, *, fid_cosmo=None, nside=256,
                  z_min=0.8, z_max=2.1, n_random_max=None, rng_seed=0):
    """Load the DESI DR1 QSO clustering catalog (NGC + SGC concatenated).

    ``catalog_paths`` and ``random_paths`` are iterables of FITS paths. The angular
    completeness ``sel_map`` is built from the randoms (required for the footprint).
    """
    if fid_cosmo is None:
        fid_cosmo = _DESI_COSMO
    rng = np.random.default_rng(rng_seed)

    parts = {k: [] for k in ("ra", "dec", "z", "wc", "wcomp", "wsys", "nt", "ft", "ps")}
    for p in catalog_paths:
        ra, dec, z, wc, wcomp, wsys, nt, ft, ps = _read_clustering_fits(p)
        m = (z >= z_min) & (z <= z_max) & np.isfinite(wc) & (wc > 0)
        for key, arr in zip(("ra", "dec", "z", "wc", "wcomp", "wsys", "nt", "ft"),
                            (ra, dec, z, wc, wcomp, wsys, nt, ft)):
            parts[key].append(arr[m])
        parts["ps"].append(ps[m] if ps.size else np.full(int(m.sum()), "?", dtype="U1"))
    cat = {k: np.concatenate(v) for k, v in parts.items()}

    if random_paths is not None:
        rr = {k: [] for k in ("ra", "dec", "z")}
        for p in random_paths:
            ra, dec, z, *_ = _read_clustering_fits(p)
            m = (z >= z_min) & (z <= z_max)
            rr["ra"].append(ra[m]); rr["dec"].append(dec[m]); rr["z"].append(z[m])
        ra_random = np.concatenate(rr["ra"]); dec_random = np.concatenate(rr["dec"])
        z_random = np.concatenate(rr["z"])
        if n_random_max is not None and len(ra_random) > n_random_max:
            idx = rng.choice(len(ra_random), n_random_max, replace=False)
            ra_random, dec_random, z_random = ra_random[idx], dec_random[idx], z_random[idx]
        sel_map = angular_completeness_from_randoms(ra_random, dec_random, nside=nside)
    else:
        raise ValueError("DESI loader requires random_paths to build the angular sel_map")

    xyz_data = np.asarray(radec_z_to_cartesian(cat["ra"], cat["dec"], cat["z"], fid_cosmo))
    xyz_random = np.asarray(radec_z_to_cartesian(ra_random, dec_random, z_random, fid_cosmo))
    return DESICatalog(
        ra_data=cat["ra"], dec_data=cat["dec"], z_data=cat["z"],
        ra_random=ra_random, dec_random=dec_random, z_random=z_random,
        sel_map=sel_map, nside=nside,
        w_data=cat["wc"], w_comp_data=cat["wcomp"], w_imaging_data=cat["wsys"],
        ntile_data=cat["nt"], frac_tlobs_data=cat["ft"], photsys_data=cat["ps"],
        fid_cosmo=fid_cosmo, xyz_data=xyz_data, xyz_random=xyz_random, w_sys_data=None,
    )


# ---- the spectroscopically-missing set: local analog clones ------------------

@dataclass
class DESITargets:
    """DESI coverage-missing quasars (analog clones; no colours → field-only z)."""
    ra: np.ndarray
    dec: np.ndarray
    colors: np.ndarray                 # (M, 0) — no imaging colours
    mags: Optional[np.ndarray]         # None
    miss_kind: np.ndarray              # (M,) all "coverage"
    host_index: np.ndarray             # (M,) index of the cloned observed object

    @property
    def N(self) -> int:
        return len(self.ra)


def load_desi_targets(catalog, *, seed=0, sel_floor=0.05):
    """Build the DESI coverage-missing set from the completeness deficit.

    Two parts, deliberately separated:

    - **count** — the total number of missing objects is the integrated completeness
      deficit ``Σ floor(max(WEIGHT_COMP·WEIGHT_ZFAIL − 1, 0) + u)`` over observed
      quasars (recovers the weighted count to <0.1%).
    - **placement** — the missing objects are NOT clones of observed quasars. DESI's
      incompleteness lives in spatially-extended under-tiled gaps, so cloning observed
      objects would just amplify the observed angular pattern (leaving large-scale
      structure imprinted) rather than filling the gaps. Instead we sample their
      positions from the survey **randoms** (which trace the footprint) weighted by the
      local missing fraction ``(1/c − 1)`` with ``c`` the pixel completeness. The survey
      randoms have angular density ∝ ``c``, so weighting by ``(1/c − 1)`` yields restored
      objects with density ∝ ``(1 − c)`` — exactly the deficit, spread uniformly across
      the under-observed footprint. Each restored object's redshift is then the
      ``build_package`` field-only posterior (local LOS density of observed quasars,
      reverting to the broad n(z) where coverage is thin — the honest prior-dominated
      regime). ``host_index`` is the nearest observed quasar (a redshift fallback only).
    """
    import healpy as hp
    from scipy.spatial import cKDTree
    from ..geometry import _radec_to_nhat
    rng = np.random.default_rng(seed)
    ra_d = np.asarray(catalog.ra_data, np.float64)
    dec_d = np.asarray(catalog.dec_data, np.float64)
    w = np.asarray(catalog.w_comp_data, np.float64)

    n_total = int(np.floor(np.maximum(w - 1.0, 0.0) + rng.random(len(w))).sum())

    ra_r = np.asarray(catalog.ra_random, np.float64)
    dec_r = np.asarray(catalog.dec_random, np.float64)
    rpix = hp.ang2pix(catalog.nside, np.deg2rad(90.0 - dec_r), np.deg2rad(ra_r % 360.0))
    c_r = np.clip(catalog.sel_map[rpix], sel_floor, 1.0)
    place_w = 1.0 / c_r - 1.0                                  # deficit weight => density ∝ (1-c)
    sw = place_w.sum()
    if sw <= 0:
        raise ValueError("no completeness deficit to restore (sel_map ~ uniform)")
    idx = rng.choice(len(ra_r), size=n_total, p=place_w / sw)
    # sub-pixel jitter to break exact duplicates from sampling-with-replacement
    s = 0.4 * np.degrees(np.sqrt(4.0 * np.pi / (12.0 * catalog.nside ** 2)))
    dec = np.clip(dec_r[idx] + rng.normal(0.0, s, n_total), -89.999, 89.999)
    ra = (ra_r[idx] + rng.normal(0.0, s, n_total) / np.cos(np.radians(dec))) % 360.0

    _, host = cKDTree(_radec_to_nhat(ra_d, dec_d)).query(_radec_to_nhat(ra, dec), workers=-1)
    return DESITargets(
        ra=ra, dec=dec, colors=np.empty((n_total, 0), np.float64), mags=None,
        miss_kind=np.full(n_total, "coverage"), host_index=host.astype(int))


# ---- registry ----------------------------------------------------------------

from .registry import register, DatasetSpec          # noqa: E402

register(DatasetSpec(
    name="desi_qso",
    load_catalog=load_desi_qso,
    completion_mode="restore_missing",
    build_targets=load_desi_targets,
    photoz_features=None,               # no imaging colours → field-only redshift posterior
    fid_cosmo=_DESI_COSMO,
    nside=256,
    product_prefix="desi_qso",
    default_engine="field",
    copula=True,
    uniform_randoms=True,               # NTILE coverage is imprinted on the survey randoms
))
