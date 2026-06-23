"""Per-galaxy geometry + imagery-survey selection for the textured local viewer.

For the real (PROV=0) galaxies of the local-neighborhood product this module derives the
metadata the textured 3D viewer needs but the catalog does not carry:

- ``ang_size_arcmin`` — on-sky angular diameter, used both to size the cutout fetched for
  each galaxy and to drive the level-of-detail (point → sprite → textured quad) switch. From a
  real D25 where a cross-match is available, else estimated from the K-band absolute magnitude
  via a size–luminosity relation.
- ``b_a`` (axis ratio), ``pa_deg`` (position angle, N→E), ``morph`` — so billboards are
  oriented/inclined, not flat face-on circles. From the cross-match; sensible defaults (round,
  PA 0) when absent.
- ``survey_code`` / ``survey_preference`` — which imaging survey to texture each galaxy from, a
  pure function of sky position (the Legacy → Pan-STARRS → DSS2/2MASS waterfall). The actual
  fetch (``pipeline/build_texture_atlas.py``) tries the preference order and validates each tile
  is non-blank before accepting, so an approximate footprint here is safe.

The SGA-2020 / HyperLEDA cross-match is optional: pass tables loaded by ``data/fetch_sga.py`` /
``data/fetch_hyperleda.py`` (positional ``scipy.spatial.cKDTree`` match, ID join on PGC where a
``pgc`` column is supplied). With neither, the module degrades cleanly to the K-mag size estimate
and circular billboards, so the pipeline runs offline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Imaging survey codes (index into SURVEY_HIPS / the viewer's atlas bookkeeping).
SURVEY_LEGACY = 0     # DESI Legacy Surveys DR10 (deepest wide color)
SURVEY_PS1 = 1        # Pan-STARRS1 3pi (Dec > -30)
SURVEY_DSS2 = 2       # DSS2 color (all-sky optical fallback)
SURVEY_2MASS = 3      # 2MASS color (all-sky near-IR; penetrates the Zone of Avoidance)
SURVEY_NONE = -1

SURVEY_HIPS = {
    SURVEY_LEGACY: "CDS/P/DESI-Legacy-Surveys/DR10/color",
    SURVEY_PS1: "CDS/P/PanSTARRS/DR1/color-i-r-g",
    SURVEY_DSS2: "CDS/P/DSS2/color",
    SURVEY_2MASS: "CDS/P/2MASS/color",
}
SURVEY_NAME = {SURVEY_LEGACY: "legacy", SURVEY_PS1: "ps1", SURVEY_DSS2: "dss2",
               SURVEY_2MASS: "2mass", SURVEY_NONE: "none"}


def galactic_latitude_deg(ra, dec):
    """Galactic latitude b [deg] for equatorial (ICRS) ``ra, dec`` [deg] (vectorized)."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    return SkyCoord(np.asarray(ra, float) * u.deg, np.asarray(dec, float) * u.deg,
                    frame="icrs").galactic.b.deg


def absolute_k_mag(ksmag, dist_mpc):
    """K-band absolute magnitude from apparent K_s and comoving distance [Mpc].

    M = m − 5 log10(d_pc/10) = m − 25 − 5 log10(d_Mpc). (Distance modulus only; for the
    nearby volume the small k-/evolution corrections are immaterial to a billboard size.)"""
    d = np.clip(np.asarray(dist_mpc, float), 1e-3, None)
    return np.asarray(ksmag, float) - 25.0 - 5.0 * np.log10(d)


def estimate_d25_kpc(ksmag, dist_mpc, *, d0_kpc=30.0, mk_star=-24.0, beta=0.30,
                     dmin_kpc=3.0, dmax_kpc=120.0):
    """Estimate the physical optical diameter D25 [kpc] from K-band luminosity.

    A simple size–luminosity scaling ``D25 = d0 · 10^(−0.4·β·(M_K − M_K*))`` anchored at an
    L* galaxy (``M_K* = −24`` → ``d0 = 30`` kpc); ``β ≈ 0.3`` matches the shallow optical
    size–luminosity slope of disc+spheroid samples (e.g. Lange+2015). Clipped to a sane range.
    Used only when no measured D25 (SGA/HyperLEDA) is available — it sets the billboard size and
    the cutout FOV, where order-of-magnitude is what matters."""
    mk = absolute_k_mag(ksmag, dist_mpc)
    d25 = d0_kpc * 10.0 ** (-0.4 * beta * (mk - mk_star))
    return np.clip(d25, dmin_kpc, dmax_kpc)


def angular_size_arcmin(ksmag, dist_mpc, **kw):
    """On-sky angular diameter [arcmin] from the K-mag size estimate.

    θ = D25 / d (small angle); arcmin = θ_rad · (180/π) · 60."""
    d25_mpc = estimate_d25_kpc(ksmag, dist_mpc, **kw) / 1000.0
    theta_rad = d25_mpc / np.clip(np.asarray(dist_mpc, float), 1e-3, None)
    return (theta_rad * (180.0 / np.pi) * 60.0).astype(np.float32)


def survey_preference(ra, dec, b_gal=None):
    """Ordered imaging-survey preference per galaxy, from sky position only.

    Outside the Zone of Avoidance: Legacy DR10 (where in its DECam footprint) → Pan-STARRS1
    (Dec > −30) → DSS2 color → 2MASS. Inside the ZoA (|b| < 10°), where the optical surveys are
    heavily extincted, prefer near-IR 2MASS, then DSS2. Footprints are approximate on purpose —
    the atlas builder validates each tile is non-blank and falls through this order, so a galaxy
    flagged Legacy that lands in a Legacy gap simply falls back to PS1/DSS2 at fetch time.

    Returns an int array ``(N, 4)`` of survey codes in priority order (padded/truncated to 4).
    """
    ra = np.asarray(ra, float) % 360.0
    dec = np.asarray(dec, float)
    if b_gal is None:
        b_gal = galactic_latitude_deg(ra, dec)
    b_gal = np.asarray(b_gal, float)
    n = len(ra)

    in_legacy = (dec > -68.0) & (dec < 34.0) & (np.abs(b_gal) > 14.0)   # DECam DR10, off-plane
    in_ps1 = dec > -30.0
    in_zoa = np.abs(b_gal) < 10.0

    out = np.full((n, 4), SURVEY_NONE, np.int8)
    for i in range(n):
        if in_zoa[i]:                                   # near-IR penetrates the dust
            order = [SURVEY_2MASS, SURVEY_DSS2]
            if in_ps1[i]:
                order.append(SURVEY_PS1)
        else:
            order = []
            if in_legacy[i]:
                order.append(SURVEY_LEGACY)
            if in_ps1[i]:
                order.append(SURVEY_PS1)
            order += [SURVEY_DSS2, SURVEY_2MASS]
        order = list(dict.fromkeys(order))[:4]          # dedupe, cap at 4
        out[i, :len(order)] = order
    return out


@dataclass
class GalaxyGeometry:
    """Per-galaxy textured-viewer metadata (all arrays length N, aligned to the input order)."""
    ang_size_arcmin: np.ndarray
    b_a: np.ndarray
    pa_deg: np.ndarray
    morph: np.ndarray            # type code string ('' when unknown)
    survey_pref: np.ndarray      # (N, 4) int8 priority-ordered survey codes
    geom_source: np.ndarray      # 'sga' | 'leda' | 'estimated' per galaxy

    def as_columns(self):
        """Flat dict of per-galaxy columns for the viewer bundle (survey_pref → first pick)."""
        return {
            "ang_size_arcmin": self.ang_size_arcmin.astype(np.float32),
            "b_a": self.b_a.astype(np.float32),
            "pa_deg": self.pa_deg.astype(np.float32),
            "survey_code": self.survey_pref[:, 0].astype(np.int8),
        }


def _xmatch(ra, dec, ref_ra, ref_dec, radius_arcsec=10.0):
    """Nearest reference within ``radius_arcsec`` for each (ra,dec); -1 where none.

    Matches on the unit sphere via ``cKDTree`` (the pattern used in ``echoes.completion``):
    a chord-length tolerance is the small-angle equivalent of the angular radius."""
    from scipy.spatial import cKDTree

    def unit(r, d):
        r = np.radians(np.asarray(r, float)); d = np.radians(np.asarray(d, float)); cd = np.cos(d)
        return np.column_stack([cd * np.cos(r), cd * np.sin(r), np.sin(d)])

    if ref_ra is None or len(np.atleast_1d(ref_ra)) == 0:
        return np.full(len(np.atleast_1d(ra)), -1)
    chord = 2.0 * np.sin(np.radians(radius_arcsec / 3600.0) / 2.0)
    tree = cKDTree(unit(ref_ra, ref_dec))
    dist, idx = tree.query(unit(ra, dec), k=1)
    idx = np.asarray(idx); idx[dist > chord] = -1
    return idx


def enrich_geometry(ra, dec, dist_mpc, ksmag, *, sga=None, leda=None, xmatch_arcsec=10.0):
    """Build :class:`GalaxyGeometry` for galaxies at ``ra, dec, dist_mpc, ksmag``.

    Optional cross-match tables ``sga`` / ``leda`` are dicts with at least
    ``ra, dec, d25_arcmin, b_a, pa_deg`` (and optional ``morph``); SGA is preferred (deeper, has
    cutouts), HyperLEDA fills the rest (all-sky incl. the ZoA). Where matched, the measured D25 /
    axis-ratio / PA override the K-mag estimate / round defaults. ``geom_source`` records the
    provenance per galaxy for QA. With both ``None`` the result is the pure K-mag estimate.
    """
    ra = np.asarray(ra, float); dec = np.asarray(dec, float)
    n = len(ra)
    b_gal = galactic_latitude_deg(ra, dec)

    ang = angular_size_arcmin(ksmag, dist_mpc)
    b_a = np.ones(n, np.float32)
    pa = np.zeros(n, np.float32)
    morph = np.array([""] * n, dtype=object)
    src = np.array(["estimated"] * n, dtype=object)

    # HyperLEDA first (broad), then SGA on top (deeper / preferred) so SGA wins on overlap.
    for table, tag in ((leda, "leda"), (sga, "sga")):
        if not table:
            continue
        idx = _xmatch(ra, dec, table.get("ra"), table.get("dec"), radius_arcsec=xmatch_arcsec)
        m = idx >= 0
        if not m.any():
            continue
        j = idx[m]
        d25 = np.asarray(table["d25_arcmin"], float)[j]
        good = np.isfinite(d25) & (d25 > 0)
        sel = np.where(m)[0][good]; jj = j[good]
        ang[sel] = d25[good].astype(np.float32)
        if "b_a" in table:
            ba = np.asarray(table["b_a"], float)[jj]
            b_a[sel] = np.where(np.isfinite(ba) & (ba > 0), ba, b_a[sel]).astype(np.float32)
        if "pa_deg" in table:
            p = np.asarray(table["pa_deg"], float)[jj]
            pa[sel] = np.where(np.isfinite(p), p, pa[sel]).astype(np.float32)
        if "morph" in table:
            mm = np.asarray(table["morph"], dtype=object)[jj]
            for k, s in zip(sel, mm):
                morph[k] = "" if s is None else str(s)
        src[sel] = tag

    pref = survey_preference(ra, dec, b_gal=b_gal)
    return GalaxyGeometry(ang_size_arcmin=ang, b_a=b_a, pa_deg=pa, morph=morph,
                          survey_pref=pref, geom_source=src)
