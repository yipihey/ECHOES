"""Fill footprint: WHERE the generative completion must inpaint galaxies.

The completed-everywhere ECHOES product fills the survey's veto-mask holes and
empty regions so the catalog is uniform. This module defines, on a HEALPix grid,
three angular fields and the regions derived from them:

  * ``observed_cover``  – completeness traced by the survey randoms (0 in veto
    holes, ~1 in covered area), from :func:`echoes.inpaint.fine_completeness_map`.
  * ``target_mask``     – the *intended-complete* survey footprint (the area that
    should hold galaxies at survey density). Built from the BOSS mangle GEOMETRY
    mask, which is ~40% larger than the LSS footprint, **clipped to within
    ``lss_clip_deg`` of the survey randoms**. That clip is the integrity guard: we
    never invent galaxies more than a correlation scale from real data, and we
    never spill outside the genuine survey volume. (Fallback when no mangle file is
    present: a morphological closing of the survey-random coverage, which fills
    enclosed holes without extending the outer boundary.)
  * ``fill_weight = target_mask · clip(1 - observed_cover, 0, 1)`` – the per-pixel
    solid-angle fraction that needs inpainting. ≈0 in covered area (already handled
    by the spec-missing restoration → no double counting), ≈target_mask in holes.

``hole_pix`` (small, data-surrounded interior holes — Regime D) and ``empty_pix``
(the remaining fill area — Regime P) split the fill region for engine dispatch and
the per-galaxy prior-dominance flag.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .inpaint import fine_completeness_map, find_interior_holes

DEFAULT_MANGLE_NPY = "data/boss/mangle_uniform_radec.npy"
DEFAULT_SELECTION_NPZ = "data/boss_selection_2048.npz"


def load_analytic_completeness(nside, *, selection_npz=DEFAULT_SELECTION_NPZ):
    """Exact angular completeness at ``nside``, shot-noise-free and INDEPENDENT of the
    survey randoms.

    The shipped LSS randoms are only a Monte-Carlo sampling of the angular selection
    ``S = completeness × Π(1−veto)`` (mangle maps), so a random-COUNT completeness is
    shot-noise-limited (split-half cover corr 0.89 @256 → 0.06 @1024 — which is why a
    random-based fill over-fills at high nside). This loads the cached high-res
    (nside=2048) selection rasterised straight from the BOSS mangle completeness +
    veto masks (``pipeline/boss_selection.py``) and averages it down to ``nside``
    (``ud_grade`` → the exact *fractional* completeness). Returns a ``(npix,)`` array
    in [0,1], or ``None`` if the cache is absent."""
    import os
    import healpy as hp
    if not os.path.exists(selection_npz):
        return None
    d = np.load(selection_npz)
    n_hi = int(d["nside"])
    m = np.zeros(12 * n_hi ** 2, np.float64)
    m[d["ipix"]] = d["sel"]
    if nside == n_hi:
        return m.astype(np.float32)
    return hp.ud_grade(m, nside_out=nside, power=0).astype(np.float32)


@dataclass
class FillFootprint:
    nside: int
    target_mask: np.ndarray        # (npix,) bool: intended-complete footprint
    observed_cover: np.ndarray     # (npix,) [0,1]: survey-random completeness
    fill_weight: np.ndarray        # (npix,) [0,1]: where/how-much to inpaint
    hole_pix: np.ndarray           # int[]: interior data-surrounded holes (Regime D)
    empty_pix: np.ndarray          # int[]: remaining fill pixels (Regime P)
    z_grid: np.ndarray             # n(z) bin centres
    nz: np.ndarray                 # n(z) values (galaxies per bin, normalised to mean 1)
    counts: np.ndarray = None      # (npix,) raw random counts (for the analog filler)
    holes: list = None             # List[inpaint.Hole]: interior holes (Regime D)
    empty_area: float = 0.0        # true empty area in pixels (Σ(1-cover) over hole neigh,
                                   #   rim fractions incl) — the mass to inpaint w/o double-count

    @property
    def fill_area_deg2(self) -> float:
        import healpy as hp
        return float(self.fill_weight.sum() * hp.nside2pixarea(self.nside, degrees=True))

    def pix(self, ra, dec):
        import healpy as hp
        # float64: healpy's ang2pix ufunc rejects float128 (e.g. pymangle.genrand output)
        return hp.ang2pix(self.nside, np.radians(90.0 - np.asarray(dec, np.float64)),
                          np.radians(np.asarray(ra, np.float64) % 360.0))

    def nbar_z(self, z):
        """Mean n(z) profile (normalised to mean 1) interpolated at ``z``."""
        return np.interp(np.asarray(z), self.z_grid, self.nz, left=0.0, right=0.0)


def _dilate(mask_bool, nside, n_iter):
    """Grow a boolean HEALPix mask by ``n_iter`` neighbour rings."""
    import healpy as hp
    m = np.asarray(mask_bool, bool).copy()
    if n_iter <= 0:
        return m
    allpix = np.arange(len(m))
    nb = hp.get_all_neighbours(nside, allpix)          # (8, npix)
    for _ in range(int(n_iter)):
        grown = m.copy()
        hit = np.zeros(len(m), bool)
        for k in range(8):
            valid = nb[k] >= 0
            hit[valid] |= m[nb[k][valid]]
        m = grown | hit
    return m


def _close(mask_bool, nside, n_iter):
    """Morphological closing (dilate then erode) — fills enclosed holes/gaps up to
    ``n_iter`` rings without extending the outer boundary."""
    import healpy as hp
    d = _dilate(mask_bool, nside, n_iter)
    # erode = complement of dilation of the complement
    e = ~_dilate(~d, nside, n_iter)
    return e


def _connected_components(pix_bool, nside):
    """Connected components of a boolean HEALPix mask (8-neighbour BFS)."""
    import healpy as hp
    pix = np.flatnonzero(pix_bool)
    pset = set(int(p) for p in pix)
    seen = set(); comps = []
    for p0 in pix:
        p0 = int(p0)
        if p0 in seen:
            continue
        comp = [p0]; seen.add(p0); stack = [p0]
        while stack:
            q = stack.pop()
            for nn in hp.get_all_neighbours(nside, q):
                nn = int(nn)
                if nn >= 0 and nn in pset and nn not in seen:
                    seen.add(nn); comp.append(nn); stack.append(nn)
        comps.append(np.array(comp))
    return comps


def _fill_interior_holes(mask_bool, nside, *, margin=24):
    """Fill interior holes of a HEALPix mask, leaving only its outer boundary.

    Vectorised flood-fill of the EXTERIOR through empty pixels, restricted to a local
    band (the mask dilated by ``margin`` rings) so the cost scales with the survey
    patch, not the whole sphere. Empty pixels inside the band that the exterior flood
    cannot reach are enclosed **interior holes** (any size up to ~``margin`` rings) and
    are merged into the mask — the topological, size-independent alternative to
    morphological closing. (A truly huge interior gap > ``margin`` rings is treated as
    a genuinely excluded region, honouring the integrity guard.)"""
    import healpy as hp
    m = np.asarray(mask_bool, bool)
    band = _dilate(m, nside, margin)
    empty = band & ~m
    if not empty.any():
        return m.copy()
    nb = hp.get_all_neighbours(nside, np.arange(len(m)))   # (8, npix)
    not_band = ~band
    # seed the exterior at empty pixels on the band's outer rim (adjacent to outside)
    exterior = np.zeros(len(m), bool)
    for k in range(8):
        v = nb[k] >= 0
        exterior[v] |= not_band[nb[k][v]]
    exterior &= empty
    # iteratively grow the exterior inward through connected empty pixels
    while True:
        grown = exterior.copy()
        for k in range(8):
            v = nb[k] >= 0
            grown[v] |= exterior[nb[k][v]]
        grown &= empty
        if int(grown.sum()) == int(exterior.sum()):
            break
        exterior = grown
    return m | (empty & ~exterior)                         # unreached empties = holes


def build_analytic_selmap(ra_data, dec_data, *, nside=256, lss_clip_deg=1.0,
                          selection_npz=DEFAULT_SELECTION_NPZ):
    """The LSS angular selection map from the EXACT pointing masks — random-independent.

    ``completeness × veto`` (:func:`load_analytic_completeness`) clipped to the LSS
    footprint = within ``lss_clip_deg`` of the real **galaxies** (not the survey
    randoms), which trims the mangle geometry's over-spill (geometry sectors excluded
    from the LSS sample) using only data. Exact, shot-noise-free at any ``nside``, and
    independent of the shipped LSS randoms. Returns ``None`` if the cache is absent.

    CAVEAT — NOT for clustering randoms. The completeness and interior veto structure
    are exact, but the LSS *clustering footprint boundary* (which sectors are in the
    sample — a tiling/chunk selection beyond the angular masks) is not, and w(θ) is
    extremely sensitive to it: a galaxy-proximity boundary over-covers by ~3–4% and
    inflates w(θ) by ~30–60% (the prior `make_mangle_randoms.py` boundary-fit limit,
    confirmed to persist with the vetos). Use this for the completeness/deficit (the
    contiguous fill) and field-interior conditioning, where the boundary is non-critical;
    for clustering normalisation the SURVEY randoms remain the gold standard."""
    ac = load_analytic_completeness(nside, selection_npz=selection_npz)
    if ac is None:
        return None
    near = _proximity_clip(ac > 0, np.asarray(ra_data), np.asarray(dec_data),
                           nside, lss_clip_deg)
    return (ac * near).astype(np.float32)


def _geometry_mask(nside, *, mangle_npy=None, mangle_ply=None):
    """Boolean HEALPix occupancy of the BOSS mangle GEOMETRY footprint, or None if
    no mangle source is available. Prefers the cached uniform-geometry randoms."""
    import healpy as hp
    ra = dec = None
    if mangle_npy and os.path.exists(mangle_npy):
        arr = np.load(mangle_npy)
        ra, dec = np.asarray(arr[:, 0], float), np.asarray(arr[:, 1], float)
    elif mangle_ply and os.path.exists(mangle_ply):
        try:
            import pymangle
            m = pymangle.Mangle(mangle_ply)
            ra, dec = m.genrand(3_000_000)
            w = m.weight(ra, dec); keep = w > 0
            # pymangle.genrand returns float128; healpy's ang2pix ufunc only supports float64
            ra, dec = np.asarray(ra[keep], np.float64), np.asarray(dec[keep], np.float64)
        except Exception:
            return None
    if ra is None:
        return None
    npix = 12 * nside ** 2
    pix = hp.ang2pix(nside, np.radians(90.0 - dec), np.radians(ra % 360.0))
    g = np.zeros(npix, bool); g[np.unique(pix)] = True
    return g


def _proximity_clip(geom_bool, ra_random, dec_random, nside, lss_clip_deg):
    """Keep geometry pixels whose centre is within ``lss_clip_deg`` of a survey
    random — drops the mangle over-spill and bounds the fill to a correlation
    scale of real data (the integrity guard)."""
    import healpy as hp
    from scipy.spatial import cKDTree
    from .geometry import _radec_to_nhat
    gpix = np.flatnonzero(geom_bool)
    if not len(gpix):
        return geom_bool
    vec = np.array(hp.pix2vec(nside, gpix)).T                  # (n,3) unit vectors
    rtree = cKDTree(_radec_to_nhat(np.asarray(ra_random), np.asarray(dec_random)))
    chord = 2.0 * np.sin(np.radians(lss_clip_deg) / 2.0)
    d, _ = rtree.query(vec, workers=-1)
    out = np.zeros_like(geom_bool); out[gpix[d <= chord]] = True
    return out


def build_fill_footprint(catalog=None, *, ra_random=None, dec_random=None, z_data=None,
                         nside=512, mangle_ply="data/boss/mask_DR12v5_CMASS_South.ply",
                         mangle_npy=DEFAULT_MANGLE_NPY, lss_clip_deg=1.0,
                         empty_thresh=0.2, min_fill_deg2=2.0, n_z=64,
                         contiguous=False, fill_deficit_thresh=0.05,
                         analytic_completeness=False) -> FillFootprint:
    """Build the :class:`FillFootprint` for a survey.

    Accepts either a loaded ``catalog`` (uses ``ra_random``/``dec_random``/``z_data``)
    or those arrays directly (for tests). ``lss_clip_deg`` bounds the fill to within
    that angular distance of real data (integrity guard); raise it to fill larger
    interior gaps, lower it to be more conservative.

    ``contiguous`` (the fully-completed product): take the survey's outer boundary
    (the mangle GEOMETRY, proximity-clipped to data), **fill every interior hole
    regardless of size** (:func:`_fill_interior_holes`), and set a **completeness-
    proportional** ``fill_weight = (1 − cover)`` over the whole footprint — so the
    striped partial-completeness regions (the tiling/veto pattern) AND empty holes are
    filled to **uniform survey density**, not just the zero-coverage pixels. The result
    is a gap-free, stripe-free field. ``fill_deficit_thresh`` skips near-full pixels.
    Trades a few-% 2-point penalty (fractional fill double-counts) for the uniform field
    topological / kNN / field-level statistics need; pair with randoms over ``target_mask``.
    """
    import healpy as hp
    if catalog is not None:
        ra_random = np.asarray(catalog.ra_random); dec_random = np.asarray(catalog.dec_random)
        z_data = np.asarray(catalog.z_data)
    ra_random = np.asarray(ra_random, float); dec_random = np.asarray(dec_random, float)
    npix = 12 * nside ** 2

    # observed coverage (0 in veto holes) + raw random counts (for hole finding)
    counts, observed_cover = fine_completeness_map(ra_random, dec_random, nside=nside)
    cover_bool = counts > 0
    if analytic_completeness:
        # use the EXACT mangle-based completeness (shot-noise-free) for the COMPLETENESS
        # value (→ the fill deficit), so the proportional fill targets the REAL veto
        # striping not Poisson noise (clean at high nside). The FOOTPRINT stays bounded by
        # the data (cover_bool = where randoms exist; proximity-clipped below) — the
        # integrity guard against filling the mangle geometry's over-spill beyond the
        # genuine survey edge. Veto holes (analytic=0) inside the data region still fill
        # (deficit=1, captured by the closing/target_mask).
        ac = load_analytic_completeness(nside)
        if ac is not None:
            observed_cover = np.where(cover_bool, ac, 0.0).astype(np.float32)

    # intended-complete footprint = morphological CLOSING of the observed coverage:
    # fills holes/gaps enclosed within ~lss_clip_deg WITHOUT extending the outer survey
    # boundary (the integrity guard — we never invent galaxies beyond the true edge or
    # more than ~lss_clip_deg from data). The mangle GEOMETRY mask, when available, only
    # TRIMS (∩): it can remove a closing-bridged gap that crosses outside the genuine
    # survey, but never extends the footprint.
    n_iter = max(1, int(round(lss_clip_deg / (hp.nside2resol(nside, arcmin=True) / 60.0))))
    target_mask = _close(cover_bool, nside, n_iter)
    geom = _geometry_mask(nside, mangle_npy=mangle_npy, mangle_ply=mangle_ply)
    if geom is not None:
        geom_near = _proximity_clip(geom, ra_random, dec_random, nside, lss_clip_deg)
        target_mask = target_mask & geom_near                  # trim only; never extend
    target_mask = target_mask | cover_bool                     # always keep covered pixels

    if contiguous:
        # FULLY-CONTIGUOUS footprint: base on the survey outer boundary (the geometry
        # near the data, else the closing) and fill EVERY interior hole, any size.
        base = geom_near if geom is not None else target_mask
        target_mask = _fill_interior_holes(base | cover_bool, nside)
        # COMPLETENESS-PROPORTIONAL fill: fill the DEFICIT (1 − cover) across the whole
        # footprint, not just zero-coverage pixels. The survey's angular completeness is
        # STRIPED (tiling / veto pattern: ~16% of CMASS-South has cover<0.8, 4%<0.5);
        # a binary cover==0 fill leaves those partial stripes under-dense. Filling the
        # deficit brings every pixel to uniform survey density → a gap-free, stripe-free
        # field. (Fractional fill double-counts at the few-% level for 2-pt — the
        # documented contiguity tradeoff; what topological / field-level stats need.)
        deficit = np.clip(1.0 - observed_cover, 0.0, 1.0)
        deficit[deficit < fill_deficit_thresh] = 0.0           # skip near-full pixels
        fill_weight = target_mask.astype(float) * deficit
    else:
        # MASKED product: fill ONLY genuine zero-coverage pixels (binary). Partial-
        # completeness rim pixels are NOT filled — at a real hole boundary they still
        # contain galaxies, so filling them would double-count and inject spurious
        # clustering power (verified: a fractional-rim fill worsens w(θ)/wp). Total density
        # stays continuous across the rim. ``empty_thresh`` = completeness below which a
        # pixel is a hole.
        fill_weight = (target_mask & (observed_cover <= empty_thresh * 1e-3)).astype(float)
        # SIZE GATE: only inpaint LARGE empty regions (>= min_fill_deg2). The 2-point
        # gate showed inpaint nets POSITIVE only for large regions and is counterproductive
        # for small veto holes (masked randoms cancel a small hole exactly). So small holes
        # are left masked, and only large empty regions are filled + flagged.
        if min_fill_deg2 and min_fill_deg2 > 0:
            pixarea = hp.nside2pixarea(nside, degrees=True)
            keep = np.zeros(npix, bool)
            for comp in _connected_components(fill_weight > 0, nside):
                if len(comp) * pixarea >= min_fill_deg2:
                    keep[comp] = True
            fill_weight = fill_weight * keep
    # true empty area (rim fractions included) over the hole neighbourhood — the total
    # galaxy mass the inpaint must place. Distributing this over the zero-coverage CORE
    # (fill_weight) conserves each hole's count without double-counting the rim.
    fill_neigh = _dilate(fill_weight > 0, nside, 2) & target_mask
    empty_area = float(np.clip(1.0 - observed_cover, 0.0, 1.0)[fill_neigh].sum())

    # split the fill region: small data-surrounded interior holes (Regime D) vs the
    # rest (Regime P — larger gaps / edges). find_interior_holes works on the counts.
    holes = find_interior_holes(counts, nside, empty_count=0.0, min_neighbour_frac=0.75)
    hole_pix = np.unique(np.concatenate([h.pixels for h in holes])) if holes else np.empty(0, int)
    fill_pix = np.flatnonzero(fill_weight > 0)
    empty_pix = np.setdiff1d(fill_pix, hole_pix, assume_unique=False)

    # n(z) profile (normalised to mean 1) for the radial draw of inpainted galaxies
    z = np.asarray(z_data, float)
    z_edges = np.linspace(z.min(), z.max(), n_z + 1)
    nz, _ = np.histogram(z, bins=z_edges)
    z_grid = 0.5 * (z_edges[1:] + z_edges[:-1])
    nz = nz.astype(float); nz /= max(nz.mean(), 1e-12)

    return FillFootprint(nside=nside, target_mask=target_mask, observed_cover=observed_cover,
                         fill_weight=fill_weight, hole_pix=hole_pix, empty_pix=empty_pix,
                         z_grid=z_grid, nz=nz, counts=counts, holes=holes, empty_area=empty_area)
