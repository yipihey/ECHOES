"""Quaia — a ``resample_redshift`` (+ footprint inpaint) dataset.

Quaia (Storey-Fisher et al. 2024) is an all-sky Gaia/unWISE quasar catalog: every
object has a precise sky position but a *photometric* redshift with a large,
per-object error (``redshift_quaia ± redshift_quaia_err``, σ_z ~ 0.04–0.08). So
the incompleteness here is not missing objects — it is the **redshift** of objects
we already see.

ECHOES completes Quaia in a new mode (see :func:`echoes.posterior.build_package`):

- **resample_redshift**: keep every object's position fixed and resample its
  redshift from a per-object posterior = the photo-z Gaussian × the local
  line-of-sight density prior. The density prior is seeded from the lowest-error
  "anchor" objects so it sharpens the broad photo-z onto real structure rather than
  re-injecting the photo-z scatter.
- **footprint inpaint**: the published selection function reports per-pixel
  completeness; where it is low, restore synthetic quasars (provenance ``selfunc``)
  to undo the angular undersampling. These carry no photo-z (field-only redshift).

Public files (download into ``data/quaia/`` from Zenodo DOI 10.5281/zenodo.8060755):
``quaia_G20.0.fits``, ``random_G20.0_10x.fits``, ``selection_function_NSIDE64_G20.0.fits``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..distance import DistanceCosmo, radec_z_to_cartesian
from .targets_generic import ResampleTargets

# Quaia fiducial cosmology (matches the graphGP-cosmology research loader)
_QUAIA_COSMO = DistanceCosmo(Om=0.31, h=0.68, w0=-1.0, wa=0.0)


@dataclass
class QuaiaCatalog:
    """Quaia quasar catalog satisfying the ECHOES ``SurveyCatalog`` contract.

    ``z_data`` is the photo-z point estimate (``redshift_quaia``) and ``z_data_err``
    its per-object 1σ width (``redshift_quaia_err``) — the quantity the
    ``resample_redshift`` mode draws from."""
    ra_data: np.ndarray
    dec_data: np.ndarray
    z_data: np.ndarray                 # redshift_quaia (photo-z point estimate)
    z_data_err: np.ndarray             # redshift_quaia_err (per-object photo-z 1σ)
    ra_random: np.ndarray
    dec_random: np.ndarray
    z_random: np.ndarray
    sel_map: np.ndarray                # HEALPix selection function, completeness in [0,1]
    nside: int
    fid_cosmo: DistanceCosmo

    @property
    def N_data(self) -> int:
        return len(self.ra_data)


def _galactic_b(ra_deg, dec_deg):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    sc = SkyCoord(ra=np.asarray(ra_deg) * u.deg, dec=np.asarray(dec_deg) * u.deg, frame="icrs")
    return sc.galactic.b.to(u.deg).value


def load_selection_function(path):
    """Read the published Quaia HEALPix selection-function FITS (NSIDE64, RING)."""
    from astropy.io import fits
    with fits.open(path) as hdul:
        hdr = hdul[1].header
        nside = int(hdr["NSIDE"])
        col = hdr.get("TTYPE1", "T")
        sel = np.asarray(hdul[1].data[col], np.float64).ravel()
    npix = 12 * nside ** 2
    if sel.size != npix:
        raise ValueError(f"selection map has {sel.size} pixels, expected NSIDE={nside} -> {npix}")
    return sel, nside


def load_quaia(catalog_path, random_path=None, *, selection_path=None, fid_cosmo=None,
               b_min=10.0, z_key="redshift_quaia", z_err_key="redshift_quaia_err",
               rng_seed=0):
    """Load the real Quaia FITS into a :class:`QuaiaCatalog`.

    Applies the standard Galactic-plane cut ``|b| >= b_min``. The selection
    function (``selection_path``) provides ``sel_map`` (and drives the footprint
    inpaint); the published randoms (``random_path``) provide the random angular
    catalog (redshifts sampled from the data n(z))."""
    from astropy.table import Table
    if fid_cosmo is None:
        fid_cosmo = _QUAIA_COSMO
    if selection_path is None:
        raise ValueError("Quaia loader requires selection_path (the NSIDE64 completeness map)")
    rng = np.random.default_rng(rng_seed)

    cat = Table.read(catalog_path)
    ra = np.asarray(cat["ra"], np.float64); dec = np.asarray(cat["dec"], np.float64)
    z = np.asarray(cat[z_key], np.float64); zerr = np.asarray(cat[z_err_key], np.float64)
    keep = (np.abs(_galactic_b(ra, dec)) >= b_min) & np.isfinite(z) & np.isfinite(zerr) & (zerr > 0)
    ra, dec, z, zerr = ra[keep], dec[keep], z[keep], zerr[keep]

    sel, nside = load_selection_function(selection_path)

    if random_path is not None:
        rnd = Table.read(random_path)
        ra_r = np.asarray(rnd["ra"], np.float64); dec_r = np.asarray(rnd["dec"], np.float64)
        rkeep = np.abs(_galactic_b(ra_r, dec_r)) >= b_min
        ra_r, dec_r = ra_r[rkeep], dec_r[rkeep]
        z_r = rng.choice(z, size=len(ra_r))                # sample randoms' z from data n(z)
    else:
        ra_r = dec_r = z_r = np.zeros(0)

    return QuaiaCatalog(
        ra_data=ra, dec_data=dec, z_data=z, z_data_err=zerr,
        ra_random=ra_r, dec_random=dec_r, z_random=z_r,
        sel_map=sel, nside=nside, fid_cosmo=fid_cosmo)


def build_quaia_targets(catalog, *, footprint_inpaint=True, sel_floor=0.20,
                        seed=0, jitter_arcmin=5.0):
    """Build the combined Quaia target set: every object (resample) + optional
    selection-function inpaint analogs.

    - **resample rows** (one per observed object): self-hosted, ``miss_kind="photoz"``,
      carrying ``photoz_mu = redshift_quaia`` and ``photoz_sigma = redshift_quaia_err``.
    - **inpaint rows** (``footprint_inpaint``): for each observed object in a pixel of
      completeness ``c = sel_map[pix]``, restore ``floor(max(1/c − 1, 0) + u)`` analog
      clones (``miss_kind="selfunc"``, no photo-z → field-only redshift) at a jitter of
      its position. ``sel_floor`` clips the completeness from below so a near-empty
      pixel does not spawn a runaway number of clones.

    Returns a :class:`ResampleTargets` consumed by ``build_package(mode=
    'resample_redshift')``; inpaint rows have NaN ``photoz_sigma`` so they get a
    flat (field-only) posterior and are excluded from the anchor pool.
    """
    import healpy as hp
    rng = np.random.default_rng(seed)
    ra = np.asarray(catalog.ra_data, np.float64)
    dec = np.asarray(catalog.dec_data, np.float64)
    mu = np.asarray(catalog.z_data, np.float64)
    sig = np.asarray(catalog.z_data_err, np.float64)
    n = len(ra)

    # resample rows: every object, self-hosted
    blocks = dict(ra=[ra], dec=[dec], mu=[mu], sig=[sig],
                  kind=[np.full(n, "photoz")], host=[np.arange(n)])

    if footprint_inpaint:
        pix = hp.ang2pix(catalog.nside, np.deg2rad(90.0 - dec), np.deg2rad(ra % 360.0))
        c = np.clip(catalog.sel_map[pix], sel_floor, 1.0)
        n_extra = np.floor(np.maximum(1.0 / c - 1.0, 0.0) + rng.random(n)).astype(int)
        host = np.repeat(np.arange(n), n_extra)
        m = len(host)
        s = jitter_arcmin / 60.0
        dra = rng.normal(0.0, s, m) / np.cos(np.radians(dec[host]))
        ddec = rng.normal(0.0, s, m)
        blocks["ra"].append(ra[host] + dra); blocks["dec"].append(dec[host] + ddec)
        blocks["mu"].append(np.full(m, np.nan)); blocks["sig"].append(np.full(m, np.nan))
        blocks["kind"].append(np.full(m, "selfunc")); blocks["host"].append(host)

    cat_ = lambda k: np.concatenate(blocks[k])
    return ResampleTargets(
        ra=cat_("ra"), dec=cat_("dec"),
        colors=np.empty((len(cat_("ra")), 0), np.float64), mags=None,
        miss_kind=cat_("kind"), host_index=cat_("host"),
        photoz_mu=cat_("mu"), photoz_sigma=cat_("sig"))


# ---- registry ----------------------------------------------------------------

from .registry import register, DatasetSpec          # noqa: E402

register(DatasetSpec(
    name="quaia",
    load_catalog=load_quaia,
    completion_mode="resample_redshift",
    build_targets=build_quaia_targets,                # combined resample + inpaint targets
    footprint_inpaint=True,
    photoz_features=None,                             # photo-z is the Gaussian N(z, z_err), not colours
    photoz_gaussian=True,
    field_pool="anchors",
    anchor_frac=0.30,
    fid_cosmo=_QUAIA_COSMO,
    nside=64,
    nq=33,                                            # near-Gaussian resample posteriors; 33 ⇒ ~half file
    product_prefix="quaia",
    default_engine="field",
    copula=False,                                     # all-resampled copula is a refinement (see docs)
    uniform_randoms=True,                             # selection function is imprinted on the randoms
))
