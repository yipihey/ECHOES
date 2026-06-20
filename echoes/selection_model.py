"""Generative selection model — what observational effects drop from the field.

This module writes down, explicitly, the statistical model for how the true
galaxy field becomes the observed spectroscopic catalog, so that the missing
galaxies can be *sampled from their proper conditional posterior* (given the
reconstructed field) rather than pasted in with a local-density weight. It is the
conceptual core of the field-level ECHOES completion: the same model (a) is the
forward simulator that the mocks sample from, and (b) supplies the likelihood
terms that couple the latent field to the data and the intensity of the dropped
galaxies that the completion resamples.

The forward chain (each step a probability term):

  true field δ(x)
    → galaxies: inhomogeneous Poisson, intensity λ(x) = n̄(x)·(1+δ(x))
    → IMAGING DETECTION: kept w.p. ``p_img(n̂)`` = mask × imaging-systematic
        modulation.  BOSS ``WEIGHT_SYSTOT`` is the inverse relative detection:
        ``p_img(n̂) ∝ 1/w_systot(n̂)`` (regions with w_systot>1 are under-detected).
    → SPECTROSCOPIC TARGETING + FIBER ASSIGNMENT: a targeted galaxy loses its
        fiber to a FIBER COLLISION with a probability that depends on its
        NEIGHBOURS — within an angular pair closer than ``collision_scale`` the
        tiling can place at most one fiber.  This is the density-coupled term:
        ``p_coll(x) ≈ 1 - (1 - f_coll)^{m(x)}`` grows with the local close-pair
        multiplicity ``m(x)``, itself ∝ (1+δ).  (PIP/bitwise weighting,
        Bianchi & Percival 2017, is the pair-level inverse of exactly this.)
    → REDSHIFT MEASUREMENT: a fibered galaxy yields a good spec-z w.p.
        ``1 - p_fail`` (faint / locally-dense biased); else a REDSHIFT FAILURE
        (known position, no redshift).
    → REDSHIFT ERROR: observed galaxies have precise spec-z; collisions,
        failures and imaging-only galaxies carry only a broad photo-z likelihood
        ``p(z_phot | z_true)`` — the term the field sharpens ("augment imaging").

The probability a true galaxy reaches the observed spec-z catalog is therefore

    p_obs(x) = p_img(n̂) · (1 - p_coll(x)) · (1 - p_fail(x)),

and the intensity of *dropped* galaxies the completion must restore is
``λ(x)·(1 - p_obs(x))`` — split into imaging-detected-but-missing (known n̂; the
collision/failure targets) and not-imaged (mask holes).  Conditioning the field
on the *thinned* observed counts and then sampling from ``λ(x)(1-p_obs)`` is the
self-consistent completion: the missingness is density-coupled, so reconstructing
it needs the field, and the field is informed by the thinned data.

The mock (:func:`echoes.mock_systematics.apply_survey_systematics`) samples from
this same model, so inject-and-recover is self-consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SelectionModel:
    """Parameters of the BOSS-like observational selection (see module docstring).

    Defaults match :func:`echoes.mock_systematics.apply_survey_systematics` so the
    mock is this model's own simulator.

    Attributes
    ----------
    collision_scale_deg : float
        Fiber-collision angular scale (62'' for BOSS).
    coll_frac : float
        Fraction of resolvable close pairs that actually lose a fiber.
    zfail_frac : float
        Mean redshift-failure probability among fibered galaxies.
    zfail_faint_bias : float
        Multiplicative tilt of the failure probability with i-band magnitude
        (>1 ⇒ fainter galaxies fail more).
    zfail_density_coupling : float
        Extra failure probability per unit local overdensity (sky crowding).
    photoz_sigma : float
        Gaussian photo-z scatter σ_z used for the imaging redshift likelihood.
    """

    collision_scale_deg: float = 62.0 / 3600.0
    coll_frac: float = 0.6
    zfail_frac: float = 0.014
    zfail_faint_bias: float = 1.0
    zfail_density_coupling: float = 0.0
    photoz_sigma: float = 0.03

    # ---- per-effect probabilities (the explicit likelihood terms) ----

    def p_img(self, w_systot: np.ndarray) -> np.ndarray:
        """Imaging detection probability relative to the mean, from WEIGHT_SYSTOT.

        ``WEIGHT_SYSTOT`` upweights under-detected regions, so the relative
        detection probability is ``min(1/w_systot, 1)`` (matching the mock's
        ``keep_sys = U < 1/w_systot`` thinning). Returns values in (0, 1].
        """
        return np.minimum(1.0 / np.clip(np.asarray(w_systot, float), 1e-3, None), 1.0)

    def p_collision(self, n_close: np.ndarray) -> np.ndarray:
        """Fiber-collision loss probability given the local close-pair count.

        Each close encounter loses a fiber with probability ``coll_frac``, and
        when it does, *this* galaxy (rather than its partner) is the one removed
        with probability ½ — so a single encounter removes this galaxy with
        probability ``q = coll_frac/2``. Over ``n_close`` neighbours, treated as
        independent encounters, ``p_collision = 1 - (1 - q)^{n_close}``. With
        ``n_close=0`` this is 0. This density-coupled term ties a galaxy's
        observability to the local field; it reproduces the mock's per-galaxy
        collision loss for the dominant low-multiplicity pairs (the
        independent-encounter form mildly over-removes at high multiplicity,
        where the exact pairwise removal saturates).
        """
        q = 0.5 * self.coll_frac
        n = np.asarray(n_close, float)
        return 1.0 - (1.0 - q) ** n

    def p_fail(self, imag: Optional[np.ndarray] = None,
               delta_local: Optional[np.ndarray] = None,
               n: Optional[int] = None) -> np.ndarray:
        """Redshift-failure probability (faint- and density-biased).

        ``p_fail = zfail_frac · bias^{(i - <i>)/σ_i} · (1 + density_coupling·δ)``,
        clipped to [0, 1]. With no photometry/field info returns the flat
        ``zfail_frac``.
        """
        if imag is None and delta_local is None:
            return np.full(int(n or 1), self.zfail_frac)
        p = np.full(len(imag) if imag is not None else len(delta_local), self.zfail_frac)
        if imag is not None and self.zfail_faint_bias != 1.0:
            imag = np.asarray(imag, float)
            p = p * self.zfail_faint_bias ** ((imag - np.median(imag)) / max(imag.std(), 1e-6))
        if delta_local is not None and self.zfail_density_coupling:
            p = p * (1.0 + self.zfail_density_coupling * np.asarray(delta_local, float))
        return np.clip(p, 0.0, 1.0)

    def p_observed(self, w_systot, n_close, imag=None, delta_local=None) -> np.ndarray:
        """Probability a true galaxy reaches the observed spec-z catalog:
        ``p_img · (1 - p_collision) · (1 - p_fail)`` (the thinning rate that
        couples the field to the observed counts)."""
        return (self.p_img(w_systot)
                * (1.0 - self.p_collision(n_close))
                * (1.0 - self.p_fail(imag=imag, delta_local=delta_local,
                                     n=len(np.atleast_1d(n_close)))))

    def photoz_loglike(self, z_grid: np.ndarray, z_phot: np.ndarray,
                       sigma: Optional[np.ndarray] = None) -> np.ndarray:
        """Photo-z log-likelihood ``log p(z_phot | z_true=z_grid)`` for each
        imaging galaxy, evaluated on a shared redshift grid.

        Gaussian by default (σ = ``photoz_sigma``); pass per-object ``sigma`` or a
        full posterior elsewhere. Returns ``(n_gal, n_z)``. This is the term the
        reconstructed field multiplies and sharpens — the "augment imaging by
        refining the redshift errors" likelihood.
        """
        z_phot = np.atleast_1d(np.asarray(z_phot, float))
        s = self.photoz_sigma if sigma is None else np.atleast_1d(np.asarray(sigma, float))
        s = np.broadcast_to(s, z_phot.shape)
        d = (z_grid[None, :] - z_phot[:, None]) / s[:, None]
        return -0.5 * d * d - np.log(s[:, None])

    # ---- field coupling (for inference + resampling) ----

    def observed_thinning(self, intensity, w_systot, n_close, imag=None, delta_local=None):
        """Expected OBSERVED spec-z intensity = true ``intensity`` × ``p_observed``.
        Drives the Poisson likelihood that conditions the field on the data."""
        return np.asarray(intensity, float) * self.p_observed(w_systot, n_close, imag, delta_local)

    def missing_intensity(self, intensity, w_systot, n_close, imag=None, delta_local=None):
        """Intensity of DROPPED galaxies = true ``intensity`` × ``(1 - p_observed)``
        — what the completion resamples from the conditional field posterior."""
        return np.asarray(intensity, float) * (1.0 - self.p_observed(w_systot, n_close, imag, delta_local))


def inpaint_intensity(fill_weight, nbar_z, opd, z):
    """Inpainting intensity λ(n̂,z) = ``fill_weight(n̂) · n̄(z) · (1+δ(n̂,z))`` — the
    angular×radial rate of galaxies to GENERATE in the un-observed footprint
    (veto holes + empty regions). ``fill_weight`` is the per-position WHERE-to-fill
    fraction (``echoes.fill_footprint``; the angular marginal of ``missing_intensity``
    with ``p_img→0`` in holes), ``nbar_z`` the mean n(z), and ``opd`` the conditional
    field draw ``1+δ`` from an engine. A global amplitude ``α_n`` is applied by the
    sampler so the expected count matches the selection-immune target density."""
    return (np.asarray(fill_weight, float) * np.asarray(nbar_z, float)
            * np.clip(np.asarray(opd, float), 0.0, None))


def local_close_pair_count(ra_deg, dec_deg, collision_scale_deg):
    """Number of neighbours within the collision scale for each galaxy — the
    ``n_close`` that drives :meth:`SelectionModel.p_collision`. Self excluded."""
    from scipy.spatial import cKDTree
    from .geometry import _radec_to_nhat
    nhat = _radec_to_nhat(np.asarray(ra_deg, float), np.asarray(dec_deg, float))
    chord = 2.0 * np.sin(np.radians(collision_scale_deg) / 2.0)
    tree = cKDTree(nhat)
    counts = tree.query_ball_point(nhat, chord, return_length=True)
    return np.asarray(counts, float) - 1.0          # drop self
