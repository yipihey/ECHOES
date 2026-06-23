"""Tier-A generative field engine (``z_mode='generative'``) — the purely
data-driven non-Gaussian completion.

A thin, additive engine layered on the calibrated fieldpost conditional posterior
(:mod:`echoes.fieldpost`): each missing-galaxy redshift is still drawn from the GP
field posterior along its sightline, but the per-sightline conditional field
``1+δ`` is pushed through a **measured monotonic transform** ``T`` (:mod:`echoes.
density_transform`) before it weights the redshift. ``T`` is fit from the data's own
counts-in-cells PDF, so the completion reproduces the data's one-point statistics
(skewed PDF, empty voids, high-density tail) and most of the kNN-CDF / non-Gaussian
structure the stationary GP cannot — while staying rank-preserving (calibration
intact) and 100% data-driven.

``transform='identity'`` makes this engine reproduce ``fieldpost`` byte-for-byte
(the hook in :func:`echoes.fieldpost._fieldpost_zmiss` is not even invoked) — the
Stage-1 skeleton whose parity is the gate before the transform is turned on.

Optional SP-flat reference: an :func:`echoes.sp_maps.isd_decontamination` weight is
carried on the model for the systematics-free infill (applied in the inpaint path;
the redshift-completion path is SP-neutral). The drop-in seam lives in
:func:`echoes.completion.complete_catalog_photoz` (``z_mode='generative'``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .density_transform import DensityTransform, fit_density_transform


@dataclass
class GenerativeModel:
    """Everything the generative engine needs, built once per catalog.

    Wraps a :class:`echoes.fieldpost.FieldContext` (the calibrated Gaussian
    conditional posterior + its K-draw cache) and a :class:`DensityTransform`."""
    field_ctx: object                            # FieldContext (carries n_samples + cache)
    transform: DensityTransform
    sigma_ref: float = 1.0                       # prior std for gaussianising the LOS field
    isd: object = None                           # ISDResult (SP-flat reference), optional
    sp_maps: object = None                       # SPMaps, optional
    cic_R: float = 8.0

    @property
    def n_samples(self) -> int:
        return int(getattr(self.field_ctx, "n_samples", 1))

    def los_transform(self):
        """Callable applied to the per-sightline ``1+δ`` field ``(M, n_z)``, or
        ``None`` for the identity transform (→ exact fieldpost parity).

        The conditional field reverts to ``1`` (neutral) in data-poor stretches with
        prior std ``sigma_ref``; gaussianising against ``(mu=1, sigma=sigma_ref)``
        and pushing through ``T`` reshapes the marginal to the data's one-point PDF
        where the field is prior-dominated, while leaving data-constrained sightlines
        (small deviation from the conditional mean) essentially fixed. Monotone in the
        field value ⇒ rank/PIT preserving."""
        dt = self.transform
        if dt.kind == "identity":
            return None
        sigma = float(self.sigma_ref)
        return lambda opd: dt.apply_to_field(np.asarray(opd, float), mu=1.0, sigma=sigma)


def _cic_overdensity(catalog, *, R=8.0, n_cells=8000, seed=3, randoms=None, return_counts=False):
    """Measure the data's counts-in-cells overdensity ``1+δ = N/⟨N⟩`` in R-spheres
    centred on footprint-uniform points — the transform target PDF.

    Cell centres come from (in order): an explicit ``randoms=(ra,dec,z)`` tuple →
    ``catalog.ra_random`` → the data galaxies themselves (last-resort fallback when
    no footprint tracer is available; mildly over-samples dense regions).
    ``return_counts`` also returns the raw cell counts (for shot-noise deconvolution)."""
    from scipy.spatial import cKDTree
    from .clustering import comoving_mpc_h

    def xyz(ra, dec, z):
        d = comoving_mpc_h(np.asarray(z)); r = np.radians(np.asarray(ra)); dd = np.radians(np.asarray(dec))
        return np.column_stack([d * np.cos(dd) * np.cos(r), d * np.cos(dd) * np.sin(r), d * np.sin(dd)])

    if randoms is not None:
        cra, cdec, cz = (np.asarray(a) for a in randoms)
    elif getattr(catalog, "ra_random", None) is not None and len(np.atleast_1d(catalog.ra_random)):
        cra, cdec, cz = (np.asarray(catalog.ra_random), np.asarray(catalog.dec_random),
                         np.asarray(catalog.z_random))
    else:                                                       # fallback: data as cells
        cra, cdec, cz = (np.asarray(catalog.ra_data), np.asarray(catalog.dec_data),
                         np.asarray(catalog.z_data))
    g = xyz(catalog.ra_data, catalog.dec_data, catalog.z_data)
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(cra), min(n_cells, len(cra)), replace=False)
    c = xyz(cra[sel], cdec[sel], cz[sel])
    n = cKDTree(g).query_ball_point(c, R, return_length=True).astype(float)
    opd = n / max(n.mean(), 1e-9)
    return (opd, n) if return_counts else opd


def build_generative_model(catalog, *, n_samples=1, seed=0, transform="identity",
                           transform_obj=None, cic_R=8.0, cic_randoms=None, deconv=True,
                           sp_reference=False, field_ctx=None, fieldpost_kwargs=None,
                           sp_kwargs=None, lognormal=False, verbose=False) -> GenerativeModel:
    """Assemble a :class:`GenerativeModel`.

    Parameters
    ----------
    transform : {'identity','empirical','lognormal'}
        ``'identity'`` ⇒ reproduces ``fieldpost`` (Stage-1 parity skeleton). Otherwise
        the data's CiC PDF (at ``cic_R``) is measured and ``T`` is fit to it.
    transform_obj : DensityTransform, optional
        Use a pre-fit transform instead of measuring one (e.g. a shot-noise-deconvolved
        target). Overrides ``transform``.
    sp_reference : bool
        If True, load the SP suite and derive the ISD decontamination weights (the
        SP-flat reference for the systematics-free infill). Carried on the model.
    field_ctx : FieldContext, optional
        Reuse a prebuilt context (amortise ξ→kernel across an ensemble).
    """
    from .fieldpost import build_field_context
    # log-Gaussian (lognormal) field: the SAMPLED field is 1+δ = exp(g) (log ρ Gaussian), realised by
    # the rank-preserving lognormal DensityTransform of the calibrated Gaussian conditional posterior.
    # (A *native* log conditioning is avoided — the delta-function observation y=1/n̄ exponentiates
    # catastrophically; a native log field would need a binned-count LGCP Laplace solve.)
    if lognormal and transform == "identity" and transform_obj is None:
        transform = "lognormal"
    if field_ctx is None:
        field_ctx = build_field_context(catalog, seed=seed, n_samples=n_samples,
                                        verbose=verbose, **(fieldpost_kwargs or {}))
    if transform_obj is not None:
        dt = transform_obj
    elif transform == "identity":
        dt = DensityTransform(kind="identity")
    else:
        opd_cells, counts = _cic_overdensity(catalog, R=cic_R, randoms=cic_randoms, return_counts=True)
        # shot-noise deconvolution (factorial moments) for the lognormal fit — removes
        # the Poisson broadening of the sparse CiC PDF so the re-sampled field matches
        # the data's true 1-pt variance (sub-1 gal/cell at R=8 is heavily broadened).
        dt = fit_density_transform(opd_cells, kind=transform, scale=cic_R,
                                   counts=(counts if deconv else None))
        if verbose:
            print(f"[generative] fit T={transform}{' (shot-noise-deconv)' if deconv else ''} "
                  f"from CiC(R={cic_R}): var={dt.var_opd:.3f} skew={dt.skew_opd:.3f}")
    # prior std of the conditional field (linearised LGCP variance ≈ K(0))
    sigma_ref = float(np.sqrt(max(float(np.asarray(field_ctx.cov[1])[0]), 1e-6)))
    isd = sp_maps = None
    if sp_reference:
        from .sp_maps import load_sp_maps, isd_decontamination
        sp_maps = load_sp_maps(verbose=verbose, **(sp_kwargs or {}))
        isd = isd_decontamination(catalog, sp_maps)
        if verbose:
            print(f"[generative] SP-flat ISD weights: range "
                  f"[{isd.weight.min():.3f}, {isd.weight.max():.3f}] clean={isd.clean}")
    return GenerativeModel(field_ctx=field_ctx, transform=dt, sigma_ref=sigma_ref,
                           isd=isd, sp_maps=sp_maps, cic_R=cic_R)


def _generative_zmiss(targets, photoz, dz_pool, gen_model, draw_index,
                      z_o, z_host, miss_kind, rng):
    """Missing-galaxy redshifts from the (transform-reshaped) field posterior.

    Delegates to :func:`echoes.fieldpost._fieldpost_zmiss` with the measured
    transform as the field hook — identical accounting / fallback / K-draw caching,
    so with ``transform='identity'`` it is byte-for-byte fieldpost."""
    from .fieldpost import _fieldpost_zmiss
    return _fieldpost_zmiss(targets, photoz, dz_pool, gen_model.field_ctx, draw_index,
                            z_o, z_host, miss_kind, rng, transform=gen_model.los_transform())
