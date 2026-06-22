"""True-3D ECHOES completion of the local neighbourhood (P2/P3, branch data/local-neighborhood).

The survey sees nothing in the Zone of Avoidance (the Galactic plane) and progressively less
at large distance (the flux limit). But the Manticore reconstruction infers the actual 3D
density field *there* from the surrounding data — so we complete the catalog by Poisson-sampling
galaxies in the unobserved volume from the field-predicted intensity

    λ(x) = n̄(d) · (1+δ_Manticore(x)) · V_voxel ,

where ``n̄(d)`` is the selection-corrected radial number density of the observed galaxies. Each
Manticore posterior realization yields one completion; the ensemble is the true-3D ECHOES product
— the reconstruction's posterior ensemble of the FIELD composed with the completion of the
CATALOG. Completed (inpaint) galaxies carry PROV=5 and a per-galaxy distance; their K-band
magnitudes are transplanted from observed galaxies at matching distance (so the flux-limited
luminosity distribution is respected).
"""
from __future__ import annotations

import numpy as np

PROV_INPAINT = 5
RA_NGP, DEC_NGP = 192.85948, 27.12825          # J2000 North Galactic Pole


def galactic_b(ra, dec):
    """Galactic latitude b [deg] from equatorial (RA, Dec) [deg], vectorised (no astropy)."""
    r = np.radians(np.asarray(ra, float)); d = np.radians(np.asarray(dec, float))
    rn = np.radians(RA_NGP); dn = np.radians(DEC_NGP)
    sinb = np.sin(d) * np.sin(dn) + np.cos(d) * np.cos(dn) * np.cos(r - rn)
    return np.degrees(np.arcsin(np.clip(sinb, -1.0, 1.0)))


def radial_nbar(dist_mpc, f_sky, edges):
    """All-sky mean comoving number density per distance shell from the observed galaxies.

    ``n̄(d) = N_obs(shell) / (V_shell · f_sky)`` — dividing the observed counts by the observed
    sky fraction gives the density a complete survey would see, declining with distance as the
    flux limit bites. Returns ``(d_centres, nbar)`` [Mpc, Mpc^-3]."""
    edges = np.asarray(edges, float)
    N, _ = np.histogram(np.asarray(dist_mpc, float), edges)
    vshell = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3) * max(f_sky, 1e-6)
    return 0.5 * (edges[1:] + edges[:-1]), N / np.maximum(vshell, 1e-9)


def _ztransplant_kmag(d_new, donor_d, donor_k, rng, K=50):
    """K-band magnitude for each new galaxy from a real observed galaxy at MATCHING distance
    (so the flux-limited luminosity distribution is preserved)."""
    if donor_k is None or not np.isfinite(donor_k).any():
        return np.full(len(d_new), np.nan, np.float32)
    ok = np.isfinite(donor_k)
    dz = np.asarray(donor_d)[ok]; dk = np.asarray(donor_k)[ok]
    order = np.argsort(dz); ds = dz[order]; ks = dk[order]
    nd = len(ds); K = int(min(K, nd))
    pos = np.searchsorted(ds, np.asarray(d_new, float))
    lo = np.clip(pos - K // 2, 0, max(nd - K, 0))
    return ks[np.clip(lo + rng.integers(0, K, size=len(d_new)), 0, nd - 1)].astype(np.float32)


def _calibrate_bias(opd_vox, target, lo=1e-3, hi=2.0, n_iter=40):
    """Find b so galaxies sampled ∝ (1+δ)^b have density-weighted mean (1+δ) = ``target``
    (i.e. the fill matches the observed galaxy–field relation rather than the mass field).

    Uses the SAME weighting as the sampler (``v**b`` with ``v = max(1+δ, 0)``; voids carry no
    weight), and bisects on b ∈ [~0, 2]. Returns ``hi`` if the target is unreachable."""
    v = np.clip(np.asarray(opd_vox, float), 0.0, None)
    def mean_at(b):
        w = v ** b
        s = w.sum()
        return float((v * w).sum() / s) if s > 0 else 1.0
    if mean_at(hi) < target:            # field can't reach the target → use the densest weighting
        return hi
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if mean_at(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def complete_local_zoa(cat, field, *, zoa_deg=5.0, dmin=5.0, dmax=300.0, n_dbin=40,
                       bias=None, seed=0, H0=68.1):
    """Complete the Zone of Avoidance (and any |b|<``zoa_deg`` gap) of ``cat`` in true 3D by
    Poisson-sampling the Manticore field ``field`` (a GriddedFieldContext, equatorial frame).

    The intensity is mass-conserving per distance shell — ``λ ∝ n̄(d) · (1+δ)^b / ⟨(1+δ)^b⟩_shell``
    — so the fill reaches the all-sky mean density at each distance, modulated by the reconstructed
    structure. ``bias`` (the galaxy-bias exponent) defaults to an auto-calibration that makes the
    filled galaxies trace the field with the SAME mean over-density as the observed galaxies (a
    faithful completion, not the over-concentrated mass field). Returns a dict of the NEW (PROV=5)
    galaxies: ``ra, dec, dist_mpc, cz, ksmag, prov``."""
    rng = np.random.default_rng(seed)
    f_sky = float((cat.sel_map > 0).mean())
    edges = np.linspace(dmin, dmax, n_dbin + 1)
    dctr, nbar = radial_nbar(cat.dist_mpc, f_sky, edges)

    N = field.nvox; L = field.box_mpc; vox = L / N
    ax = ((np.arange(N) + 0.5) / N * L - L / 2).astype(np.float32)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    d = np.sqrt(X * X + Y * Y + Z * Z)
    shell = (d >= dmin) & (d <= dmax)
    xs, ys, zs, ds = X[shell], Y[shell], Z[shell], d[shell]
    opd_raw = np.clip(field.delta[shell].astype(np.float64), 0.0, None)   # 1+δ
    ra = np.degrees(np.arctan2(ys, xs)) % 360.0
    dec = np.degrees(np.arcsin(np.clip(zs / ds, -1.0, 1.0)))
    zoa = np.abs(galactic_b(ra, dec)) < zoa_deg          # the unobserved Galactic-plane voxels

    o = opd_raw[zoa]; dz = ds[zoa]
    if bias is None:                                     # match the observed galaxy–field relation
        in_range = cat.dist_mpc <= dmax
        target = float(field.overdensity_at(cat.xyz_data[in_range]).mean())
        bias = _calibrate_bias(o, target)
    mod = o ** bias
    # normalise the modulation to mean 1 within each distance shell (mass conservation)
    ibin = np.clip(np.searchsorted(edges, dz) - 1, 0, n_dbin - 1)
    bsum = np.bincount(ibin, weights=mod, minlength=n_dbin)
    bcnt = np.bincount(ibin, minlength=n_dbin)
    mod_norm = mod / np.maximum((bsum / np.maximum(bcnt, 1))[ibin], 1e-30)
    lam = np.interp(dz, dctr, nbar) * mod_norm * vox ** 3
    counts = rng.poisson(np.clip(lam, 0.0, None))
    occ = counts > 0
    if not occ.any():
        z0 = np.zeros(0, np.float32)
        return {"ra": z0, "dec": z0, "dist_mpc": z0, "cz": z0, "ksmag": z0,
                "prov": np.zeros(0, np.int8)}

    # expand occupied voxels to galaxies, jittered within the voxel
    src = np.repeat(np.where(occ)[0], counts[occ])
    zx, zy, zz = xs[zoa][src], ys[zoa][src], zs[zoa][src]
    n = len(src)
    jit = (rng.random((n, 3)) - 0.5) * vox
    px, py, pz = zx + jit[:, 0], zy + jit[:, 1], zz + jit[:, 2]
    dist = np.sqrt(px * px + py * py + pz * pz).astype(np.float32)
    nra = (np.degrees(np.arctan2(py, px)) % 360.0).astype(np.float32)
    ndec = np.degrees(np.arcsin(np.clip(pz / dist, -1.0, 1.0))).astype(np.float32)
    # cz = H0·d + reconstructed radial peculiar velocity
    nhat = np.column_stack([px, py, pz]) / dist[:, None]
    vr = np.einsum("ij,ij->i", field.velocity_at(np.column_stack([px, py, pz])), nhat)
    cz = (H0 * dist + vr).astype(np.float32)
    ksmag = _ztransplant_kmag(dist, cat.dist_mpc, getattr(cat, "ksmag_data", None), rng)
    return {"ra": nra, "dec": ndec, "dist_mpc": dist, "cz": cz, "ksmag": ksmag,
            "prov": np.full(n, PROV_INPAINT, np.int8)}


def complete_local_ensemble(cat, mcmc_list, *, manticore_dir=None, **kw):
    """One ZoA completion per Manticore realization → the true-3D posterior ensemble.

    Returns a list of completion dicts (one per ``mcmc`` in ``mcmc_list``); pair each with the
    observed catalogue. The seed is tied to the realization for reproducibility."""
    from .surveys.manticore import manticore_field_context
    out = []
    for m in mcmc_list:
        fc = (manticore_field_context(m, manticore_dir) if manticore_dir
              else manticore_field_context(m))
        out.append(complete_local_zoa(cat, fc, seed=1000 + m, **kw))
    return out
