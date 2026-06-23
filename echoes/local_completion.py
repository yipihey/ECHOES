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
    (i.e. the fill matches the observed galaxy–field MEAN relation). NOTE: a single power-law
    exponent matches only the mean and (for sub-linear b) SMOOTHS the contrast — the
    log-density bias below is the contrast-preserving replacement.

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


def _rank_gaussianize(o, ref, max_ref=200000, seed=0):
    """Map field values ``o`` to a standard normal via the empirical CDF of ``ref`` — the
    lognormal model's latent Gaussian field ``s = log(1+δ)`` realised by rank (exact regardless
    of the field's shape; the reconstructed Manticore density is only approximately lognormal)."""
    from scipy.special import ndtri
    ref = np.asarray(ref, float)
    if len(ref) > max_ref:
        ref = ref[np.random.default_rng(seed).choice(len(ref), max_ref, replace=False)]
    rs = np.sort(ref)
    u = (np.searchsorted(rs, np.asarray(o, float), side="right") + 0.5) / (len(rs) + 1.0)
    return ndtri(np.clip(u, 1e-6, 1.0 - 1e-6))


def observed_cic_transform(cat, field, dmax, *, kind="lognormal", d_complete=100.0, zoa_deg=5.0):
    """Fit a :class:`~echoes.density_transform.DensityTransform` to the OBSERVED galaxy
    counts-in-cells at the Manticore voxel scale (the nearby complete volume) — the lognormal
    (or empirical) galaxy-density PDF whose **variance and skew** the fill must reproduce. The
    fit is shot-noise-free (factorial moments via ``field_moments_from_counts``)."""
    import healpy as hp
    from .density_transform import fit_density_transform
    N = field.nvox; L = field.box_mpc
    m = cat.dist_mpc < d_complete
    r = np.radians(cat.ra_data[m]); dd = np.radians(cat.dec_data[m]); cd = np.cos(dd)
    dist = cat.dist_mpc[m].astype(float)
    xyz = dist[:, None] * np.column_stack([cd * np.cos(r), cd * np.sin(r), np.sin(dd)])
    idx = np.floor((xyz + L / 2) / L * N).astype(int)
    idx = idx[((idx >= 0) & (idx < N)).all(1)]
    Ncnt = np.bincount((idx[:, 0] * N + idx[:, 1]) * N + idx[:, 2], minlength=N ** 3)
    ax = ((np.arange(N) + 0.5) / N * L - L / 2).astype(np.float32)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    dg = np.sqrt(X * X + Y * Y + Z * Z)
    inb = (dg > 5.0) & (dg < d_complete)
    ra = np.degrees(np.arctan2(Y, X)) % 360.0
    dec = np.degrees(np.arcsin(np.clip(Z / np.maximum(dg, 1e-6), -1.0, 1.0)))
    obs = (inb & (np.abs(galactic_b(ra, dec)) >= zoa_deg)
           & (cat.sel_map[hp.ang2pix(cat.nside, np.radians(90 - dec), np.radians(ra))] > 0))
    counts = Ncnt.reshape(N, N, N)[obs].astype(float)
    return fit_density_transform(counts / max(counts.mean(), 1e-9), kind=kind, counts=counts)


def _modulation(o, ref, intensity, field, cat, dmax, bias, T=None):
    """Field modulation for the Cox intensity. ``intensity='transform'`` (log-Gaussian, the
    user's request): rank-gaussianise the field, then map through the transform ``T`` fit to the
    OBSERVED galaxy CiC PDF — so the fill reproduces the observed variance/skew (sharp
    voids/peaks). ``'bias'``: the legacy mean-matched power-law ``(1+δ)^b`` (smoother)."""
    if intensity == "transform" and T is not None:
        return np.clip(T.T(_rank_gaussianize(o, ref)), 0.0, None)
    if bias is None:
        target = float(field.overdensity_at(cat.xyz_data[cat.dist_mpc <= dmax]).mean())
        bias = _calibrate_bias(o, target)
    return np.clip(o, 0.0, None) ** bias


def complete_local_zoa(cat, field, *, zoa_deg=5.0, dmin=5.0, dmax=300.0, n_dbin=40,
                       bias=None, intensity="transform", seed=0, H0=68.1):
    """Complete the Zone of Avoidance (and any |b|<``zoa_deg`` gap) of ``cat`` in true 3D by
    Poisson-sampling the Manticore field ``field`` (a GriddedFieldContext, equatorial frame).

    The intensity is mass-conserving per distance shell — ``λ ∝ n̄(d) · mod / ⟨mod⟩_shell`` — so the
    fill reaches the all-sky mean density at each distance, modulated by the reconstructed structure.
    ``intensity='transform'`` (default) uses the contrast-preserving **log-density bias** so the
    fill reproduces the observed galaxy–field PDF (sharp voids/peaks); ``'bias'`` uses the legacy
    mean-matched power-law (smoother). Returns a dict of the NEW (PROV=5) galaxies."""
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
    T = observed_cic_transform(cat, field, dmax) if intensity == "transform" else None
    mod = _modulation(o, opd_raw, intensity, field, cat, dmax, bias, T)
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


def absolute_mag(kmag, dist_mpc, H0=68.1, kcorr=True):
    """K-band absolute magnitude: ``M = K − 5log10(d/Mpc) − 25 − k(z)``. The K-band
    k-correction at z<0.1 is small; ``k(z) ≈ −6·log10(1+z)`` (Kochanek+ 2001 form)."""
    d = np.asarray(dist_mpc, float)
    M = np.asarray(kmag, float) - 5.0 * np.log10(np.maximum(d, 1e-3)) - 25.0
    if kcorr:
        z = np.clip(H0 * d / 299792.458, 0.0, 0.3)
        M += 6.0 * np.log10(1.0 + z)        # subtract k(z)=-6log(1+z)  ->  M -= k  ->  M += 6log(1+z)
    return M


def estimate_lf(cat, m_faint, k_lim, f_sky, H0=68.1, m_bright=-25.5):
    """Data-driven K-band luminosity function from the OBSERVED galaxies (no Schechter fit).

    Returns ``(nbar0, sorted_absmag, d_complete)``: the comoving number density of galaxies in
    ``[m_bright, m_faint]`` (estimated in the nearby volume where ``m_faint`` is fully
    observable), and the sorted absolute magnitudes to draw from. ``m_bright`` clips the
    spurious super-bright tail (distance-error outliers brighter than any real galaxy)."""
    M = absolute_mag(cat.ksmag_data, cat.dist_mpc, H0=H0)
    d_complete = 10.0 ** ((k_lim - m_faint - 25.0) / 5.0)        # m_faint observable within this d
    near = (cat.dist_mpc < d_complete) & np.isfinite(M) & (M < m_faint) & (M > m_bright)
    vol = 4.0 / 3.0 * np.pi * d_complete ** 3 * max(f_sky, 1e-6)
    nbar0 = near.sum() / max(vol, 1e-9)
    return nbar0, np.sort(M[near]), d_complete


def complete_local(cat, field, *, m_faint=-21.0, k_lim=11.5, zoa_deg=5.0, dmin=5.0, dmax=300.0,
                   n_dbin=40, bias=None, intensity="transform", uncert_fields=None, seed=0, H0=68.1):
    """Full true-3D completion: fills the **Zone of Avoidance** AND restores the **faint
    galaxies below the flux limit everywhere**, to a uniform volume-limited density to
    ``m_faint`` modulated by the Manticore field.

    Per voxel the missing density is ``n̄0 · (1 − f_obs(d)·observed_sky) · (1+δ)^b_norm``, where
    ``f_obs(d)`` is the fraction of the LF brighter than the flux limit at distance d, ``n̄0`` the
    volume-limited density to ``m_faint`` (data-driven LF), and ``observed_sky`` is 0 in the ZoA.
    Restored galaxies draw absolute mags from the LF fainter than the local limit and carry an
    apparent ``K = M + DM(d)``, a ``cz``, a PROV=5, a ``kind`` (zoa/faint), and a per-galaxy
    ``uncert`` (see :func:`completion_uncert`)."""
    import healpy as hp
    rng = np.random.default_rng(seed)
    f_sky = float((cat.sel_map > 0).mean())
    edges = np.linspace(dmin, dmax, n_dbin + 1)
    nbar0, lf_M, _ = estimate_lf(cat, m_faint, k_lim, f_sky, H0=H0)

    N = field.nvox; L = field.box_mpc; vox = L / N
    ax = ((np.arange(N) + 0.5) / N * L - L / 2).astype(np.float32)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    d = np.sqrt(X * X + Y * Y + Z * Z)
    shell = (d >= dmin) & (d <= dmax)
    xs, ys, zs, ds = X[shell], Y[shell], Z[shell], d[shell]
    o = np.clip(field.delta[shell].astype(np.float64), 0.0, None)
    ra = np.degrees(np.arctan2(ys, xs)) % 360.0
    dec = np.degrees(np.arcsin(np.clip(zs / ds, -1.0, 1.0)))
    bgal = galactic_b(ra, dec)
    in_zoa = np.abs(bgal) < zoa_deg
    obs_sky = (cat.sel_map[hp.ang2pix(cat.nside, np.radians(90 - dec), np.radians(ra))] > 0)

    # fraction observed at distance d (brighter than the flux limit), and the missing fraction
    m_lim = k_lim - 5.0 * np.log10(np.maximum(ds, 1e-3)) - 25.0
    f_obs = np.searchsorted(lf_M, m_lim, side="right") / max(len(lf_M), 1)   # frac of LF brighter
    missing = 1.0 - np.clip(f_obs, 0.0, 1.0) * (obs_sky & ~in_zoa)

    # field modulation: contrast-preserving log-Gaussian transform (or legacy power-law), mean-1/shell
    T = observed_cic_transform(cat, field, dmax) if intensity == "transform" else None
    mod = _modulation(o, o, intensity, field, cat, dmax, bias, T)
    ibin = np.clip(np.searchsorted(edges, ds) - 1, 0, n_dbin - 1)
    bmean = (np.bincount(ibin, weights=mod, minlength=n_dbin)
             / np.maximum(np.bincount(ibin, minlength=n_dbin), 1))[ibin]
    lam = nbar0 * missing * (mod / np.maximum(bmean, 1e-30)) * vox ** 3
    counts = rng.poisson(np.clip(lam, 0.0, None))
    occ = counts > 0
    if not occ.any():
        return _empty_completion()

    src = np.repeat(np.where(occ)[0], counts[occ])
    n = len(src)
    jit = (rng.random((n, 3)) - 0.5) * vox
    px, py, pz = xs[src] + jit[:, 0], ys[src] + jit[:, 1], zs[src] + jit[:, 2]
    dist = np.sqrt(px * px + py * py + pz * pz).astype(np.float32)
    nra = (np.degrees(np.arctan2(py, px)) % 360.0).astype(np.float32)
    ndec = np.degrees(np.arcsin(np.clip(pz / dist, -1.0, 1.0))).astype(np.float32)
    nhat = np.column_stack([px, py, pz]) / dist[:, None]
    vr = np.einsum("ij,ij->i", field.velocity_at(np.column_stack([px, py, pz])), nhat)
    cz = (H0 * dist + vr).astype(np.float32)
    is_zoa_gal = in_zoa[src] | ~obs_sky[src]

    # absolute mags: draw from the LF fainter than the local flux limit (the whole LF in the ZoA)
    m_lim_gal = np.where(is_zoa_gal, -np.inf, k_lim - 5.0 * np.log10(dist) - 25.0)
    start = np.searchsorted(lf_M, m_lim_gal)
    pick = np.clip(start + (rng.random(n) * np.maximum(len(lf_M) - start, 1)).astype(int),
                   0, len(lf_M) - 1)
    absmag = lf_M[pick].astype(np.float32)
    ksmag = (absmag + 5.0 * np.log10(dist) + 25.0).astype(np.float32)

    xyz = np.column_stack([px, py, pz]).astype(np.float32)
    unc = completion_uncert(xyz, dist, is_zoa_gal, uncert_fields, dmax=dmax)
    return {"ra": nra, "dec": ndec, "dist_mpc": dist, "cz": cz, "ksmag": ksmag,
            "absmag": absmag, "prov": np.full(n, PROV_INPAINT, np.int8),
            "kind": np.where(is_zoa_gal, "zoa", "faint"), "uncert": unc}


def _empty_completion():
    z = np.zeros(0, np.float32)
    return {"ra": z, "dec": z, "dist_mpc": z, "cz": z, "ksmag": z, "absmag": z,
            "prov": np.zeros(0, np.int8), "kind": np.zeros(0, "<U5"), "uncert": z}


def completion_uncert(xyz, dist, is_zoa, uncert_fields=None, dmax=300.0, r_reliable=200.0):
    """Per-galaxy completion uncertainty in [0,1].

    Principled measure when ``uncert_fields`` (a list of GriddedFieldContext posterior
    realizations) is given: the normalised ensemble scatter of ``1+δ`` at the galaxy position
    (where the realizations disagree, the completion is uncertain). Fallback: a distance
    heuristic (reconstruction degrades beyond ``r_reliable``) plus a ZoA penalty."""
    dist = np.asarray(dist, float)
    if uncert_fields:
        vals = np.array([f.overdensity_at(xyz) for f in uncert_fields])   # (n_real, N)
        scatter = vals.std(0) / np.maximum(vals.mean(0), 0.3)
        return np.clip(scatter, 0.0, 1.0).astype(np.float32)
    u = np.clip(dist / r_reliable, 0.0, 1.0)
    return np.clip(u + 0.2 * np.asarray(is_zoa, float), 0.0, 1.0).astype(np.float32)


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
