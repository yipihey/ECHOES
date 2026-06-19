"""Survey-property systematic decontamination + validation (ISD + LSS check).

Methodology adopted from the DES Y6 galaxy-clustering analyses (Weaverdyck et al.
2026 and companions): Iterative Systematic Decontamination (ISD) of the galaxy
overdensity against survey-property (SP) templates, plus the LSS-template-rejection
safeguard that avoids subtracting real large-scale structure. Pure numpy/scipy, no
new dependencies.

In ECHOES this is used to **validate** that the completed spectroscopic catalog
carries no residual SP systematic — the per-template residual amplitude after
completion should be consistent with zero (χ²/dof ≈ 1) — not to re-weight the
released product. For a spectroscopic sample the relevant SP templates are the
imaging systematics that modulated *targeting* (Galactic extinction, stellar
density, …), the same quantities BOSS WEIGHT_SYSTOT was built from. The ISD weight
it returns is also a drop-in systematic model for extending ECHOES to spectroscopic
surveys that lack a pre-computed weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


def density_vs_template(t_data, t_rand, edges, w_data=None, w_rand=None):
    """Binned density contrast ``F(bin) = (n_data/n_rand)`` normalised to mean 1,
    with a per-bin Poisson error. A systematics-clean sample has ``F ≈ 1``."""
    t_data = np.asarray(t_data, float); t_rand = np.asarray(t_rand, float)
    w_data = np.ones_like(t_data) if w_data is None else np.asarray(w_data, float)
    w_rand = np.ones_like(t_rand) if w_rand is None else np.asarray(w_rand, float)
    nd = np.histogram(t_data, bins=edges, weights=w_data)[0]
    nr = np.histogram(t_rand, bins=edges, weights=w_rand)[0]
    cd = np.histogram(t_data, bins=edges)[0]                  # raw counts for Poisson err
    norm = w_data.sum() / w_rand.sum()
    ok = (nr > 0) & (cd > 0)
    F = np.ones(len(edges) - 1)
    F[ok] = (nd[ok] / nr[ok]) / norm
    sigma = np.where(cd > 0, np.where(ok, F, 1.0) / np.sqrt(np.maximum(cd, 1)), np.inf)
    return F, sigma, ok


def _chi2_flat(F, sigma, ok):
    """χ²/dof of the density-vs-template relation against flat (F=1)."""
    m = ok & np.isfinite(sigma) & (sigma > 0)
    if m.sum() < 2:
        return 0.0
    return float(np.sum(((F[m] - 1.0) / sigma[m]) ** 2) / m.sum())


@dataclass
class ISDResult:
    names: List[str]
    chi2_before: np.ndarray          # χ²/dof per template, no correction
    chi2_after: np.ndarray           # χ²/dof per template, after ISD weighting
    weight: np.ndarray               # (N_data,) per-galaxy systematic weight w=1/F
    removal_order: List[int]         # template indices in the order they were removed
    edges: list                      # per-template bin edges (for plotting)
    relations: list                  # per-template (F, sigma, ok) BEFORE correction

    @property
    def clean(self) -> bool:
        """All templates consistent with no residual systematic after ISD."""
        return bool(np.all(self.chi2_after < 2.0))


def isd_fit(data_templates, random_templates, *, names=None, n_bins=10, order=3,
            thresh=2.0, max_iter=30, data_weights=None, random_weights=None,
            clip=(0.2, 5.0)) -> ISDResult:
    """Iterative Systematic Decontamination of the galaxy density vs SP templates.

    ``data_templates``/``random_templates`` are ``(N, n_tpl)`` arrays of the SP
    values at each data galaxy / random point. Iteratively: measure the
    density-vs-template relation for every not-yet-removed template, take the most
    significant (largest χ²/dof) above ``thresh``, fit a degree-``order`` polynomial
    to it, and divide it out (the per-galaxy weight ``1/F``); repeat until all
    templates fall below ``thresh``. Returns the per-template χ²/dof before and
    after, the accumulated weight, and the removal order. ``clean`` is True when no
    residual systematic remains.
    """
    D = np.atleast_2d(np.asarray(data_templates, float))
    R = np.atleast_2d(np.asarray(random_templates, float))
    if D.shape[1] != R.shape[1] and D.shape[0] == R.shape[0]:
        D, R = D.T, R.T                                       # accept (n_tpl, N)
    n_tpl = D.shape[1]
    names = list(names) if names is not None else [f"t{i}" for i in range(n_tpl)]
    wd = np.ones(len(D)) if data_weights is None else np.asarray(data_weights, float).copy()
    wr = np.ones(len(R)) if random_weights is None else np.asarray(random_weights, float)

    edges = [np.quantile(R[:, i], np.linspace(0, 1, n_bins + 1)) for i in range(n_tpl)]
    for e in edges:
        e[0] -= 1e-9; e[-1] += 1e-9

    def chi2_all(weights):
        out = np.zeros(n_tpl)
        rel = []
        for i in range(n_tpl):
            F, s, ok = density_vs_template(D[:, i], R[:, i], edges[i], weights, wr)
            out[i] = _chi2_flat(F, s, ok); rel.append((F, s, ok))
        return out, rel

    chi2_before, relations = chi2_all(np.ones(len(D)))
    weight = np.ones(len(D))
    removed, removal_order = set(), []
    for _ in range(max_iter):
        chi2, rel = chi2_all(weight)
        cand = [(chi2[i], i) for i in range(n_tpl) if i not in removed]
        if not cand:
            break
        c, j = max(cand)
        if c < thresh:
            break
        F, s, ok = rel[j]
        cen = 0.5 * (edges[j][1:] + edges[j][:-1])
        m = ok & np.isfinite(s) & (s > 0)
        if int(m.sum()) <= order:
            removed.add(j); continue
        coeff = np.polyfit(cen[m], F[m], order, w=1.0 / s[m])
        Fi = np.clip(np.polyval(coeff, D[:, j]), clip[0], clip[1])
        weight = weight / Fi
        removed.add(j); removal_order.append(j)

    chi2_after, _ = chi2_all(weight)
    return ISDResult(names=names, chi2_before=chi2_before, chi2_after=chi2_after,
                     weight=weight, removal_order=removal_order, edges=edges,
                     relations=relations)


def lss_template_check(template_at_gal, delta_gal, template_at_rand=None):
    """Reject SP templates that trace real large-scale structure.

    Spearman-correlates the SP template value at galaxy positions with the local
    galaxy overdensity there (e.g. from
    :func:`echoes.selection_coupling.local_overdensity`). A template strongly
    correlated with the real density would, if regressed out, subtract genuine
    signal — the DES "LSS-template-rejection" safeguard. Returns
    ``(spearman_r, p_value, reject)`` with ``reject`` True at |r| significant
    (p<0.01) and |r|>0.1.
    """
    from scipy import stats
    t = np.asarray(template_at_gal, float); d = np.asarray(delta_gal, float)
    m = np.isfinite(t) & np.isfinite(d)
    r, p = stats.spearmanr(t[m], d[m])
    reject = bool((p < 0.01) and (abs(r) > 0.1))
    return float(r), float(p), reject
