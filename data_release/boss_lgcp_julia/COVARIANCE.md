# BOSS CMASS-South LGCP covariance (1000 mocks, Julia GPU)

`pipeline/build_boss_covariance.py --n-realizations 1000 --n-cand-factor 20 --n-data-meas 60000
--n-jk 50 --n-rand-meas-clust 4 --sampling poisson --device gpu --batch 200`

1000 anisotropic-LGCP realizations on the Julia GPU engine, **exact mangle mask + FKP weights** on
galaxies and randoms, wp(rp)/ξ₀(s)/ξ₂(s) per realization (Corrfunc Landy–Szalay, shared RR cache),
sample covariance + Anderson–Hartlap, cross-checked against a 50-region data jackknife.

## Performance

| stage | time |
|---|---:|
| load + K_in kernel | 70 s |
| generate 1000 (5 batches × 200, GPU) | 1115 s (1.11 s/real, graph_builds=5) |
| measure wp/ξ₀/ξ₂ ×1000 | 1092 s (1.1 s/real, RR cached) |
| 50-region data jackknife | 105 s |
| **total** | **2383 s (~40 min)** |

Hartlap factors valid (h ≈ 0.985–0.987 at N=1000 ≫ n_bins). The whole 1000-mock covariance is ~40 min
on one A6000 — the practical payoff of the GPU engine + the amortized build.

## Validation: jackknife(data) vs mock diagonal σ (the trust check)

| stat | median σ_jk/σ_mock | reading |
|---|---:|---|
| **wp** | **0.97** | median excellent; but scale-dependent (see below) |
| ξ₀ | 2.74 | mock variance ~2–3× **too small** (under-dispersed) at most bins |
| ξ₂ | 0.49 | mock variance ~2× **too large** (over-dispersed) at most bins |

**wp is the trustworthy one at intermediate scales** (rp ≈ 2–8 Mpc/h, ratio 0.8–3). Two honest gaps:
- **Small rp (<2 Mpc/h): mock over-dispersed** (ratio →0.05) — the Δθ→0 Poisson multi-occupancy spike
  (multi_frac ≈ 0.20) inflates the small-scale variance.
- **Large rp (>10 Mpc/h): mock under-dispersed** (ratio →10) — the realizations share candidate
  positions within each batch (only 5 candidate sets for 1000 mocks), so the large-scale sample
  variance is under-sampled. (Jackknife also *over*-estimates large-scale variance, so the truth is
  between — but the under-dispersion is real.)

The convergence diagnostic is NOT flat (last-half→end change ~55–73%): a few multi-occupancy-tail
realizations dominate the small-scale variance, so N=1000 is not yet converged there.

## What this means / next steps (in priority)

1. **Kill multi-occupancy properly** — raise `n_cand` until p<1 even at field peaks (density-matched),
   NOT Bernoulli (which under-clusters globally). This fixes the small-scale over-dispersion + the
   non-convergence, and is the single biggest fidelity lever.
2. **More independent candidate sets** — smaller `--batch` (more graph builds) so the large-scale
   sample variance is properly sampled; fixes the wp/ξ₀ large-scale under-dispersion.
3. **External cross-check** — compare ξ₀/ξ₂ covariance to EZmocks/Patchy; the angular jackknife
   structurally mis-estimates the multipole variance (over-estimates monopole large-scale,
   under-estimates quadrupole), so it cannot by itself validate the multipoles.
4. **Field fidelity** — for publishable multipole covariance, the lognormal model likely needs the
   non-Gaussian Tier-A transform (it mismatches higher-order statistics).

**Status:** the covariance *pipeline* is validated and fast (1000 mocks/40 min, valid Hartlap, jackknife
cross-check); the wp covariance is usable at intermediate scales; the multipole + small/large-scale
covariance need the multi-occupancy + candidate-independence + external-cross-check work above before
they are publication-grade. Products: `covariance/covariance_poisson_N1000_cf20.npz` (cov + Hartlap +
jackknife) and `covariance/measurements_poisson_N1000_cf20.npz` (per-realization wp/ξ₀/ξ₂).
