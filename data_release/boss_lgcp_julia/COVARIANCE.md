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

## What the four next-steps actually found (resolved; see DIAGNOSTICS.md for the data)

All four were executed. The headline: **against the external Patchy truth the lognormal-LGCP
clustering covariance is 2–6× over-dispersed, and that over-dispersion is intrinsic — two independent
in-house levers (#1, #4) provably cannot remove it.**

1. **#1 density-match via `n_cand` — INTRACTABLE.** `multi_frac ∝ cf⁻⁰·⁴⁷` (measured 0.195/0.143/0.101
   at cf=20/40/80), so the target ≲0.02 needs cf≈2400 (≈2.6×10⁸ points). Closed.
2. **#2 more independent candidate sets — REAL but PARTIAL.** `--batch 50` → 20 candidate sets (vs 5)
   dropped wp over-dispersion vs Patchy 5.8→4.0× and the small-scale extremes (bin-0 33→22). Kept as
   the better LGCP covariance, but 2–4× over-dispersion remains.
3. **#3 external Patchy cross-check — DONE, and it is the answer.** 600 MultiDark-Patchy DR12-SGC mocks
   (`pipeline/build_patchy_covariance.py`, Hartlap 0.975, same estimator/binning, data-matched
   z∈0.45–0.60, N=109,636). σ_lgcp/σ_patchy median: wp 3.98, ξ₀ 2.08, ξ₂ 2.53 (#2 covariance). The
   angular jackknife disagrees with Patchy (flips sign for ξ₀), confirming it cannot validate the
   multipoles. **Patchy is the standard BOSS covariance and is now built — use it for the covariance.**
4. **#4 Tier-A non-Gaussian transform — DEAD, same deep cause as #1.** Built + measured
   (`pipeline/diag_tierA.py`): no intensity transform reduces multi-occupancy while preserving the
   2-pt, because multi-occupancy ∝ `var(1+δ)=ξ(0,0)≈63`, which the 2-pt locks (skew=30 → multi_frac
   unchanged; skew≤10 → 2-pt collapses, K_out/K_in 0.32→0.09). The only lever left is *reducing* σ²
   (smoothing to the candidate scale), a separate method-changing direction that sacrifices the
   sub-candidate-spacing 2-pt.

**Bottom line.** For the **clustering covariance**, ship **Patchy** (external standard, now built);
the lognormal-LGCP covariance is intrinsically over-dispersed (a known property of high-σ² lognormal
mocks) and is documented as such. The **LGCP mocks remain the right engine for the field-level
completion / posterior work** (their purpose), where the over-dispersion is irrelevant. Reusable
infra delivered: f32 bridge output (2× ZIP32 batch ceiling), `echoes/nongauss_lgcp.py` (validated
Gaussianization), the Patchy covariance pipeline, `pipeline/compare_covariances.py`.

**Products:**
- `covariance/covariance_patchy_N600.npz` — **the recommended BOSS clustering covariance** (wp/ξ₀/ξ₂).
- `covariance/covariance_poisson_N1000_cf20_sets20.npz` — best LGCP covariance (20 candidate sets;
  2–4× over-dispersed vs Patchy, documented).
- `covariance/covariance_poisson_N1000_cf20.npz` — original 5-set LGCP covariance (kept for the record).
