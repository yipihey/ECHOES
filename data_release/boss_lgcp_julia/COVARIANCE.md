# BOSS CMASS-South clustering covariance

## ⚠ Two different objects — do not confuse them

ECHOES estimates a **conditional posterior**, NOT a survey covariance. The two objects:

- **Unconditional (survey) covariance** — the scatter of clustering across *independent universes*
  (cosmic variance). Cosmology- and bias-dependent. Patchy is the standard; the lognormal-LGCP mock
  ensemble is an in-house approximation to it. **This is NOT the ECHOES posterior.** Most of this file
  (the LGCP↔Patchy work below) is about this object — useful only for the released *mock catalogs*.
- **Conditional completion posterior (THE ECHOES object)** — the range of clustering across
  realizations of *this* universe: every realization holds the securely-observed galaxies fixed and
  identical, and differs only where the observation process left freedom (collided/failed redshifts,
  ZoA/masked regions, accuracy). Realizations are highly correlated by construction; the covariance is
  the **observation-model-dependent** range given what we know for sure. Cosmology-free, and small.
  `pipeline/build_conditional_covariance.py`.

**Measured (N=100, `count=round`, observed galaxies held fixed): σ_conditional / σ_Patchy = 0.20 (wp),
0.17 (ξ₀), 0.26 (ξ₂)** — the ECHOES posterior is ~5× tighter than cosmic variance in σ (~20× in
variance); observation-model uncertainty on the clustering is ~0.6% (wp), 0.5% (ξ₀), 3% (ξ₂).
**Using Patchy or the LGCP mocks for the ECHOES posterior would OVER-state the uncertainty ~5×.** Two
ways to "regress to the average," both wrong: inflate with cosmic variance (Patchy), or collapse with a
factorized/plug-in posterior (this `round` number is the factorized lower bound — the true *joint*
conditional posterior sits above it but still well below cosmic variance). See DIAGNOSTICS.md.

---

## Unconditional LGCP↔Patchy covariance (for the released mock catalogs only — NOT the posterior)

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

**Bottom line (unconditional object).** If an *unconditional survey* covariance is ever needed (e.g.
to characterise the released LGCP *mock catalogs*), **Patchy** is the external standard; the
lognormal-LGCP version is intrinsically over-dispersed (a known property of high-σ² lognormal mocks).
But — see the top of this file — **the ECHOES posterior is NOT this object**: it is the conditional
completion covariance (`build_conditional_covariance.py`), ~5× tighter than cosmic variance and
cosmology-free. Do not use Patchy or the LGCP mocks for the ECHOES posterior; the LGCP mocks are for
the field completion engine. Reusable infra: f32 bridge output (2× ZIP32 batch ceiling),
`echoes/nongauss_lgcp.py` (validated Gaussianization), the Patchy + conditional covariance pipelines,
`pipeline/compare_covariances.py`.

**Products:**
- `covariance/covariance_conditional_N100_round_data.npz` — **the ECHOES conditional posterior
  covariance** (observation-model range; the object that matters for ECHOES inference).
- `covariance/covariance_patchy_N600.npz` — unconditional Patchy survey covariance (for the *mock
  catalogs*, NOT the posterior).
- `covariance/covariance_poisson_N1000_cf20_sets20.npz` — best LGCP unconditional covariance (20
  candidate sets; 2–4× over-dispersed vs Patchy, documented).
- `covariance/covariance_poisson_N1000_cf20.npz` — original 5-set LGCP covariance (kept for the record).
