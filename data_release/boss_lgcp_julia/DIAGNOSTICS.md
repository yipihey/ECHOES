# Covariance next-step diagnostics

Findings that redirect the COVARIANCE.md next-steps. Reproduce with
`pipeline/diag_multifrac_vs_cf.py`.

## #1 (density-match via `n_cand`) is INTRACTABLE — multi-occupancy is a field-model problem

The N=1000 covariance is over-dispersed at small rp because the σ²=4.14 lognormal intensity
`opd = exp(f − σ²/2)` has a runaway tail: a candidate at field value g=4σ gets intensity
≈430× the mean, so even with cf=20 it Poisson-samples ~20 stacked galaxies. Next-step #1 proposed
raising `n_cand` until the per-candidate rate p<1 even at peaks. Measured `multi_frac(cf)` (fraction
of occupied candidates with ≥2 galaxies), CMASS-South kernel σ²=4.144, GPU:

| n_cand_factor | n_cand | multi_frac | N_gal | maxbatch f32 / f64 | build+2gen |
|---:|---:|---:|---:|---:|---:|
| 20  |  2,192,720 | 0.1954 | 117,626 | 456 / 228 | 88 s |
| 40  |  4,385,440 | 0.1428 | 117,580 | 228 / 114 | 141 s |
| 80  |  8,770,880 | 0.1013 | 117,692 | 114 / 57 | 290 s |
| 160 | 17,541,760 | ~0.073 (predicted) | — | 57 / 28 | (aborted: 17.5M build) |

**The decay is a slow power law `multi_frac ∝ cf^−0.475`** (0.1954/0.1428/0.1013 fit it to <1%) —
NOT the `∝1/cf` that density-matching would need. Extrapolating to the target `multi_frac ≲ 0.02`:

> **cf ≈ 2400  →  n_cand ≈ 2.6×10⁸ points** — flatly infeasible (build cost, GPU memory).

Even `multi_frac = 0.05` needs cf ≈ 350 (≈3.8×10⁷ points, multi-minute builds, OOM risk). So #1 as
written cannot work: the heavy multi-occupancy tail is **intrinsic to the lognormal σ²=4.14 field**,
not a candidate-density deficit. Bernoulli was the other lever and it globally under-clusters (~2×).
⇒ The real fix is **next-step #4** (the measured non-Gaussian transform `T(g)`): the data's own CiC
1-point PDF has skew ≈3.1, vs the σ²=4.14 lognormal's skew ≈500, so `T(g)` bounds the peak intensity
and multi-occupancy collapses at the source — what no feasible cf can do. #1 is closed; #4 absorbs it.

## Infra win (folded in): f32 bridge output

`echoes/jl/run_graphgp.jl` now writes the `generate` output in the run dtype (f32 in production)
instead of always-f64 — a pure serialization change (GPU compute was already f32). It **halves** the
`(batch, n_cand)` output and so **doubles the ZIP32 4 GB/entry batch ceiling** (`maxbatch f32` column
above = 2× f64), which is what makes any high-`n_cand` ensemble writable at all. The f64 parity gate
is unchanged (T=Float64 → bit-identical).

## #4 (Tier-A non-Gaussian transform) is ALSO dead — for the SAME deep reason as #1

The Tier-A idea (replace the lognormal `exp(f−σ²/2)` with a measured transform `T(g)` whose 1-point
PDF has the data's lower skew, re-deriving the kernel via `ξ_T⁻¹` so the 2-pt is preserved —
`echoes/nongauss_lgcp.py` + `kernel_from_K2d_tierA`) was built, validated, and **measured to fail**
(`pipeline/diag_tierA.py`, σ²=4.164 ⇒ ξ(0,0)=63):

| intensity transform | skew | multi_frac | K_out/K_in (2-pt) |
|---|---:|---:|---:|
| lognormal (baseline) | ≈528 | 0.197 | ≈1.0 (preserved) |
| Tier-A shifted-lognormal | 30 | 0.206 | 0.32 |
| Tier-A shifted-lognormal | 10 | 0.171 | 0.18 |
| Tier-A shifted-lognormal | 3.1 | 0.116 | **0.09** |

**No transform reduces multi_frac while preserving the 2-pt.** The rigorous reason: the per-candidate
Poisson rate has variance ∝ `var(1+δ) = ξ(0,0) ≈ 63`, which is **fixed by the 2-pt**. Multi-occupancy
is set by that variance, not the skew — so any transform that keeps the 2-pt keeps the multi-occupancy
(skew=30 → multi_frac unchanged). Worse, a mean-1 **non-negative** field with variance 63 has a hard
skew floor ≈ √63 ≈ 8 (the empty-or-huge two-point limit); pushing skew below it makes the shifted
lognormal go negative in the bulk (`T(0)<0`, clipped to 0 → field mostly empty → the slight multi_frac
drop comes with a collapsed 2-pt, K_out/K_in = 0.32 → 0.18 → 0.09). Both #1 and #4 fail because
**multi-occupancy is locked to the field's high zero-lag variance σ²=4.16, and that is locked by the
2-pt.** The only lever that can move it is *reducing* that variance (smoothing the field to the
candidate scale), which deliberately sacrifices the sub-candidate-spacing 2-pt — a separate, larger,
method-changing direction.

`echoes/nongauss_lgcp.py` (the `ξ_T(ρ)` Gauss–Hermite Gaussianization, validated to 1e-15 vs the
lognormal closed form) and `kernel_from_K2d_tierA` remain as correct, reusable machinery for regimes
where σ² is moderate (e.g. a smoothed field, or the local-volume work) — they are not wrong, the LGCP
σ²=4.16 regime is just outside where a 1-point reshape can help.

## #2 (candidate independence) — real but partial; #3 (Patchy) is the answer

**#3 (external truth).** `pipeline/build_patchy_covariance.py` — 600 MultiDark-Patchy DR12-SGC mocks
(Hartlap h=0.975), same estimator/binning, matched to the data slice (z∈0.45–0.60, N=109,636). vs the
LGCP (`pipeline/compare_covariances.py`, σ_lgcp/σ_patchy median):

| covariance | wp | ξ₀ | ξ₂ |
|---|---:|---:|---:|
| LGCP baseline (5 candidate sets) | 5.81 | 2.27 | 2.64 |
| LGCP #2 (20 candidate sets) | 3.98 | 2.08 | 2.53 |
| Patchy (reference) | 1.0 | 1.0 | 1.0 |

**#2.** `--batch 50` → 20 independent candidate sets (vs the baseline's 5). It HELPED — decorrelating
the shared candidate peaks dropped wp over-dispersion 5.8→4.0 and the small-scale extremes (wp bin-0
ratio 33.6→22.0, ξ₀ 50→17, ξ₂ 65→22). But the LGCP is still **2–4× over-dispersed** — the irreducible
multi-occupancy floor (#1/#4). The angular jackknife disagrees with Patchy and even flips sign for ξ₀
(σ_jk/σ_mock=2.99 says "mock too small"; Patchy says 2× too large) — confirming it cannot validate the
multipoles. **Patchy is the external truth and the LGCP clustering covariance is intrinsically
over-dispersed; for the covariance deliverable, use Patchy (now built) directly.**
