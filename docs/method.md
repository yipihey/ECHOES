# How ECHOES works — a pedagogical walkthrough

This is the long-form, equation-light companion to the paper. It explains *why*
ECHOES is built the way it is, with the intuition behind each step. If you just
want to draw catalogs, see the [README](../README.md); if you want the rigorous
version with validation numbers, see the [paper](../paper/).

---

## 1. The problem: the interface is a summary statistic

A galaxy survey and a theory model never exchange raw data — they exchange a
**summary statistic**. For thirty years that statistic has almost always been a
two-point function (a correlation function or power spectrum), because the
observational corrections for two-point clustering have been studied, calibrated
in mocks, and baked into survey "weights" and random catalogs.

The catch: those corrections were worked out *for two-point statistics*. The moment
you want a different statistic — a void function, counts-in-cells, a marked
correlation, a nearest-neighbor distribution, a persistent-homology summary, or a
field-level likelihood — you inherit the survey's angular holes, fiber collisions,
redshift failures, and imaging systematics, and you have to re-derive how each of
them affects *your* statistic. That re-derivation is the real barrier, not the
statistic itself.

**ECHOES moves the interface one step earlier.** Instead of releasing weighted
galaxies plus randoms and asking every new statistic to understand the weights, it
releases a *posterior over completed catalogs* — equal-weight point sets that have
already absorbed the main incompleteness. You then run any statistic directly on
points.

## 2. The key insight: only the redshift is missing

BOSS targets were selected from SDSS imaging. Every galaxy that the spectrograph
*failed* to measure is still a real photometric detection: we know its **position**
and its **colors**. The incompleteness is almost entirely in the **redshift**
dimension.

So ECHOES never guesses *where* a missing galaxy is. It places each missing galaxy
at its **measured imaging position** and only samples the one thing that is actually
unknown — its redshift. This is what keeps the angular and two-dimensional
clustering faithful: the positions are data, not a model.

## 3. Which galaxies are missing, and how many

The survey's completeness weights are bookkeeping for incompleteness:
- `WEIGHT_CP` (close-pair) counts galaxies lost to **fiber collisions** — two targets
  closer than the 62″ fiber limit can't both be observed on one plate.
- `WEIGHT_NOZ` counts **redshift failures** — a spectrum was taken but no reliable
  redshift was measured.
- `WEIGHT_SYSTOT` is a smooth, degree-scale **imaging-systematic** modulation
  (stellar density, seeing, extinction).

ECHOES turns these weights back into explicit galaxies. It builds the missing-target
list by tying each weighted survivor to its real, un-observed imaging neighbors:
a `WEIGHT_CP` survivor claims its nearest unmatched photometric neighbors within
62″; a `WEIGHT_NOZ` survivor claims redshift-failure targets nearby. For
CMASS-South this recovers 5,272 fiber-collision and 1,505 redshift-failure targets.

## 4. The redshift posterior

For each missing galaxy we sample a redshift from a **data-driven posterior** that
multiplies three things along the galaxy's line of sight:

1. **A photo-z color likelihood.** A k-nearest-neighbor estimator in color space
   returns the redshift distribution of the most similar observed galaxies. It is
   *calibrated* (its probability-integral-transform is flat), so drawing from it is
   faithful — but on its own it is broad (σ_z ≈ 0.03, ~90 Mpc/h), which would smear
   line-of-sight clustering.
2. **A local-density (line-of-sight) prior.** The redshifts of the K nearest
   *observed* galaxies on the sky tell you where real structure sits along that
   sightline. Multiplying the photo-z by this local field sharpens it onto actual
   structure — this is what recovers 3-D clustering. (The default engine estimates
   this field with a fast KNN kernel; the optional graphGP engine replaces it with a
   proper conditional Gaussian-process field — see §6.)
3. **A close-pair prior (for collisions only).** Fiber-collided galaxies are often
   true physical pairs at the host's redshift; the empirical Δz distribution of
   observed close pairs encodes how often that is so.

Each missing galaxy's redshift is a draw from this product. Nothing about it assumes
a cosmology — it is angles, colors, and observed redshifts only.

## 5. Imaging systematics as *analog galaxies*, not weights

`WEIGHT_SYSTOT` is a smooth few-percent density modulation. Rather than carry it as
a per-object weight (which breaks equal-weight estimators), ECHOES restores the
implied galaxies as **local analogs**: it transplants nearby real galaxies (carrying
their redshifts) with a sub-arcsecond jitter. Crucially it never makes an *exact*
duplicate — exact duplicates put two galaxies at zero separation, which corrupts
nearest-neighbor, counts-in-cells, and every higher-order statistic. This is why the
nearest-neighbor CDFs come out clean (no spike at zero separation).

## 6. Two engines for the line-of-sight field

- **KNN-field (default).** A fast kernel estimate of the local redshift density from
  the K nearest observed galaxies. Cosmology-free, sharp at the sub-Mpc
  fiber-collision scale, and — because each missing galaxy's posterior is independent
  — it compresses to a tiny released file (§7). This is the default release.
- **graphGP (optional).** A conditional anisotropic Gaussian-process posterior over
  the *whole* density field, sampled by Matheron's rule on a sparse (Vecchia)
  neighbor graph, then evaluated along each missing sightline. Because all sightlines
  share one field draw, the redshift assignments are **correlated** across missing
  galaxies — the more faithful posterior object when a statistic is sensitive to
  coherent redshift uncertainty. It uses a fiducial distance–redshift relation only
  as an internal coordinate gauge: completing under Planck vs Einstein–de Sitter
  changes the assigned redshifts by ~0.001 and the recovered clustering by <0.1%, so
  it carries **no cosmological prior**.
  `pip install "echoes[graphgp] @ git+https://github.com/yipihey/ECHOES.git"`.

The two agree to the percent level; neither dominates. The KNN engine is sharper on
small scales, graphGP is smoother and slightly closer to the weighted reference on
large scales.

## 7. Why the release is 2 MB: relative samples

Every realization shares the *same* observed galaxies and the *same* missing-galaxy
positions — only the missing redshifts vary, and each missing galaxy's redshift
posterior is fixed once computed. So ECHOES stores the observed catalog **once** plus
a compact inverse-CDF of each missing galaxy's redshift posterior. A "sample" is then
just a seed: draw one uniform number per missing galaxy and invert its CDF. The whole
posterior is ~2 MB, you can draw unlimited realizations locally, and a reproducible
ensemble is simply a list of integer seeds.

## 8. How we know it works (validation logic)

Reproducing the survey's own weighted clustering is necessary but not sufficient —
it can be true by construction. ECHOES is validated by **recovering an unobserved
truth**:

- **Truth-known recovery.** Hide the redshifts of a subset of real BOSS galaxies (or
  use independent MultiDark-Patchy mocks), complete them, and check that the *full*
  truth clustering comes back — to 1–2% in the projected correlation. An "oracle"
  that restores the hidden galaxies at their *true* redshifts sets the achievable
  floor, isolating the redshift-assignment error.
- **Higher-order checks.** Nearest-neighbor CDFs and counts-in-cells (not just the
  two-point function) are recovered, confirming the analog step introduced no
  artifacts.
- **Calibration.** The realization spread is the *completion* uncertainty
  (0.17–0.44% on wp), which is well below the survey's sample variance — so it must be
  *added* to a cosmic-variance covariance, never used in its place.
- **Consistency.** On the real data, the equal-weight completed catalogs reproduce
  the official `w_c`-weighted wp and multipoles at the 1–5% level.

## 9. Using ECHOES in an analysis

```python
from echoes import load_package, draw
pkg = load_package("data_release/cmass_south_posterior.npz")

# run YOUR statistic on many seeds; mean = estimate, covariance = completion uncertainty
import numpy as np
vals = [my_statistic(draw(pkg, seed=s)) for s in range(200)]
estimate = np.mean(vals, axis=0)
completion_cov = np.cov(np.array(vals).T)     # ADD this to your sample/sim covariance
```
Pair the catalogs with `data_release/cmass_south_randoms.npz` and use **equal
weights** — no completeness weights. That is the whole point: the observational
correction has already been absorbed into the points.
