# ECHOES tutorials

Available now:

1. **[01_draw_samples.ipynb](01_draw_samples.ipynb)** — draw completed catalogs from
   the released posterior in ~10 lines (NumPy only; no large downloads).

Developer pointers for the full pipeline:

- Full BOSS completion from raw inputs: `pipeline/build_release.py`.
- graphGP/KNN engine comparison: `validation/graphgp_vs_knn.py`.
- Truth-known recovery: `validation/truth_recovery.py` and
  `validation/graphgp_truth_recovery.py`.
- Custom statistics across seeds: see `docs/method.md` §9.
