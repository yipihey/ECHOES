# ECHOES tutorials

1. **[01_draw_samples.ipynb](01_draw_samples.ipynb)** — draw completed catalogs from
   the released posterior in ~10 lines (NumPy only; no large downloads).
2. **02_complete_a_survey** — run the full completion on BOSS from the raw inputs
   (`echoes.surveys.load_boss` → `complete_catalog_photoz`). See
   `pipeline/build_release.py` and `validation/` for runnable end-to-end examples.
3. **03_graphgp_engine** — the optional correlated graphGP redshift engine
   (`pip install echoes[graphgp]`; `build_gp_field` + `z_mode='graphgp'`). See
   `validation/graphgp_vs_knn.py`.
4. **04_custom_statistic** — measure your own statistic across seeds and propagate
   the completion covariance (see `docs/method.md` §9).
