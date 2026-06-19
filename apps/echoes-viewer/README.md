# ECHOES WebGPU visualizer

Static browser app for comparing the observed BOSS CMASS-South catalog with
ECHOES completed-catalog realizations.

Build the GitHub Pages artifact and data bundle from the repository root:

```bash
uv run python pipeline/build_viewer_bundle.py --seeds 0 1 2 3
```

Optional enriched builds can append any numeric one-dimensional NPZ columns of
length `n_obs` or `n_base`:

```bash
uv run python pipeline/build_viewer_bundle.py --enriched-npz enriched_columns.npz
```

The generated site is written to `docs/visualizer/` and can be served by any
static file server. The runtime expects WebGPU and uses the manifest at
`data/viewer_manifest.json` to discover methods, realizations, columns,
coordinate modes, and binary array chunks.

Method adapters should add manifest entries under `methods[]` and provide the
same chunk contract: fixed observed/base columns once, plus per-realization
missing-redshift and extra-object chunks.
