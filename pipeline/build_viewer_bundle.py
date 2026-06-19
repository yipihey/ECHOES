"""Build the static ECHOES WebGPU viewer bundle.

The viewer is served from ``docs/visualizer/`` by GitHub Pages.  This script
keeps the browser payload reproducible from the released posterior package:

    uv run python pipeline/build_viewer_bundle.py --seeds 0 1 2 3

It writes a small manifest plus little-endian typed-array chunks.  The observed
catalog is stored once; each realization stores only the missing-galaxy redshift
draws and the WEIGHT_SYSTOT analog extras produced by that seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from echoes.posterior import draw, load_package  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE = ROOT / "data_release" / "cmass_south_posterior.npz"
DEFAULT_SOURCE = ROOT / "apps" / "echoes-viewer"
DEFAULT_OUT = ROOT / "docs" / "visualizer"

PROVENANCE_CODES = {
    "0": {
        "label": "observed spectroscopy",
        "short_label": "observed",
        "description": "Original BOSS CMASS-South galaxy with a measured spectroscopic redshift.",
        "color": "#d8dde5",
    },
    "1": {
        "label": "fiber-collision completion",
        "short_label": "collided",
        "description": "ECHOES galaxy restored at an imaging-target position affected by the fiber-collision scale.",
        "color": "#39b5ff",
    },
    "2": {
        "label": "redshift-failure completion",
        "short_label": "zfail",
        "description": "ECHOES galaxy restored for a failed spectroscopic redshift.",
        "color": "#c071ff",
    },
    "3": {
        "label": "imaging-systematic analog",
        "short_label": "systot",
        "description": "ECHOES analog galaxy sampled from the WEIGHT_SYSTOT multiplicity model.",
        "color": "#ffb84d",
    },
    "4": {
        "label": "host-redshift fallback",
        "short_label": "zhost",
        "description": "Fallback redshift assignment used when a missing-galaxy posterior is degenerate.",
        "color": "#ff6f61",
    },
}


def _jsonify(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonify(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_array(data_dir: Path, rel: str, arr: np.ndarray, dtype: str) -> dict[str, Any]:
    path = data_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.ascontiguousarray(np.asarray(arr).astype(np.dtype(dtype), copy=False))
    path.write_bytes(out.tobytes(order="C"))
    return {
        "file": rel,
        "dtype": np.dtype(dtype).str,
        "shape": list(out.shape),
        "count": int(out.size),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _copy_static(source: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "styles.css", "app.js", "README.md"):
        src = source / name
        if src.exists():
            shutil.copy2(src, out / name)


def _column_stats(arr: np.ndarray) -> dict[str, float]:
    a = np.asarray(arr)
    return {"min": float(np.nanmin(a)), "max": float(np.nanmax(a))}


def _safe_column_id(name: str) -> str:
    out = []
    for ch in name.strip().lower():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "_":
            out.append("_")
    col = "".join(out).strip("_")
    return col or "column"


def _column_role(name: str, arr: np.ndarray) -> str:
    lname = name.lower()
    if "weight" in lname or lname.startswith("w_") or lname.startswith("w"):
        return "weight"
    if np.issubdtype(arr.dtype, np.integer) and len(np.unique(arr[: min(len(arr), 10000)])) <= 32:
        return "categorical"
    return "scalar"


def _add_enriched_npz(
    *,
    data_dir: Path,
    base_columns: dict[str, Any],
    manifest_columns: list[dict[str, Any]],
    enriched_npz: Path | None,
    n_obs: int,
    n_base: int,
) -> list[str]:
    """Append optional one-dimensional observed/base columns from an NPZ file.

    Arrays of length ``n_obs`` are interpreted as observed-galaxy columns. Arrays
    of length ``n_base`` are interpreted as fixed base-catalog columns. Other
    arrays are ignored because they cannot be aligned with the current public
    bundle without an explicit adapter.
    """
    if enriched_npz is None:
        return []
    enriched_npz = Path(enriched_npz)
    if not enriched_npz.exists():
        raise FileNotFoundError(f"enriched column bundle not found: {enriched_npz}")
    added = []
    with np.load(enriched_npz, allow_pickle=False) as d:
        used_ids = set(base_columns) | {c["id"] for c in manifest_columns}
        for raw_name in sorted(d.files):
            arr = np.asarray(d[raw_name])
            if arr.ndim != 1 or len(arr) not in (n_obs, n_base):
                continue
            if not (np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_)):
                continue
            col_id = _safe_column_id(raw_name)
            base_id = col_id
            suffix = 2
            while col_id in used_ids:
                col_id = f"{base_id}_{suffix}"
                suffix += 1
            used_ids.add(col_id)
            role = _column_role(raw_name, arr)
            if role == "categorical" and np.nanmin(arr) >= 0 and np.nanmax(arr) <= 255:
                dtype = "u1"
                out = arr.astype(np.uint8)
            else:
                dtype = "<f4"
                out = arr.astype(np.float32)
            base_columns[col_id] = _write_array(data_dir, f"base/enriched/{col_id}.{np.dtype(dtype).str.replace('|', '').replace('<', '')}.bin", out, dtype)
            meta = {
                "id": col_id,
                "label": raw_name,
                "units": "",
                "role": role,
                "availability": "observed" if len(arr) == n_obs else "base",
                **_column_stats(out.astype(np.float32)),
            }
            if role == "categorical":
                vals = np.unique(out[: min(len(out), 100000)])
                if len(vals) <= 32:
                    meta["categories"] = {str(int(v)): {"label": str(int(v))} for v in vals}
            manifest_columns.append(meta)
            added.append(col_id)
    return added


def build_viewer_bundle(
    *,
    package: Path = DEFAULT_PACKAGE,
    source: Path = DEFAULT_SOURCE,
    out: Path = DEFAULT_OUT,
    seeds: list[int] | tuple[int, ...] = (0, 1, 2, 3),
    enriched_npz: Path | None = None,
) -> Path:
    out = Path(out)
    source = Path(source)
    package = Path(package)
    if not package.exists():
        raise FileNotFoundError(f"posterior package not found: {package}")
    if not source.exists():
        raise FileNotFoundError(f"viewer source directory not found: {source}")

    _copy_static(source, out)
    data_dir = out / "data"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    pkg = load_package(package)
    n_obs = int(pkg["n_obs"])
    n_miss = int(pkg["n_miss"])
    n_base = n_obs + n_miss

    base_columns = {
        "ra": _write_array(data_dir, "base/base_ra.f32.bin", pkg["base_ra"], "<f4"),
        "dec": _write_array(data_dir, "base/base_dec.f32.bin", pkg["base_dec"], "<f4"),
        "weight_systot": _write_array(data_dir, "base/base_weight_systot.f32.bin", pkg["base_wsys"], "<f4"),
        "provenance": _write_array(data_dir, "base/base_provenance.u8.bin", pkg["base_prov"].astype(np.uint8), "u1"),
        "observed_z": _write_array(data_dir, "base/observed_z.f32.bin", pkg["obs_z"], "<f4"),
    }

    manifest_columns = [
        {"id": "ra", "label": "RA", "units": "deg", "role": "coordinate", **_column_stats(pkg["base_ra"])},
        {"id": "dec", "label": "Dec", "units": "deg", "role": "coordinate", **_column_stats(pkg["base_dec"])},
        {"id": "z", "label": "redshift", "units": "", "role": "coordinate", "min": float(pkg["zmin"]), "max": float(pkg["zmax"])},
        {"id": "weight_systot", "label": "WEIGHT_SYSTOT analog", "units": "", "role": "weight", **_column_stats(pkg["base_wsys"])},
        {"id": "provenance", "label": "provenance", "units": "", "role": "categorical", "categories": PROVENANCE_CODES},
        {"id": "source", "label": "source", "units": "", "role": "categorical", "categories": {
            "0": {"label": "observed", "color": PROVENANCE_CODES["0"]["color"]},
            "1": {"label": "ECHOES completion", "color": "#41d6b0"},
        }},
    ]
    enriched_columns = _add_enriched_npz(
        data_dir=data_dir,
        base_columns=base_columns,
        manifest_columns=manifest_columns,
        enriched_npz=enriched_npz,
        n_obs=n_obs,
        n_base=n_base,
    )

    realizations = []
    for seed in seeds:
        cat = draw(pkg, seed=int(seed), systot=True)
        z_miss = np.asarray(cat["z"][n_obs:n_base], dtype=np.float32)
        extra_slice = slice(n_base, int(cat["N"]))
        extra_count = int(cat["N"] - n_base)
        prefix = f"methods/knn-field/seed-{int(seed):04d}"
        chunks = {
            "missing_z": _write_array(data_dir, f"{prefix}/missing_z.f32.bin", z_miss, "<f4"),
            "extra_ra": _write_array(data_dir, f"{prefix}/extra_ra.f32.bin", cat["ra"][extra_slice], "<f4"),
            "extra_dec": _write_array(data_dir, f"{prefix}/extra_dec.f32.bin", cat["dec"][extra_slice], "<f4"),
            "extra_z": _write_array(data_dir, f"{prefix}/extra_z.f32.bin", cat["z"][extra_slice], "<f4"),
            "extra_provenance": _write_array(
                data_dir,
                f"{prefix}/extra_provenance.u8.bin",
                np.asarray(cat["prov"][extra_slice], dtype=np.uint8),
                "u1",
            ),
        }
        counts = dict(zip(*[a.tolist() for a in np.unique(cat["prov"], return_counts=True)]))
        realizations.append({
            "id": f"seed-{int(seed):04d}",
            "label": f"seed {int(seed)}",
            "seed": int(seed),
            "total_count": int(cat["N"]),
            "base_count": n_base,
            "extra_count": extra_count,
            "provenance_counts": {str(k): int(v) for k, v in sorted(counts.items())},
            "chunks": chunks,
        })

    manifest = {
        "schema_version": "echoes.viewer.v1",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "pipeline/build_viewer_bundle.py",
        "dataset": {
            "id": "boss-dr12-cmass-south",
            "label": "BOSS DR12 CMASS-South",
            "release_product": str(package.relative_to(ROOT)) if package.is_relative_to(ROOT) else str(package),
            "description": "ECHOES completed-catalog posterior for BOSS DR12 CMASS-South.",
        },
        "counts": {
            "observed": n_obs,
            "missing": n_miss,
            "base": n_base,
        },
        "cosmology": {
            "id": "echoes-fiducial-flat-lcdm",
            "label": "ECHOES fiducial flat LCDM",
            "Om": 0.31,
            "h": 0.68,
            "w0": -1.0,
            "wa": 0.0,
            "c_over_H100_Mpch": 2997.92458,
            "distance_units": ["observed variables", "comoving Mpc/h", "proper Mpc/h"],
            "note": "Matches echoes.distance.DistanceCosmo defaults; distances are for visualization and validation, not for defining the released catalog.",
        },
        "coordinate_modes": [
            {"id": "observed", "label": "Observed variables", "axes": ["wrapped RA [deg]", "Dec [deg]", "redshift-scaled z"]},
            {"id": "comoving", "label": "Fiducial comoving Mpc/h", "axes": ["x [Mpc/h]", "y [Mpc/h]", "z [Mpc/h]"]},
            {"id": "proper", "label": "Fiducial proper Mpc/h", "axes": ["x [Mpc/h]", "y [Mpc/h]", "z [Mpc/h]"]},
        ],
        "columns": manifest_columns,
        "base": {
            "description": "Fixed observed galaxies plus fixed missing-galaxy angular positions. Observed redshifts are stored once; missing redshifts are realization-specific.",
            "columns": base_columns,
        },
        "methods": [
            {
                "id": "knn-field",
                "label": "KNN-field ECHOES",
                "description": "Default compact local-density/posterior engine used by the released ECHOES BOSS bundle.",
                "realizations": realizations,
            }
        ],
        "provenance_codes": PROVENANCE_CODES,
        "enriched_bundle": {
            "supported": True,
            "description": "Pass --enriched-npz with one-dimensional observed/base columns to append raw catalog parameters, computed weights, or method diagnostics without changing the viewer runtime.",
            "column_extension_point": "columns",
            "method_extension_point": "methods",
            "columns_added": enriched_columns,
        },
        "default_view": {
            "method_id": "knn-field",
            "realization_id": realizations[0]["id"],
            "coordinate_mode": "comoving",
            "projection": "3d",
            "color_by": "provenance",
            "size_by": "source",
        },
    }

    manifest_path = data_dir / "viewer_manifest.json"
    manifest_path.write_text(json.dumps(_jsonify(manifest), indent=2, sort_keys=True) + "\n")
    return manifest_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the ECHOES WebGPU viewer bundle.")
    p.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument(
        "--enriched-npz",
        type=Path,
        default=None,
        help="optional NPZ of 1-D numeric columns of length n_obs or n_base to expose in the viewer",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest_path = build_viewer_bundle(
        package=args.package,
        source=args.source,
        out=args.out,
        seeds=args.seeds,
        enriched_npz=args.enriched_npz,
    )
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
