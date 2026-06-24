"""Shared chunk + integrity primitives for the ECHOES viewer/pack builders.

The little-endian typed-array chunk + sha256 manifest format is the single on-disk contract the
browser viewer (`apps/echoes-viewer`) and the extension-pack system both read. These helpers were
factored out of ``pipeline/build_viewer_bundle.py`` so ``pipeline/build_pack.py`` reuses them
verbatim — there is exactly one implementation of "write an array as a chunk and describe it".
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def jsonify(x: Any) -> Any:
    """Recursively coerce numpy scalars/containers to JSON-native types."""
    if isinstance(x, dict):
        return {str(k): jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonify(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


def sha256(path: Path) -> str:
    """Streaming sha256 hex digest of a file (matches the viewer's crypto.subtle check)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_array(data_dir: Path, rel: str, arr: np.ndarray, dtype: str) -> dict[str, Any]:
    """Write ``arr`` as a contiguous little-endian binary chunk at ``data_dir/rel``.

    Returns the manifest descriptor ``{file, dtype, shape, count, bytes, sha256}`` the viewer's
    ``fetchArray`` consumes. ``dtype`` is a numpy dtype string (e.g. ``'<f4'``, ``'|u1'``, ``'|u2'``).
    """
    path = Path(data_dir) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.ascontiguousarray(np.asarray(arr).astype(np.dtype(dtype), copy=False))
    path.write_bytes(out.tobytes(order="C"))
    return {
        "file": rel,
        "dtype": np.dtype(dtype).str,
        "shape": list(out.shape),
        "count": int(out.size),
        "bytes": int(path.stat().st_size),
        "sha256": sha256(path),
    }


def column_stats(arr: np.ndarray) -> dict[str, float]:
    """Min/max of a column (NaN-safe) for the manifest's colour/size range hints."""
    a = np.asarray(arr)
    return {"min": float(np.nanmin(a)), "max": float(np.nanmax(a))}


def write_json(path: Path, obj: Any) -> Path:
    """Write a JSON document (jsonify'd) with a trailing newline; returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonify(obj), indent=2) + "\n")
    return path
