"""Build ECHOES "extension packs": chunked, progressively-loadable datasets for the browser viewer.

A pack lives at ``docs/packs/<id>/`` and is the canonical on-disk format shared by both renderers
(the WebGPU explorer streams it; the k3d fork references its texture sheets). Each pack carries a
``pack_manifest.json`` (``echoes.pack.v1`` — a superset of the viewer's ``echoes.viewer.v1``) that
declares load **tiers** (``core`` rendered first for fast startup, ``refinement``/``texture`` streamed
after) and **layers** (renderable point sets), each a set of typed-array chunks with sha256.

Subcommands:
  points   <product>   build a points pack (currently: local-2mpp true-3D neighbourhood)
  registry             (re)generate docs/packs/packs.json from the committed packs + known externals

    JAX_PLATFORMS=cpu ~/.venv/k3d/bin/python3 pipeline/build_pack.py points local-2mpp
    JAX_PLATFORMS=cpu ~/.venv/k3d/bin/python3 pipeline/build_pack.py registry
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))            # for _pack_io when run as a script
import numpy as np

from _pack_io import write_array, write_json, column_stats

ROOT = Path(__file__).resolve().parents[1]
PACKS_DIR = ROOT / "docs" / "packs"
LOCAL = ROOT / "data_release" / "local"

# the registry always carries the BOSS bundle (it lives in docs/visualizer/, not docs/packs/).
BOSS_ENTRY = {
    "id": "boss-cmass", "title": "BOSS DR12 CMASS-South",
    "description": "ECHOES completed posterior of BOSS DR12 CMASS-South (~120k galaxies, multiple seeds).",
    "version": "1.0.0", "kind": "points", "total_bytes": 3850000,
    "manifest_url": "../visualizer/data/viewer_manifest.json", "thumbnail": None,
    "requires": [], "default": True,
}


def _xyz(ra, dec, dist):
    ra = np.radians(np.asarray(ra, float)); dec = np.radians(np.asarray(dec, float))
    r = np.asarray(dist, float); cd = np.cos(dec)
    return np.column_stack([r * cd * np.cos(ra), r * cd * np.sin(ra), r * np.sin(dec)]).astype(np.float32)


def _point_layer(data_dir: Path, lid: str, tier: str, lod: int, color: str, xyz, value):
    """Write a renderable point layer (xyz + a scalar value for size/brightness) → manifest dict."""
    cols = {
        "xyz": write_array(data_dir, f"{tier}/{lid}_xyz.f32.bin", xyz, "<f4"),
        "value": write_array(data_dir, f"{tier}/{lid}_value.f32.bin", value, "<f4"),
    }
    bbox = [float(xyz[:, i].min()) for i in range(3)] + [float(xyz[:, i].max()) for i in range(3)]
    return {"id": lid, "tier": tier, "lod": lod, "count": int(len(xyz)), "color": color,
            "value_range": column_stats(value), "bbox": bbox, "columns": cols}


def build_points_local2mpp(out_dir: Path, mcmc: int = 0, core_extra: int = 0):
    """Pack the true-3D local 2M++ neighbourhood. **Core = the observed survey only** (fast, ~1.1 MB,
    shows the Zone-of-Avoidance gap); the ZoA and faint-end completions stream as **refinement** (you
    watch the gap fill). Positions are equatorial comoving [Mpc] (cartesian). ``core_extra`` optionally
    seeds the core with a bright subsample of completions for instant large-scale structure."""
    obs = np.load(LOCAL / "local_2mpp_observed.npz")
    ip = np.load(LOCAL / f"local_2mpp_full_mcmc{mcmc}.npz", allow_pickle=True)
    kind = ip["kind"].astype(str)
    ksm = ip["ksmag"]
    obs_xyz = _xyz(obs["ra"], obs["dec"], obs["dist_mpc"])
    z, f = kind == "zoa", kind == "faint"
    zoa_xyz, zoa_val = _xyz(ip["ra"][z], ip["dec"][z], ip["dist_mpc"][z]), ksm[z]
    fa_xyz, fa_val = _xyz(ip["ra"][f], ip["dec"][f], ip["dist_mpc"][f]), ksm[f]

    layers = [_point_layer(out_dir, "observed", "core", 0, "#9aa0a6", obs_xyz, obs["ksmag"])]
    if core_extra:                                           # optional bright completion preview
        cx = _xyz(ip["ra"], ip["dec"], ip["dist_mpc"]); cv = ksm
        sel = np.argsort(cv)[:core_extra]
        layers.append(_point_layer(out_dir, "completed-coarse", "core", 0, "#6a8fd0",
                                    cx[sel], cv[sel]))
    layers.append(_point_layer(out_dir, "completed-zoa", "refinement", 1, "#ff3b30", zoa_xyz, zoa_val))
    layers.append(_point_layer(out_dir, "completed-faint", "refinement", 1, "#3a78ff", fa_xyz, fa_val))

    manifest = {
        "schema_version": "echoes.pack.v1",
        "pack": {"id": "local-2mpp", "kind": "points", "version": "1.0.0",
                 "coordinate": "cartesian_mpc", "frame": "equatorial comoving"},
        "dataset": {"title": "Local 2M++ neighbourhood (true 3D)", "mcmc": mcmc,
                    "n_observed": int(len(obs_xyz)), "n_completed": int(len(ip["ra"]))},
        "tiers": {"core": {"label": "Core (fast start)", "priority": 0, "autoload": True},
                  "refinement": {"label": "Full completion", "priority": 10, "autoload": True}},
        "value_label": "K_s magnitude",
        "layers": layers,
    }
    write_json(out_dir / "pack_manifest.json", manifest)
    core_bytes = sum(c["bytes"] for L in layers if L["tier"] == "core" for c in L["columns"].values())
    total = sum(c["bytes"] for L in layers for c in L["columns"].values())
    print(f"[pack:local-2mpp] {len(layers)} layers, {sum(L['count'] for L in layers):,} points; "
          f"core {core_bytes/1e6:.2f} MB, total {total/1e6:.1f} MB -> {out_dir}")
    return manifest, core_bytes, total


def cmd_points(args):
    if args.product != "local-2mpp":
        sys.exit(f"unknown product {args.product!r} (only 'local-2mpp' so far)")
    out = Path(args.out) if args.out else PACKS_DIR / "local-2mpp"
    build_points_local2mpp(out, mcmc=args.mcmc, core_extra=args.core_extra)


def cmd_registry(args):
    """Regenerate docs/packs/packs.json: the BOSS entry + a scan of docs/packs/*/pack_manifest.json."""
    packs = [dict(BOSS_ENTRY)]
    for pm in sorted(PACKS_DIR.glob("*/pack_manifest.json")):
        m = json.loads(pm.read_text())
        pid = m["pack"]["id"]
        total = sum(c["bytes"] for L in m.get("layers", []) for c in L["columns"].values())
        total += sum(s.get("bytes", 0) for s in m.get("atlas", {}).get("sheets", []))
        packs.append({
            "id": pid, "title": m.get("dataset", {}).get("title", pid),
            "description": m.get("dataset", {}).get("description", ""),
            "version": m["pack"].get("version", "1.0.0"), "kind": m["pack"].get("kind", "points"),
            "total_bytes": int(total),
            "manifest_url": f"{pid}/pack_manifest.json", "thumbnail": None,
            "requires": m["pack"].get("requires", []), "default": False,
        })
    write_json(PACKS_DIR / "packs.json",
               {"schema_version": "echoes.packs.v1", "generator": "pipeline/build_pack.py registry",
                "packs": packs})
    print(f"[registry] {len(packs)} packs -> {PACKS_DIR / 'packs.json'}: {[p['id'] for p in packs]}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("points"); p.add_argument("product"); p.add_argument("--out", default=None)
    p.add_argument("--mcmc", type=int, default=0); p.add_argument("--core-extra", type=int, default=0)
    p.set_defaults(func=cmd_points)
    r = sub.add_parser("registry"); r.set_defaults(func=cmd_registry)
    args = ap.parse_args(); args.func(args)


if __name__ == "__main__":
    main()
