"""Stage-B textured local viewer using the custom k3d fork's TexturedPoints object.

Unlike the stock-k3d demonstrator (``build_local_viewer.py``, world-fixed sky-tangent quads that must
be size-exaggerated), this uses the ECHOES k3d fork's ``textured_points``: instanced, CAMERA-FACING
galaxy-image billboards at TRUE physical scale, oriented by axis ratio + position angle, with the sky
keyed transparent by luminance so galaxies glow in 3D. Combined with the point-cloud layers this gives
an implicit level-of-detail — distant galaxies read as points, and the real image resolves as the
camera approaches. ``size_scale`` exaggerates uniformly for overview framing.

Requires the fork on the path (it provides ``k3d.textured_points`` + the matching JS renderer) and the
self-contained ``snapshot_type='full'`` so the viewer embeds the fork's renderer (no CDN):

    PYTHONPATH=~/Projects/echoes-k3d JAX_PLATFORMS=cpu ~/.venv/k3d/bin/python3 \
        pipeline/build_local_viewer_fork.py --atlas-dir docs/visualizer-local/data \
        --size-scale 1.0 --out docs/local_viewer_fork.html
"""
import argparse, io, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

LOCAL = os.path.join("data_release", "local")
ARCMIN2RAD = np.pi / (180.0 * 60.0)


def _xyz(d):
    ra = np.radians(d["ra"].astype(float)); dec = np.radians(d["dec"].astype(float))
    r = d["dist_mpc"].astype(float); cd = np.cos(dec)
    return np.column_stack([r * cd * np.cos(ra), r * cd * np.sin(ra), r * np.sin(dec)]).astype(np.float32)


def _sample(xyz, n, rng):
    return xyz if len(xyz) <= n else xyz[rng.choice(len(xyz), n, replace=False)]


def _tile_rects(tile_idx, m):
    """Atlas (u0,v0,u1,v1) per instance from tile indices + manifest geometry (gutter-inset)."""
    per = m["tiles_per_row"]; tp = m["tile_px"]; sh = m["sheet_px"]; g = m["gutter_px"]
    within = tile_idx % m["tiles_per_sheet"]
    row = within // per; col = within % per
    u0 = (col * tp + g) / sh; v0 = (row * tp + g) / sh
    u1 = (col * tp + tp - g) / sh; v1 = (row * tp + tp - g) / sh
    return np.column_stack([u0, v0, u1, v1]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas-dir", default=os.path.join("docs", "visualizer-local", "data"))
    ap.add_argument("--mcmc", type=int, default=0)
    ap.add_argument("--mode", default="full")
    ap.add_argument("--n-obs", type=int, default=60000)
    ap.add_argument("--n-faint", type=int, default=60000)
    ap.add_argument("--size-scale", type=float, default=1.0,
                    help="size exaggeration (1.0 = true physical scale; bump for overview framing)")
    ap.add_argument("--max-tex", type=int, default=None, help="cap textured galaxies (brightest first)")
    ap.add_argument("--tex-quality", type=int, default=88, help="embedded atlas JPEG quality")
    ap.add_argument("--out", default="docs/local_viewer_fork.html")
    args = ap.parse_args()
    import k3d
    if not hasattr(k3d, "textured_points"):
        sys.exit("k3d.textured_points not found — run with the fork on PYTHONPATH "
                 "(PYTHONPATH=~/Projects/echoes-k3d).")
    from PIL import Image
    rng = np.random.default_rng(0)

    gz = np.load(os.path.join(args.atlas_dir, "atlas_galaxies.npz"))
    m = json.load(open(os.path.join(args.atlas_dir, "atlas_manifest.json")))
    have = np.where(gz["atlas_tile_index"] >= 0)[0]
    if args.max_tex and len(have) > args.max_tex:        # atlas_galaxies.npz is brightest-first
        have = have[:args.max_tex]
    xyz = gz["xyz"][have]
    dist = gz["dist_mpc"][have].astype(float)
    diam = (gz["ang_size_arcmin"][have].astype(float) * ARCMIN2RAD * dist).astype(np.float32)  # Mpc, true
    ba = np.clip(gz["b_a"][have].astype(float), 0.15, 1.0).astype(np.float32)
    pa = np.radians(gz["pa_deg"][have].astype(float)).astype(np.float32)
    tile = gz["atlas_tile_index"][have].astype(int)
    rects = _tile_rects(tile, m)
    sheet_of = tile // m["tiles_per_sheet"]

    plot = k3d.plot(grid_visible=False, camera_auto_fit=True, snapshot_type="full",
                    name="ECHOES local-neighborhood (textured, true 3D)")
    # point-cloud context (observed + completed) — the far-field LOD
    obs = np.load(os.path.join(LOCAL, "local_2mpp_observed.npz"))
    ip = np.load(os.path.join(LOCAL, f"local_2mpp_{args.mode}_mcmc{args.mcmc}.npz"), allow_pickle=True)
    obs_xyz = _xyz(obs); ip_xyz = _xyz(ip)
    kind = ip["kind"].astype(str) if "kind" in ip.files else np.full(len(ip_xyz), "zoa")
    plot += k3d.points(_sample(obs_xyz, args.n_obs, rng), color=0x9aa0a6, point_size=2.0,
                       shader="flat", name=f"observed 2M++ ({len(obs_xyz):,})")
    if (kind == "faint").any():
        plot += k3d.points(_sample(ip_xyz[kind == "faint"], args.n_faint, rng), color=0x3a78ff,
                           point_size=1.6, shader="flat", name="completed: faint-end")
    plot += k3d.points(_sample(ip_xyz[kind == "zoa"], args.n_faint, rng), color=0xff3b30,
                       point_size=2.4, shader="flat", name="completed: Zone of Avoidance")

    # one TexturedPoints per atlas sheet (camera-facing, true-scale galaxy images)
    n_tex = 0
    for s in range(m["n_sheets"]):
        sel = np.where(sheet_of == s)[0]
        if len(sel) == 0:
            continue
        img = Image.open(os.path.join(args.atlas_dir, m["sheets"][s]["file"])).convert("RGB")
        buf = io.BytesIO(); img.save(buf, "JPEG", quality=args.tex_quality)
        plot += k3d.textured_points(
            positions=xyz[sel], atlas_uv=rects[sel], sizes=diam[sel],
            axis_ratio=ba[sel], position_angle=pa[sel],
            texture=buf.getvalue(), texture_file_format="jpg", size_scale=args.size_scale,
            name=f"galaxy images (sheet {s}, {len(sel)})")
        n_tex += len(sel)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(plot.get_snapshot())
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB) — {n_tex:,} camera-facing "
          f"galaxy-image billboards (true scale x{args.size_scale}) + observed/completed points "
          f"(mcmc{args.mcmc}); self-contained fork snapshot.")


if __name__ == "__main__":
    main()
