"""Self-contained interactive 3D viewer of the true-3D local-neighborhood ECHOES product.

Renders the observed 2M++ galaxies + one realization's completion (Zone-of-Avoidance fills and
faint-end restorations) in equatorial comoving Cartesian [Mpc], coloured by provenance, as a
standalone HTML (k3d snapshot — orbit/zoom in the browser, toggle layers). The ZoA fills are
highlighted: galaxies the survey cannot see behind the Milky Way, reconstructed in true 3D.

    JAX_PLATFORMS=cpu python pipeline/build_local_viewer.py --mcmc 0 --out docs/local_viewer.html

With ``--atlas-dir docs/visualizer-local/data`` (the output of ``pipeline/build_texture_atlas.py``)
it also renders the real galaxy IMAGES as textured billboards: one atlas-textured ``k3d.mesh`` of
sky-tangent quads per atlas sheet, each quad placed at the galaxy's comoving position, sized by its
angular diameter, and oriented/inclined by its axis ratio + position angle. This is the stock-k3d
Stage-A demonstrator (fixed sky-tangent quads, exaggerated by ``--size-boost`` so they are visible
in a static snapshot); the custom-k3d fork adds true per-frame point→sprite→texture LOD.
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
    if len(xyz) <= n:
        return xyz
    return xyz[rng.choice(len(xyz), n, replace=False)]


def _tile_uv(idx, m):
    """Atlas (u0,v0,u1,v1) for tile ``idx`` given the manifest geometry ``m`` (gutter-inset)."""
    per_row = m["tiles_per_row"]; tile = m["tile_px"]; sheet = m["sheet_px"]; g = m["gutter_px"]
    within = idx % m["tiles_per_sheet"]
    row, col = divmod(within, per_row)
    u0 = (col * tile + g) / sheet; v0 = (row * tile + g) / sheet
    u1 = (col * tile + tile - g) / sheet; v1 = (row * tile + tile - g) / sheet
    return u0, v0, u1, v1


def _tangent_basis(xyz):
    """Per-galaxy sky-tangent frame: radial n̂ (line of sight), e_east, e_north (equatorial)."""
    n = xyz / np.clip(np.linalg.norm(xyz, axis=1, keepdims=True), 1e-9, None)
    zaxis = np.broadcast_to([0.0, 0.0, 1.0], n.shape)
    east = np.cross(zaxis, n)                                  # east ∝ ẑ × n̂
    en = np.linalg.norm(east, axis=1, keepdims=True)
    bad = en[:, 0] < 1e-6                                      # near the poles ẑ∥n̂ → degenerate
    east = np.where(en > 1e-9, east / np.clip(en, 1e-9, None), [1.0, 0.0, 0.0])
    if bad.any():                                             # fall back to x̂ × n̂ at the poles
        alt = np.cross(np.broadcast_to([1.0, 0.0, 0.0], n[bad].shape), n[bad])
        east[bad] = alt / np.clip(np.linalg.norm(alt, axis=1, keepdims=True), 1e-9, None)
    north = np.cross(n, east)
    return n, east, north


def _textured_meshes(plot, atlas_dir, size_boost, max_tex, rng):
    """Add one atlas-textured k3d.mesh of sky-tangent galaxy-image quads per atlas sheet."""
    import k3d
    from PIL import Image
    gz = np.load(os.path.join(atlas_dir, "atlas_galaxies.npz"))
    m = json.load(open(os.path.join(atlas_dir, "atlas_manifest.json")))
    have = np.where(gz["atlas_tile_index"] >= 0)[0]
    if max_tex and len(have) > max_tex:                       # brightest-first (already sorted)
        have = have[:max_tex]
    xyz = gz["xyz"][have].astype(np.float32)
    dist = gz["dist_mpc"][have].astype(float)
    ang = gz["ang_size_arcmin"][have].astype(float)
    b_a = np.clip(gz["b_a"][have].astype(float), 0.15, 1.0)
    pa = np.radians(gz["pa_deg"][have].astype(float))
    tile_idx = gz["atlas_tile_index"][have].astype(int)

    n, east, north = _tangent_basis(xyz)
    # physical half-size [Mpc] along the major axis (exaggerated for the static demonstrator)
    half = (0.5 * ang * ARCMIN2RAD * dist * size_boost)[:, None]
    major = np.cos(pa)[:, None] * north + np.sin(pa)[:, None] * east       # PA from N→E
    minor = (-np.sin(pa)[:, None] * north + np.cos(pa)[:, None] * east) * b_a[:, None]
    hmaj = major * half; hmin = minor * half
    # 4 corners (TL,TR,BR,BL): major≈image-up (v), minor≈image-right (u)
    c_tl = xyz + hmaj - hmin; c_tr = xyz + hmaj + hmin
    c_br = xyz - hmaj + hmin; c_bl = xyz - hmaj - hmin

    sheet_of = tile_idx // m["tiles_per_sheet"]
    for s in range(m["n_sheets"]):
        sel = np.where(sheet_of == s)[0]
        if len(sel) == 0:
            continue
        V = np.empty((len(sel) * 4, 3), np.float32)
        UV = np.empty((len(sel) * 4, 2), np.float32)
        F = np.empty((len(sel) * 2, 3), np.uint32)
        for j, i in enumerate(sel):
            u0, v0, u1, v1 = _tile_uv(int(tile_idx[i]), m)
            base = j * 4
            V[base + 0] = c_tl[i]; V[base + 1] = c_tr[i]
            V[base + 2] = c_br[i]; V[base + 3] = c_bl[i]
            UV[base + 0] = (u0, v0); UV[base + 1] = (u1, v0)
            UV[base + 2] = (u1, v1); UV[base + 3] = (u0, v1)
            F[j * 2 + 0] = (base + 0, base + 1, base + 2)
            F[j * 2 + 1] = (base + 0, base + 2, base + 3)
        img = Image.open(os.path.join(atlas_dir, m["sheets"][s]["file"])).convert("RGB")
        buf = io.BytesIO(); img.save(buf, "JPEG", quality=88)   # k3d texture: jpg bytes
        plot += k3d.mesh(V, F, uvs=UV.astype(np.float32), texture=buf.getvalue(),
                         texture_file_format="jpg", flat_shading=False,
                         name=f"galaxy images (sheet {s}, {len(sel)})")
    return len(have)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcmc", type=int, default=0)
    ap.add_argument("--mode", default="full")
    ap.add_argument("--n-obs", type=int, default=60000, help="observed points to show")
    ap.add_argument("--n-faint", type=int, default=60000, help="faint-completion points to show")
    ap.add_argument("--atlas-dir", default=None,
                    help="texture-atlas dir (build_texture_atlas.py output) → add image billboards")
    ap.add_argument("--size-boost", type=float, default=30.0,
                    help="exaggerate textured-quad size for the static demonstrator (1=true scale)")
    ap.add_argument("--max-tex", type=int, default=4000, help="cap textured galaxies in the snapshot")
    ap.add_argument("--out", default="docs/local_viewer.html")
    args = ap.parse_args()
    import k3d
    rng = np.random.default_rng(0)

    obs = np.load(os.path.join(LOCAL, "local_2mpp_observed.npz"))
    ip = np.load(os.path.join(LOCAL, f"local_2mpp_{args.mode}_mcmc{args.mcmc}.npz"), allow_pickle=True)
    obs_xyz = _xyz(obs)
    ip_xyz = _xyz(ip)
    kind = ip["kind"].astype(str) if "kind" in ip.files else np.full(len(ip_xyz), "zoa")
    zoa_xyz = ip_xyz[kind == "zoa"]; faint_xyz = ip_xyz[kind == "faint"]

    plot = k3d.plot(grid_visible=False, camera_auto_fit=True,
                    name="ECHOES local-neighborhood (true 3D)")
    plot += k3d.points(_sample(obs_xyz, args.n_obs, rng), color=0x9aa0a6, point_size=2.2,
                       shader="flat", name=f"observed 2M++ ({len(obs_xyz):,})")
    if len(faint_xyz):
        plot += k3d.points(_sample(faint_xyz, args.n_faint, rng), color=0x3a78ff, point_size=1.8,
                           shader="flat", name=f"completed: faint-end ({len(faint_xyz):,})")
    plot += k3d.points(_sample(zoa_xyz, args.n_faint, rng), color=0xff3b30, point_size=2.6,
                       shader="flat", name=f"completed: Zone of Avoidance ({len(zoa_xyz):,})")

    n_tex = 0
    if args.atlas_dir:
        n_tex = _textured_meshes(plot, args.atlas_dir, args.size_boost, args.max_tex, rng)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(plot.get_snapshot())
    extra = f", {n_tex:,} galaxy-image billboards" if n_tex else ""
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB) — "
          f"observed {len(obs_xyz):,}, ZoA {len(zoa_xyz):,}, faint {len(faint_xyz):,}{extra} "
          f"(mcmc{args.mcmc}); open in a browser to orbit the local universe in true 3D.")


if __name__ == "__main__":
    main()
