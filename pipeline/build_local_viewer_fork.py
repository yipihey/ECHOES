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


def _context_menu_js(gz, have):
    """Injected JS for right-click → external-astronomy-links context menu on the textured galaxies.

    Embeds a compact per-galaxy table + the shared ``apps/echoes-viewer/astrolinks.js`` (single source
    of truth for the URL templates), projects each galaxy through the live k3d camera on
    ``contextmenu``, picks the nearest billboard within 18 px, and builds a dark menu of
    NED/SIMBAD/Aladin/Legacy links + local info + copy-coordinates. Runs inside the standalone
    snapshot's ``.then(function(K3DInstance){…})`` so ``K3DInstance`` is in scope."""
    al_path = os.path.join(os.path.dirname(__file__), "..", "apps", "echoes-viewer", "astrolinks.js")
    astrolinks_src = open(al_path).read()
    pgc = gz["pgc"][have] if "pgc" in gz.files else np.full(len(have), -1)
    gal = {
        "n": int(len(have)),
        "ra": [round(float(v), 5) for v in gz["ra"][have]],
        "dec": [round(float(v), 5) for v in gz["dec"][have]],
        "dist": [round(float(v), 1) for v in gz["dist_mpc"][have]],
        "ksmag": [round(float(v), 2) for v in gz["ksmag"][have]],
        "pgc": [int(v) for v in pgc],
        "xyz": [round(float(v), 3) for v in gz["xyz"][have].astype(float).ravel()],
    }
    gal_json = json.dumps(gal, separators=(",", ":"))
    return (
        "(function(){\n" + astrolinks_src + "\n"
        "var GAL=" + gal_json + ";var AL=self.AstroLinks;\n"
        "function m4(a,b){var o=new Array(16);for(var c=0;c<4;c++)for(var r=0;r<4;r++){var s=0;"
        "for(var k=0;k<4;k++)s+=a[k*4+r]*b[c*4+k];o[c*4+r]=s;}return o;}\n"
        "function menu(o,x,y){if(window.__emenu)window.__emenu.remove();"
        "var c=AL.formatCoord(o.ra,o.dec),nm=o.pgc>0?('PGC '+o.pgc):null,"
        "ls=AL.astroLinks({ra:o.ra,dec:o.dec,distMpc:o.dist,name:nm,pgc:o.pgc});"
        "var d=document.createElement('div');d.style.cssText='position:fixed;z-index:10000;"
        "min-width:240px;max-width:330px;background:rgba(11,11,15,0.97);border:1px solid #333;"
        "border-radius:8px;padding:9px;color:#ddd;font:12px -apple-system,sans-serif;"
        "box-shadow:0 6px 22px rgba(0,0,0,0.7)';"
        "var h='<div style=\"font-weight:600;margin-bottom:3px\">Galaxy'+(nm?(' \\u00b7 '+nm):'')+'</div>';"
        "h+='<div style=\"color:#9aa\">'+c.sexagesimalStr+'</div>';"
        "h+='<div style=\"color:#9aa;margin-bottom:6px\">'+o.dist.toFixed(1)+' Mpc \\u00b7 Ks '+o.ksmag.toFixed(2)+'</div>';"
        "var lg=null;ls.forEach(function(l){if(l.group!==lg){h+='<div style=\"color:#6cc;"
        "text-transform:uppercase;font-size:10px;letter-spacing:.04em;margin:5px 0 1px\">'+l.group+'</div>';lg=l.group;}"
        "h+='<a href=\"'+l.href+'\" target=\"_blank\" rel=\"noopener noreferrer\" style=\"display:block;"
        "color:#bdf;text-decoration:none;padding:3px 5px;border-radius:4px\">'+l.label+' \\u2197</a>';});"
        "h+='<button id=\"ecpy\" style=\"margin-top:7px;background:#222;color:#ccc;border:1px solid #444;"
        "border-radius:4px;cursor:pointer;padding:3px 9px\">Copy coordinates</button>';"
        "d.innerHTML=h;document.body.appendChild(d);var w=d.offsetWidth,hh=d.offsetHeight;"
        "d.style.left=Math.max(4,Math.min(x,innerWidth-w-8))+'px';"
        "d.style.top=Math.max(4,Math.min(y,innerHeight-hh-8))+'px';"
        "d.querySelector('#ecpy').onclick=function(){try{navigator.clipboard.writeText(c.sexagesimalStr);"
        "this.textContent='Copied';}catch(e){}};window.__emenu=d;}\n"
        "var world=K3DInstance.getWorld(),dom=world.targetDOMNode;\n"
        "dom.addEventListener('contextmenu',function(ev){ev.preventDefault();var cam=world.camera;"
        "if(!cam)return;cam.updateMatrixWorld();"
        "var vp=m4(cam.projectionMatrix.elements,cam.matrixWorldInverse.elements);"
        "var rect=dom.getBoundingClientRect(),W=rect.width,H=rect.height,"
        "cx=ev.clientX-rect.left,cy=ev.clientY-rect.top,best=-1,bd=324;"
        "for(var i=0;i<GAL.n;i++){var X=GAL.xyz[i*3],Y=GAL.xyz[i*3+1],Z=GAL.xyz[i*3+2];"
        "var ww=vp[3]*X+vp[7]*Y+vp[11]*Z+vp[15];if(ww<=0)continue;"
        "var xc=(vp[0]*X+vp[4]*Y+vp[8]*Z+vp[12])/ww,yc=(vp[1]*X+vp[5]*Y+vp[9]*Z+vp[13])/ww;"
        "var px=(xc*0.5+0.5)*W,py=(-yc*0.5+0.5)*H,dx=px-cx,dy=py-cy,d2=dx*dx+dy*dy;if(d2<bd){bd=d2;best=i;}}"
        "if(best<0)return;menu({ra:GAL.ra[best],dec:GAL.dec[best],dist:GAL.dist[best],"
        "ksmag:GAL.ksmag[best],pgc:GAL.pgc[best]},ev.clientX,ev.clientY);});\n"
        "document.addEventListener('pointerdown',function(ev){if(window.__emenu&&"
        "!window.__emenu.contains(ev.target))window.__emenu.remove();},true);\n"
        "document.addEventListener('keydown',function(ev){if(ev.key==='Escape'&&window.__emenu)"
        "window.__emenu.remove();});\n})();\n")


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
    ap.add_argument("--tex-downscale", type=int, default=1,
                    help="downscale each atlas sheet by this factor before embedding (UVs are "
                         "normalized, so this only trades tile resolution for HTML size + VRAM)")
    ap.add_argument("--veusz", default=None,
                    help="path (relative to the viewer HTML) to a .vsz figure to embed as a "
                         "collapsible browser-editable overlay (e.g. figs/local_completion.vsz)")
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
        if args.tex_downscale > 1:
            img = img.resize((img.width // args.tex_downscale, img.height // args.tex_downscale),
                             Image.LANCZOS)
        buf = io.BytesIO(); img.save(buf, "JPEG", quality=args.tex_quality)
        plot += k3d.textured_points(
            positions=xyz[sel], atlas_uv=rects[sel], sizes=diam[sel],
            axis_ratio=ba[sel], position_angle=pa[sel],
            texture=buf.getvalue(), texture_file_format="jpg", size_scale=args.size_scale,
            name=f"galaxy images (sheet {s}, {len(sel)})")
        n_tex += len(sel)

    # right-click a galaxy -> context menu of external astronomy links (NED/SIMBAD/Aladin/Legacy)
    # + local info. Injected JS projects the embedded galaxy table through the live k3d camera and
    # picks the nearest billboard; reuses the shared apps/echoes-viewer/astrolinks.js (single source).
    add_js = _context_menu_js(gz, have)
    # optional: inject a collapsible, browser-editable Veusz figure overlay (merge the Veusz work
    # into the viewer). The figure shows a static poster instantly and boots the WASM editor on click.
    if args.veusz:
        import json as _json
        from tools.veusz_vsz import embed_tag, EMBED_SCRIPT_VERSION
        fig_html = embed_tag(args.veusz, width=720, height=360)
        embed_src = f"https://yipihey.github.io/veusz/embed/{EMBED_SCRIPT_VERSION}/veusz-embed.js"
        add_js += (
            "(function(){"
            "var s=document.createElement('script');s.type='module';"
            f"s.src={_json.dumps(embed_src)};document.head.appendChild(s);"
            "var p=document.createElement('div');"
            "p.style.cssText='position:fixed;right:12px;bottom:12px;width:744px;max-width:46vw;"
            "background:rgba(11,11,15,0.93);border:1px solid #333;border-radius:8px;padding:6px;"
            "z-index:9999;box-shadow:0 4px 18px rgba(0,0,0,0.6)';"
            "p.innerHTML='<div style=\"display:flex;justify-content:space-between;align-items:center;"
            "color:#ccc;font:13px sans-serif;padding:2px 4px\"><b>Completion summary "
            "(click to edit)</b><button id=\"vzx\" style=\"background:#222;color:#ccc;border:1px "
            "solid #444;border-radius:4px;cursor:pointer\">hide</button></div>'+"
            f"{_json.dumps(fig_html)};"
            "document.body.appendChild(p);"
            "document.getElementById('vzx').onclick=function(){p.style.display='none';};"
            "})();")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(plot.get_snapshot(additional_js_code=add_js) if add_js else plot.get_snapshot())
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB) — {n_tex:,} camera-facing "
          f"galaxy-image billboards (true scale x{args.size_scale}) + observed/completed points "
          f"(mcmc{args.mcmc}); self-contained fork snapshot.")


if __name__ == "__main__":
    main()
