"""Provenance-aware visualizer for the ECHOES completed catalog.

The default monochrome render hides the whole point of ECHOES — you cannot see
that any completion happened. This tool draws colour from the per-object ``prov``
code so the three origins are separable, exactly as the data product stores them:

  * observed (spec-z)                 — the fixed base catalogue (dim teal),
  * completed: fiber-collision        — missing to a fiber collision, restored at
                                        its imaging position (orange),
  * completed: redshift-failure       — missing because its spectrum gave no
                                        redshift, restored from imaging (yellow),
  * inpainted: imaging-systematic     — a SYNTHETIC point added to undo an imaging
                                        density deficit, no imaging counterpart of
                                        its own (green).

``prov_rgba`` / ``prov_k3d_colors`` expose the SAME mapping (echoes.completion
PROV_COLOR) for any front-end — k3d, datashader, a web viewer — so every
visualizer separates the categories identically.

    # static raster (full footprint + a zoom panel), from the released product:
    ~/.venv/k3d/bin/python3 tools/viz_provenance.py \
        --package data_release/cmass_south_posterior.npz --seed 0 \
        --out output/provenance_map.png --zoom 12 22 -3 3
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from echoes.completion import PROV, PROV_NAME, PROV_GROUP, PROV_COLOR


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def prov_rgba(prov, *, alpha=1.0):
    """(N,4) float RGBA in [0,1] for each provenance code (echoes PROV_COLOR)."""
    prov = np.asarray(prov)
    out = np.zeros((len(prov), 4), np.float32); out[:, 3] = alpha
    for code, hexc in PROV_COLOR.items():
        m = prov == code
        if m.any():
            out[m, :3] = np.array(_hex_to_rgb(hexc), np.float32) / 255.0
    return out


def prov_k3d_colors(prov):
    """(N,) uint32 0xRRGGBB packed colours for k3d.points(colors=...)."""
    prov = np.asarray(prov); out = np.zeros(len(prov), np.uint32)
    for code, hexc in PROV_COLOR.items():
        r, g, b = _hex_to_rgb(hexc)
        out[prov == code] = (r << 16) | (g << 8) | b
    return out


def _group_of(prov):
    """Map per-object prov codes to coarse group labels (PROV_GROUP)."""
    return np.array([PROV_GROUP.get(int(p), "other") for p in prov])


# draw order: observed first (background), completions/inpaint painted on top so
# the few-percent restored points are never buried under the 91% observed base.
_ORDER = [PROV["observed"], PROV["systot"], PROV["inpaint"],
          PROV["zhost"], PROV["collided"], PROV["zfail"]]


def render(ra, dec, prov, out, *, zoom=None, dpi=150):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ra = np.asarray(ra); dec = np.asarray(dec); prov = np.asarray(prov)
    panels = [("full footprint", None)]
    if zoom is not None:
        panels.append((f"zoom RA[{zoom[0]},{zoom[1]}] DEC[{zoom[2]},{zoom[3]}]", zoom))

    fig, axes = plt.subplots(1, len(panels), figsize=(9.5 * len(panels), 8.6),
                             facecolor="black")
    axes = np.atleast_1d(axes)
    # sizes: dim small observed, bright larger completions/inpaint
    size = {PROV["observed"]: 0.6, PROV["systot"]: 3.0, PROV["inpaint"]: 3.0,
            PROV["collided"]: 3.5, PROV["zhost"]: 3.5, PROV["zfail"]: 5.0}
    alpha = {PROV["observed"]: 0.45}

    for ax, (title, zb) in zip(axes, panels):
        ax.set_facecolor("black")
        if zb is not None:
            sel = (ra > zb[0]) & (ra < zb[1]) & (dec > zb[2]) & (dec < zb[3])
        else:
            sel = np.ones(len(ra), bool)
        for code in _ORDER:
            m = sel & (prov == code)
            if not m.any():
                continue
            ax.scatter(ra[m], dec[m], s=size.get(code, 3.0),
                       c=[np.array(_hex_to_rgb(PROV_COLOR[code])) / 255.0],
                       alpha=alpha.get(code, 0.95), edgecolors="none", marker="s",
                       rasterized=True)
        ax.set_title(title, color="white", fontsize=12)
        ax.set_xlabel("RA [deg]", color="0.8"); ax.set_ylabel("DEC [deg]", color="0.8")
        ax.tick_params(colors="0.6"); ax.set_aspect("equal")
        for s in ax.spines.values():
            s.set_color("0.3")
        if zb is not None:
            ax.set_xlim(zb[0], zb[1]); ax.set_ylim(zb[2], zb[3])

    # one legend per coarse group, with census
    grp = _group_of(prov)
    handles, labels = [], []
    for code in _ORDER:
        g = PROV_GROUP.get(code)
        if g in labels:
            continue
        n = int((grp == g).sum())
        if n == 0:
            continue
        labels.append(g)
        handles.append(Line2D([0], [0], marker="s", linestyle="none",
                              markerfacecolor=np.array(_hex_to_rgb(PROV_COLOR[code])) / 255.0,
                              markeredgecolor="none", markersize=9,
                              label=f"{g}  (N={n:,}, {100*n/len(prov):.1f}%)"))
    leg = axes[0].legend(handles, [h.get_label() for h in handles], loc="upper right",
                         frameon=True, facecolor="0.1", edgecolor="0.3", fontsize=9,
                         labelcolor="white")
    fig.suptitle(f"ECHOES completed catalog by provenance  (N={len(prov):,})",
                 color="white", fontsize=14, y=0.995)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out, dpi=dpi, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    return out


def _load(args):
    if args.package:
        from echoes.posterior import load_package, draw
        cat = draw(load_package(args.package), seed=args.seed)
        return cat["ra"], cat["dec"], cat["prov"]
    if args.fits:
        from astropy.table import Table
        t = Table.read(args.fits)
        return np.asarray(t["RA"]), np.asarray(t["DEC"]), np.asarray(t["PROV"])
    raise SystemExit("give --package <npz> or --fits <catalog.fits>")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--package", help="posterior .npz (draws a realization)")
    p.add_argument("--fits", help="a drawn catalog FITS (RA,DEC,Z,PROV)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="output/provenance_map.png")
    p.add_argument("--zoom", type=float, nargs=4, default=None,
                   metavar=("RA0", "RA1", "DEC0", "DEC1"))
    args = p.parse_args()
    ra, dec, prov = _load(args)
    print(f"N={len(prov):,}  provenance census:")
    for code, n in zip(*np.unique(prov, return_counts=True)):
        print(f"  {int(code)} {PROV_NAME[int(code)]:9s} -> {PROV_GROUP.get(int(code)):30s} "
              f"{n:7d}  ({100*n/len(prov):.2f}%)")
    out = render(ra, dec, prov, args.out, zoom=tuple(args.zoom) if args.zoom else None)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
