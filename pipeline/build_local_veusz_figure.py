"""Generate a browser-editable Veusz figure summarising the local-neighborhood completion, for the
textured 3D viewer's overlay (the "merge Veusz into the viewer" step).

Two panels, from the released local catalog (`data_release/local/`):
  1. galaxy counts vs comoving distance — observed 2M++ vs the ZoA + faint-end completion (the
     true-3D completion the viewer shows);
  2. K_s apparent-magnitude histogram — observed vs faint-restored (the flux-limit completion).

Self-contained `.vsz` (data embedded via `ImportString`, the form Veusz itself writes), rendered and
edited in the browser by the hosted WASM embed (`tools/veusz_vsz.embed_tag`). Output:
`docs/figs/local_completion.vsz` (+ a matplotlib `poster.png` so the panel shows instantly).

    JAX_PLATFORMS=cpu ~/.venv/k3d/bin/python3 pipeline/build_local_veusz_figure.py [--mcmc 0]
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from tools.veusz_vsz import Series, Panel, grid

LOCAL = os.path.join("data_release", "local")
OUT_VSZ = os.path.join("docs", "figs", "local_completion.vsz")


def _counts(values, edges):
    h, _ = np.histogram(values, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, h.astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcmc", type=int, default=0)
    ap.add_argument("--mode", default="full")
    ap.add_argument("--out", default=OUT_VSZ)
    args = ap.parse_args()

    obs = np.load(os.path.join(LOCAL, "local_2mpp_observed.npz"))
    ip = np.load(os.path.join(LOCAL, f"local_2mpp_{args.mode}_mcmc{args.mcmc}.npz"), allow_pickle=True)
    kind = ip["kind"].astype(str) if "kind" in ip.files else np.full(len(ip["dist_mpc"]), "zoa")

    # Panel 1: N(d) in 10-Mpc shells — observed vs ZoA-completed vs faint-completed
    dedges = np.arange(0.0, 305.0, 10.0)
    dc, n_obs = _counts(obs["dist_mpc"], dedges)
    _, n_zoa = _counts(ip["dist_mpc"][kind == "zoa"], dedges)
    _, n_faint = _counts(ip["dist_mpc"][kind == "faint"], dedges)
    p1 = Panel(
        series=[
            Series(dc, n_obs, label="observed 2M++", color="#9aa0a6", marker="circle", line=True),
            Series(dc, n_zoa, label="completed: ZoA", color="#ff3b30", marker="square", line=True),
            Series(dc, n_faint, label="completed: faint-end", color="#3a78ff", marker="triangle",
                   line=True),
        ],
        xlabel="comoving distance [Mpc]", ylabel="galaxies / 10 Mpc shell",
        title="True-3D completion vs distance", xrange=(0, 300))

    # Panel 2: K_s histogram — observed vs faint-restored (the flux-limit completion)
    kedges = np.arange(3.0, 16.01, 0.5)
    kc, k_obs = _counts(obs["ksmag"], kedges)
    kf = ip["ksmag"][kind == "faint"]
    _, k_faint = _counts(kf[np.isfinite(kf)], kedges)
    p2 = Panel(
        series=[
            Series(kc, k_obs, label="observed", color="#9aa0a6", marker="none", line_only=True),
            Series(kc, k_faint, label="faint-restored", color="#3a78ff", marker="none",
                   line_only=True, line_style="dashed"),
        ],
        xlabel="K_s apparent magnitude", ylabel="galaxies / 0.5 mag",
        title="Flux-limit completion (K_s)", xrange=(3, 16))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    grid(args.out, [p1, p2], rows=1, cols=2, width="30cm", height="13cm")
    print(f"wrote {args.out}  (observed {int(n_obs.sum()):,}; ZoA {int(n_zoa.sum()):,}; "
          f"faint {int(n_faint.sum()):,})")

    # a static poster PNG so the embed shows instantly (before the WASM engine boots)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
        a1.plot(dc, n_obs, "-o", color="#9aa0a6", ms=3, label="observed 2M++")
        a1.plot(dc, n_zoa, "-s", color="#ff3b30", ms=3, label="completed: ZoA")
        a1.plot(dc, n_faint, "-^", color="#3a78ff", ms=3, label="completed: faint-end")
        a1.set_xlabel("comoving distance [Mpc]"); a1.set_ylabel("galaxies / 10 Mpc shell")
        a1.set_title("True-3D completion vs distance"); a1.set_xlim(0, 300); a1.legend(fontsize=8)
        a2.plot(kc, k_obs, color="#9aa0a6", label="observed")
        a2.plot(kc, k_faint, "--", color="#3a78ff", label="faint-restored")
        a2.set_xlabel("K_s apparent magnitude"); a2.set_ylabel("galaxies / 0.5 mag")
        a2.set_title("Flux-limit completion (K_s)"); a2.set_xlim(3, 16); a2.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(args.out[:-4] + ".png", dpi=96); plt.close(fig)
        print(f"wrote {args.out[:-4]}.png (poster)")
    except Exception as e:                                  # poster is optional
        print(f"[warn] poster not written: {e}")


if __name__ == "__main__":
    main()
