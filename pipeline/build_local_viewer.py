"""Self-contained interactive 3D viewer of the true-3D local-neighborhood ECHOES product.

Renders the observed 2M++ galaxies + one realization's completion (Zone-of-Avoidance fills and
faint-end restorations) in equatorial comoving Cartesian [Mpc], coloured by provenance, as a
standalone HTML (k3d snapshot — orbit/zoom in the browser, toggle layers). The ZoA fills are
highlighted: galaxies the survey cannot see behind the Milky Way, reconstructed in true 3D.

    JAX_PLATFORMS=cpu python pipeline/build_local_viewer.py --mcmc 0 --out docs/local_viewer.html
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

LOCAL = os.path.join("data_release", "local")


def _xyz(d):
    ra = np.radians(d["ra"].astype(float)); dec = np.radians(d["dec"].astype(float))
    r = d["dist_mpc"].astype(float); cd = np.cos(dec)
    return np.column_stack([r * cd * np.cos(ra), r * cd * np.sin(ra), r * np.sin(dec)]).astype(np.float32)


def _sample(xyz, n, rng):
    if len(xyz) <= n:
        return xyz
    return xyz[rng.choice(len(xyz), n, replace=False)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcmc", type=int, default=0)
    ap.add_argument("--mode", default="full")
    ap.add_argument("--n-obs", type=int, default=60000, help="observed points to show")
    ap.add_argument("--n-faint", type=int, default=60000, help="faint-completion points to show")
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

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(plot.get_snapshot())
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB) — "
          f"observed {len(obs_xyz):,}, ZoA {len(zoa_xyz):,}, faint {len(faint_xyz):,} "
          f"(mcmc{args.mcmc}); open in a browser to orbit the local universe in true 3D.")


if __name__ == "__main__":
    main()
