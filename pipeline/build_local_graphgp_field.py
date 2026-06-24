"""Local-volume GraphGP field + Vecchia inpaint, generated on the JULIA backend, on the real 2M++
data — and validated to match the JAX path. The local volume is real-space comoving (Manticore/CF4
velocity-corrected), so the field is ISOTROPIC: no anisotropic K(Δθ,Δz) is needed here (that is a
BOSS observed-coordinate concern), and the Julia engine is a pure backend swap.

Two products + their validation:
  A. build_local_gp_field cube (lognormal Cox prior) via Julia, plus an engine-equivalence check:
     generate_field(jax) vs generate_field(julia) on the SAME shared grid graph + SAME xi → tight.
  B. field_posterior_vecchia inpaint conditioned on the real 2M++ galaxies via Julia, plus a
     correctness check vs gp_posterior_dense (the JAX dense reference) on a real-data subsample.

    OMP_NUM_THREADS=8 JAX_PLATFORMS=cpu python pipeline/build_local_graphgp_field.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import graphgp as gp

from echoes.surveys.twompp import read_2mpp
from echoes import graphgp_julia as ggj
from echoes import graphgp_backend as gb
from echoes.local_field_graphgp import build_local_gp_field, _grid_points
from echoes.field_posterior import gp_posterior_dense, field_posterior_vecchia

H0 = 68.1
OUT = os.path.join(os.path.dirname(__file__), "..", "data_release", "local_graphgp")
os.makedirs(OUT, exist_ok=True)


def load_2mpp_xyz(dmax):
    c = read_2mpp(os.path.join(os.path.dirname(__file__), "..",
                               "data", "local", "2mpp", "2mpp_catalog.fits"))
    d = c.vcmb / H0
    nhat = np.stack([np.cos(np.radians(c.dec)) * np.cos(np.radians(c.ra)),
                     np.cos(np.radians(c.dec)) * np.sin(np.radians(c.ra)),
                     np.sin(np.radians(c.dec))], axis=1)
    xyz = nhat * d[:, None]
    m = d < dmax
    return xyz[m], d[m]


def fiducial_kernel(sigma2=0.8, r0=6.0, alpha=1.4, rmax=120.0, n=600):
    """Isotropic log-field covariance K_g(r) for the lognormal Cox field — a fiducial galaxy
    clustering scale (r0~6 Mpc); the engine validation is kernel-agnostic."""
    r = np.linspace(0.0, rmax, n)
    return (r, sigma2 * np.exp(-((r / r0) ** alpha)))


def part_a(cov, box=300.0):
    print("\n=== A. build_local_gp_field cube (Julia) + engine-equivalence vs JAX ===", flush=True)
    # A1 — engine equivalence on the real local grid: ONE shared graph, same xi, jax vs julia.
    n_eq, n0, k = 32, 256, 30
    pts, _ = _grid_points(n_eq, box)
    work = os.path.join(OUT, "_eq_work"); os.makedirs(work, exist_ok=True)
    npz = os.path.join(work, "graph.npz")
    ggj.build_graph_npz(pts, n0, k, np.asarray(cov[0]), np.asarray(cov[1]), npz)
    d = dict(np.load(npz))
    scale = float(d["scale"])
    ref_graph = gp.Graph(points=jnp.asarray(scale * d["coords"].astype(np.float64)),
                         neighbors=jnp.asarray(d["neighbors"].astype(np.int32)),
                         offsets=tuple(int(x) for x in d["offsets"]),
                         indices=jnp.asarray(d["indices"].astype(np.int64)))
    rng = np.random.default_rng(0)
    xi = rng.standard_normal(len(pts))
    g_jax = gb.generate_field(pts, cov, xi, n0=n0, k=k, backend="jax", graph=ref_graph)
    g_jul = gb.generate_field(pts, cov, xi, n0=n0, k=k, backend="julia", graph_npz=npz, dtype="f64")
    rel = float(np.linalg.norm(g_jax - g_jul) / (np.linalg.norm(g_jax) + 1e-30))
    print(f"  engine equivalence (shared graph, same xi): L2-rel(jax,julia) = {rel:.3e}")

    # A2 — the actual product cube at full resolution, on Julia.
    n_grid = 64
    fc = build_local_gp_field(box, n_grid, cov, mode="prior", backend="julia",
                              device="gpu", build_in_julia=True, n0=256, k=30, seed=7)
    cube = fc.delta
    print(f"  product cube {cube.shape}: mean={cube.mean():.3f} var={cube.var():.3f} "
          f"min={cube.min():.3f} max={cube.max():.3f}  positive={bool((cube>0).all())}")
    np.savez_compressed(os.path.join(OUT, "local_2mpp_graphgp_prior_cube.npz"),
                        delta=cube.astype(np.float32), box_mpc=np.float64(box),
                        backend="julia", kernel="strexp_r0_6_sigma2_0.8")
    return rel, cube


def part_b(cov, dmax=150.0, box=300.0):
    print("\n=== B. field_posterior_vecchia inpaint (Julia) + validation vs dense (JAX) ===", flush=True)
    xyz, dist = load_2mpp_xyz(dmax)
    print(f"  conditioning on {len(xyz):,} real 2M++ galaxies (d<{dmax:.0f} Mpc)", flush=True)

    # B1 — correctness vs the dense JAX reference on a real-data subsample (exact Vecchia limit).
    rng = np.random.default_rng(1)
    sub = rng.choice(len(xyz), 200, replace=False)
    x_data = xyz[sub]
    x_pred = xyz[rng.choice(len(xyz), 150, replace=False)] + rng.normal(0, 3.0, (150, 3))
    nbar = np.full(len(x_data), 0.3)              # fiducial local mean density (gal / Mpc^3 scale)
    y = 1.0 / nbar - 1.0
    nv = 1.0 / nbar
    Nj = len(x_data) + len(x_pred)
    m_d, _ = gp_posterior_dense(x_data, y, nv, x_pred, cov)
    m_v = field_posterior_vecchia(x_data, y, nv, x_pred, cov, n0=Nj - 1, k=Nj - 1, cg_tol=1e-10)
    rel = float(np.linalg.norm(m_v - m_d) / (np.linalg.norm(m_d) + 1e-30))
    print(f"  posterior mean vs dense (real-data subsample, exact k=N-1): L2-rel = {rel:.3e}")

    # B2 — the product: a conditional Matheron inpaint cube on Julia, conditioned on the galaxies.
    n_grid = 48
    nbar_all = np.full(len(xyz), 0.3)
    fc = build_local_gp_field(box, n_grid, cov, mode="posterior_sample",
                              points_data=xyz, nbar_data=nbar_all, backend="julia",
                              device="gpu", build_in_julia=True, n0=256, k=30, seed=3)
    cube = fc.delta
    print(f"  inpaint cube {cube.shape}: mean={cube.mean():.3f} "
          f"min={cube.min():.3f} max={cube.max():.3f}  positive={bool((cube>0).all())}", flush=True)
    np.savez_compressed(os.path.join(OUT, "local_2mpp_graphgp_inpaint_cube.npz"),
                        delta=cube.astype(np.float32), box_mpc=np.float64(box),
                        n_conditioned=int(len(xyz)), backend="julia")
    return rel, cube


def main():
    import time
    t = {}
    t0 = time.perf_counter()
    cov = fiducial_kernel()
    ta = time.perf_counter(); relA, _ = part_a(cov); t["A_prior_cube"] = time.perf_counter() - ta
    tb = time.perf_counter(); relB, _ = part_b(cov); t["B_posterior_inpaint"] = time.perf_counter() - tb
    total = time.perf_counter() - t0
    print("\n=== SUMMARY ===")
    print(f"  A engine equivalence (jax vs julia, shared graph): {relA:.3e}  "
          f"[{'PASS' if relA < 1e-5 else 'FAIL'}]")
    print(f"  B posterior vs dense JAX (real 2M++ subsample):    {relB:.3e}  "
          f"[{'PASS' if relB < 1e-3 else 'FAIL'}]")
    print(f"  TIMING  A(prior 64^3 cube)={t['A_prior_cube']:.1f}s  "
          f"B(posterior 48^3 inpaint, 36.9k galaxies)={t['B_posterior_inpaint']:.1f}s  "
          f"TOTAL={total:.1f}s")
    print(f"  products written under {os.path.relpath(OUT)}")


if __name__ == "__main__":
    main()
