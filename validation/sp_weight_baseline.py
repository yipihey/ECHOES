"""G4 — ISD vs a Rezaie-style NN-weight baseline for SP decontamination.

Rezaie et al. (2020, MNRAS 495, 1613; DECaLS imaging-systematics mitigation) regress
the observed galaxy density on the imaging SP templates with a feed-forward neural
net — a JOINT, MULTIVARIATE, nonlinear selection model — and weight each galaxy by
1/F_NN(SP). ECHOES uses ISD (echoes.systematics.isd_fit): iterative, UNIVARIATE,
one template at a time. This head-to-head asks whether ISD's simpler scheme is
competitive: does it remove as much SP dependence as the NN?

Both produce per-galaxy weights; we measure the residual density-vs-SP χ²/dof
(jackknife covariance, the G1 statistic) per template after each. The NN is a small
MLP (5→16→16→1, softplus output) trained in JAX/optax on pixel density vs pixel SP.

PASS: ISD residual ≤ NN-weight residual (within ~30%) on every template — i.e. the
data-driven univariate scheme is as clean as the multivariate NN.

    OMP_NUM_THREADS=16 JAX_PLATFORMS=cpu ~/.venv/k3d/bin/python3 \
        validation/sp_weight_baseline.py
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from echoes.surveys.boss import load_boss
from echoes.sp_maps import load_sp_maps, isd_decontamination, _pix, NSIDE_SP
from echoes.systematics import density_vs_template_jk, _chi2_flat, JackknifeMap

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"


def train_nn_selection(X_pix, y_pix, w_pix, *, seed=0, epochs=1500, lr=5e-3, hidden=16):
    """Small MLP F_NN(SP)→(1+δ), trained (weighted MSE) on pixel density vs SP.
    Returns a vectorised predictor on standardised features. Pure JAX + optax."""
    import jax, jax.numpy as jnp
    import optax
    jax.config.update("jax_enable_x64", False)
    key = jax.random.PRNGKey(seed)
    nf = X_pix.shape[1]
    def init(key):
        k1, k2, k3 = jax.random.split(key, 3)
        sc = lambda k, a, b: jax.random.normal(k, (a, b)) * np.sqrt(2.0 / a)
        return dict(W1=sc(k1, nf, hidden), b1=jnp.zeros(hidden),
                    W2=sc(k2, hidden, hidden), b2=jnp.zeros(hidden),
                    W3=sc(k3, hidden, 1), b3=jnp.zeros(1))
    def fwd(p, X):
        h = jnp.tanh(X @ p["W1"] + p["b1"])
        h = jnp.tanh(h @ p["W2"] + p["b2"])
        return jax.nn.softplus(h @ p["W3"] + p["b3"])[:, 0]      # > 0
    Xj, yj, wj = jnp.asarray(X_pix), jnp.asarray(y_pix), jnp.asarray(w_pix)
    def loss(p):
        pred = fwd(p, Xj)
        return jnp.sum(wj * (pred - yj) ** 2) / jnp.sum(wj)
    p = init(key)
    opt = optax.adam(lr); st = opt.init(p)
    gl = jax.jit(jax.value_and_grad(loss))
    for _ in range(epochs):
        _, g = gl(p); upd, st = opt.update(g, st); p = optax.apply_updates(p, upd)
    return lambda Xq: np.asarray(fwd(p, jnp.asarray(Xq)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nside", type=int, default=NSIDE_SP)
    p.add_argument("--n-jk", type=int, default=48)
    p.add_argument("--n-bins", type=int, default=10)
    args = p.parse_args()

    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    ra = np.asarray(cat.ra_data); dec = np.asarray(cat.dec_data)
    rar = np.asarray(cat.ra_random); decr = np.asarray(cat.dec_random)
    sp = load_sp_maps(RAND, nside=args.nside, verbose=False)
    nm_list = sp.names
    print(f"templates: {nm_list}")

    # ---- pixel-level training data: density (1+δ) vs SP features ----
    npix = 12 * args.nside ** 2
    ng = np.bincount(_pix(ra, dec, args.nside), minlength=npix).astype(float)
    nr = np.bincount(_pix(rar, decr, args.nside), minlength=npix).astype(float)
    foot = nr > 0
    alpha = ng[foot].sum() / nr[foot].sum()
    opd_pix = ng[foot] / (alpha * nr[foot])                      # (1+δ) per footprint pixel
    Xall = np.column_stack([sp.maps[nm] for nm in nm_list])      # (npix, nf)
    mu = Xall[foot].mean(0); sg = Xall[foot].std(0) + 1e-9
    Xpix = (Xall[foot] - mu) / sg
    Fnn = train_nn_selection(Xpix, opd_pix, nr[foot])            # predictor on standardised SP

    # ---- per-galaxy weights: NN (1/F_NN) and ISD ----
    Xg = (sp.stack_at(ra, dec) - mu) / sg
    w_nn = 1.0 / np.clip(Fnn(Xg), 0.2, 5.0)
    w_nn *= len(w_nn) / w_nn.sum()                               # normalise mean→1
    isd = isd_decontamination(cat, sp)
    w_isd = isd.weight

    # ---- residual density-vs-SP χ²/dof (jackknife), same machinery as G1 ----
    jk = JackknifeMap(rar, decr, n_reg=args.n_jk)
    reg_g = jk.assign(ra, dec); reg_r = jk.assign(rar, decr)
    sp_r = {nm: sp.at(rar, decr, nm) for nm in nm_list}
    sp_g = {nm: sp.at(ra, dec, nm) for nm in nm_list}
    edges = {nm: np.quantile(sp_r[nm], np.linspace(0, 1, args.n_bins + 1)) for nm in nm_list}
    for nm in nm_list:
        edges[nm][0] -= 1e-9; edges[nm][-1] += 1e-9

    def resid(w):
        out = {}
        for nm in nm_list:
            F, s, ok = density_vs_template_jk(sp_g[nm], sp_r[nm], edges[nm], reg_g, reg_r, w_data=w)
            out[nm] = _chi2_flat(F, s, ok)
        return out
    chi_none = resid(np.ones(len(ra)))
    chi_isd = resid(w_isd)
    chi_nn = resid(w_nn)

    print(f"\n=== residual density-vs-SP χ²/dof (jackknife; ≈1 = clean) ===")
    print(f"{'template':12s} {'none':>8s} {'ISD':>8s} {'NN(Rezaie)':>12s}")
    n_isd_ok = 0
    for nm in nm_list:
        flag = "" if chi_isd[nm] <= 1.3 * max(chi_nn[nm], 1.0) else "  <-ISD worse"
        n_isd_ok += (chi_isd[nm] <= 1.3 * max(chi_nn[nm], 1.0))
        print(f"{nm:12s} {chi_none[nm]:8.2f} {chi_isd[nm]:8.2f} {chi_nn[nm]:12.2f}{flag}")
    print(f"\nmean χ²/dof:  none={np.mean(list(chi_none.values())):.2f}  "
          f"ISD={np.mean(list(chi_isd.values())):.2f}  NN={np.mean(list(chi_nn.values())):.2f}")
    print(f"G4 {'PASS' if n_isd_ok == len(nm_list) else 'CHECK'}: ISD is "
          f"{'competitive with' if n_isd_ok == len(nm_list) else 'NOT uniformly ≤'} the NN-weight "
          f"baseline ({n_isd_ok}/{len(nm_list)} templates within 30%)")


if __name__ == "__main__":
    main()
