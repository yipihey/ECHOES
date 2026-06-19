"""NIFTy (nifty8.re) engine stand-up: correlated field + selection mask + Poisson.

Validates the second field engine on a 1D toy: a log-normal intensity
exp(s(x)) with a LEARNED power spectrum (CorrelatedFieldMaker), observed through
a selection mask with a hole, Poisson counts; geoVI (optimize_kl) recovers the
field and fills the hole by the learned correlations. Reports the correlation
between the posterior-mean field and the truth in the observed region and the
hole. This is the NIFTy analogue of the graphGP conditional solve.

    JAX_PLATFORMS=cpu ~/.venv/k3d/bin/python3 validation/nifty_toy.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import jax, jax.numpy as jnp
import jax.random as random
import nifty8.re as jft


def main():
    N = 256
    cfm = jft.CorrelatedFieldMaker("cf")
    cfm.set_amplitude_total_offset(offset_mean=0.0, offset_std=(1e-1, 1e-2))
    cfm.add_fluctuations((N,), distances=1.0 / N, fluctuations=(1.0, 0.5),
                         loglogavgslope=(-4.0, 0.5), prefix="")
    cf = cfm.finalize()

    key = random.PRNGKey(0)
    k_truth, k_data, k_opt = random.split(key, 3)
    true_latent = cf.init(k_truth)
    true_field = np.asarray(cf(true_latent))            # log-intensity
    mask = np.ones(N, bool); mask[110:150] = False     # interior hole (no data)
    obs_idx = jnp.asarray(np.flatnonzero(mask))
    lam_full = np.exp(true_field)
    data_full = np.asarray(random.poisson(k_data, jnp.asarray(lam_full)))
    data_obs = data_full[np.asarray(obs_idx)].astype(np.int32)   # only observed voxels
    print(f"toy: N={N}, hole=[110,150), observed counts={int(data_obs.sum())}")

    # selection response: the likelihood sees lambda only at observed voxels;
    # the hole is reconstructed from the learned correlations (prior).
    signal = jft.Model(lambda x: jnp.exp(cf(x))[obs_idx],
                       domain=cf.domain, init=cf.init)
    lh = jft.Poissonian(jnp.asarray(data_obs)).amend(signal)

    pos0 = jft.Vector(cf.init(k_opt))
    samples, state = jft.optimize_kl(
        lh, pos0, key=k_opt, n_total_iterations=8, n_samples=4,
        sample_mode="nonlinear_resample")
    post = np.asarray(jnp.mean(jnp.stack([cf(s) for s in samples]), axis=0))
    spread = np.asarray(jnp.std(jnp.stack([cf(s) for s in samples]), axis=0))

    obs = mask > 0; hole = ~obs
    def corr(a, b): return float(np.corrcoef(a, b)[0, 1])
    print(f"posterior-mean field vs truth:  observed corr = {corr(post[obs], true_field[obs]):.3f}"
          f"   hole corr = {corr(post[hole], true_field[hole]):.3f}")
    print(f"posterior spread:  observed median = {np.median(spread[obs]):.3f}"
          f"   hole median = {np.median(spread[hole]):.3f}  (hole should be larger)")
    ok = (corr(post[obs], true_field[obs]) > 0.6
          and np.median(spread[hole]) > np.median(spread[obs]))
    print("NIFTy engine stand-up:", "PASS" if ok else "CHECK")


if __name__ == "__main__":
    main()
