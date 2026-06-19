"""fieldpost engine: the conditional field overdensity localizes structure on a
known sightline (uses a hand-built FieldContext, no ξ-measurement / Corrfunc)."""
import numpy as np
import pytest

pytest.importorskip("jax")

from echoes.geometry import _radec_to_nhat
from echoes.clustering import comoving_mpc_h
from echoes.fieldpost import FieldContext, los_overdensity


def test_los_overdensity_localizes_a_clump():
    rng = np.random.default_rng(0)
    cov = (np.linspace(0, 50, 500), np.exp(-(np.linspace(0, 50, 500) / 6.0) ** 1.5))
    ra0, dec0, z_clump = 150.0, 10.0, 0.55
    nhat0 = _radec_to_nhat(np.array([ra0]), np.array([dec0]))[0]
    chi0 = float(comoving_mpc_h(np.array([z_clump]))[0])
    # an overdense clump on the sightline near z_clump + sparse background around it
    clump = chi0 * nhat0[None, :] + rng.normal(0, 3.0, (90, 3))
    bg_dir = _radec_to_nhat(rng.uniform(148, 152, 400), rng.uniform(8, 12, 400))
    bg = comoving_mpc_h(rng.uniform(0.45, 0.65, 400))[:, None] * bg_dir
    x_obs = np.vstack([clump, bg])
    nhat_obs = x_obs / np.linalg.norm(x_obs, axis=1, keepdims=True)
    fc = FieldContext(
        x_obs=x_obs, nhat_obs=nhat_obs, cov=cov, nbar=len(x_obs) / 1e9,
        z_centres=np.linspace(0.45, 0.65, 8), nz_profile=np.ones(8),
        neigh_chord=2.0 * np.sin(np.radians(2.0) / 2.0), max_neigh=300)

    zgrid = np.linspace(0.45, 0.65, 120)
    opd = los_overdensity(fc, np.array([ra0]), np.array([dec0]), zgrid)
    assert opd.shape == (1, 120)
    o = opd[0]
    # 1+δ peaks at the clump redshift and is well above the off-clump baseline
    z_peak = zgrid[o.argmax()]
    assert abs(z_peak - z_clump) < 0.02
    far = np.abs(zgrid - z_clump) > 0.08
    assert o.max() > 3.0 * np.median(o[far])
    assert np.all(np.isfinite(o)) and np.all(o >= 0)
