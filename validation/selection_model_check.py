"""Self-consistency: does the generative selection model reproduce the mock?

The completion is only as trustworthy as the forward model. Here we check that
:class:`echoes.selection_model.SelectionModel` (the explicit written-down model)
predicts the same observed fraction that the mock simulator
(:func:`echoes.mock_systematics.apply_survey_systematics`) actually produces, as a
function of the drivers it couples to — WEIGHT_SYSTOT, the local close-pair count
(density), and i-band magnitude. Agreement means the mock is this model's own
simulator, so inject-and-recover is self-consistent.

    JAX_PLATFORMS=cpu OMP_NUM_THREADS=16 ~/.venv/k3d/bin/python3 \
        validation/selection_model_check.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from echoes.surveys.boss import load_boss
from echoes.mock_systematics import apply_survey_systematics
from echoes.selection_model import SelectionModel, local_close_pair_count

DATA = "data/boss/galaxy_DR12v5_CMASS_South.fits.gz"
RAND = "data/boss/random0_DR12v5_CMASS_South.fits.gz"


def main():
    cat = load_boss([DATA], [RAND], sample="CMASS", nside=256, with_photometry=True)
    ra = np.asarray(cat.ra_data); dec = np.asarray(cat.dec_data); z = np.asarray(cat.z_data)
    colors = np.asarray(cat.colors_data); mags = np.asarray(cat.mags_data); wsys = np.asarray(cat.w_sys_data)

    coll_frac, zfail_frac, faint_bias = 0.6, 0.03, 1.5
    obs, tg, kept, _ = apply_survey_systematics(
        ra, dec, z, colors, mags, wsys, coll_frac=coll_frac,
        zfail_frac=zfail_frac, zfail_faint_bias=faint_bias, seed=0)
    observed = np.asarray(kept, bool)
    print(f"mock observed fraction = {observed.mean():.4f}  (N_obs={observed.sum():,}/{len(ra):,})")

    sm = SelectionModel(coll_frac=coll_frac, zfail_frac=zfail_frac, zfail_faint_bias=faint_bias)
    n_close = local_close_pair_count(ra, dec, sm.collision_scale_deg)
    imag = mags[:, 3]
    p_obs = sm.p_observed(wsys, n_close, imag=imag)
    print(f"model mean p_observed  = {p_obs.mean():.4f}")

    def table(name, driver, edges):
        which = np.clip(np.digitize(driver, edges) - 1, 0, len(edges) - 2)
        print(f"\n  observed fraction vs {name}:  bin   mock    model")
        for b in range(len(edges) - 1):
            m = which == b
            if m.sum() < 50:
                continue
            print(f"    {0.5*(edges[b]+edges[b+1]):8.3f}   {observed[m].mean():.3f}   {p_obs[m].mean():.3f}")

    table("WEIGHT_SYSTOT", wsys, np.quantile(wsys, np.linspace(0, 1, 7)))
    table("local close-pair count", n_close, np.array([-0.5, 0.5, 1.5, 2.5, 4.5, 20.5]))
    table("i-band mag", imag, np.quantile(imag, np.linspace(0, 1, 7)))
    # overall agreement
    print(f"\n  overall: mock {observed.mean():.4f}  vs  model {p_obs.mean():.4f}  "
          f"(Δ {100*(p_obs.mean()-observed.mean()):+.2f}pp)")
    print("\n(close agreement, esp. the density (close-pair) and systot trends, means the "
          "written-down model is the mock's own forward model — the collision term is an "
          "independent-neighbour approximation of the exact pairwise removal.)")


if __name__ == "__main__":
    main()
