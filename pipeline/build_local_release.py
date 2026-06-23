"""Build the true-3D local-neighborhood ECHOES product (branch data/local-neighborhood).

The observed 2M++ catalogue (placed at true comoving distances via the Manticore peculiar-
velocity field) plus, for each Manticore posterior realization, a Zone-of-Avoidance completion
that fills the Galactic-plane hole with galaxies drawn from that realization's reconstructed 3D
density. The ensemble over realizations is the posterior product: the reconstruction's field
ensemble composed with the catalogue completion.

Writes to ``data_release/local/``:
  local_2mpp_observed.npz        observed 2M++ at true-3D positions (shared base; written once)
  local_2mpp_zoa_mcmc<i>.npz     per-realization ZoA completion (PROV=5; the seed-varying part)
  local_2mpp_manifest.json       realizations, counts, parameters

A completed catalog for realization i = observed + zoa_mcmc<i>. Equatorial comoving frame.

    JAX_PLATFORMS=cpu python pipeline/build_local_release.py --realizations 0 1 2
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from echoes.surveys.local import load_local_2mpp
from echoes.surveys.manticore import available_realizations, manticore_field_context
from echoes.local_completion import complete_local_zoa, complete_local

OUT = os.path.join("data_release", "local")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--realizations", type=int, nargs="+", default=None,
                    help="Manticore mcmc indices (default: all locally fetched)")
    ap.add_argument("--mode", choices=["zoa", "full"], default="full",
                    help="'zoa' (fill the Galactic-plane hole only; small) or 'full' (also "
                         "restore the faint galaxies below the flux limit everywhere; large)")
    ap.add_argument("--m-faint", type=float, default=-22.0, help="full-mode completion depth (M_K)")
    ap.add_argument("--k-lim", type=float, default=11.5, help="2M++ apparent K flux limit")
    ap.add_argument("--dmax", type=float, default=300.0, help="max distance [Mpc]")
    ap.add_argument("--zoa-deg", type=float, default=5.0)
    ap.add_argument("--intensity", choices=["transform", "bias"], default="transform",
                    help="'transform' (log-Gaussian, matches the observed density PDF / sharp "
                         "contrast) or 'bias' (legacy mean-matched power-law, smoother)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    reals = args.realizations or available_realizations()
    if not reals:
        raise SystemExit("no Manticore realizations fetched — run data/fetch_manticore.py")
    fields = {m: manticore_field_context(m) for m in reals}

    cat = load_local_2mpp(field_mcmc=reals[0], dmax_mpc=args.dmax)
    obs_path = os.path.join(OUT, "local_2mpp_observed.npz")
    np.savez_compressed(obs_path, ra=cat.ra_data, dec=cat.dec_data, dist_mpc=cat.dist_mpc,
                        cz=(cat.z_data * 299792.458).astype(np.float32), ksmag=cat.ksmag_data,
                        prov=np.zeros(cat.N_data, np.int8))
    print(f"observed 2M++: {cat.N_data:,} galaxies -> {os.path.basename(obs_path)}", flush=True)

    meta = []
    for m in reals:
        tag = "full" if args.mode == "full" else "zoa"
        if args.mode == "full":
            ip = complete_local(cat, fields[m], m_faint=args.m_faint, k_lim=args.k_lim,
                                zoa_deg=args.zoa_deg, dmax=args.dmax, intensity=args.intensity,
                                uncert_fields=list(fields.values()), seed=1000 + m)
            extra = {"absmag": ip["absmag"], "uncert": ip["uncert"],
                     "kind": ip["kind"].astype("S5")}
        else:
            ip = complete_local_zoa(cat, fields[m], zoa_deg=args.zoa_deg, dmax=args.dmax,
                                    intensity=args.intensity, seed=1000 + m)
            extra = {}
        p = os.path.join(OUT, f"local_2mpp_{tag}_mcmc{m}.npz")
        np.savez_compressed(p, ra=ip["ra"], dec=ip["dec"], dist_mpc=ip["dist_mpc"],
                            cz=ip["cz"], ksmag=ip["ksmag"], prov=ip["prov"], **extra)
        meta.append({"mcmc": int(m), "n_inpaint": int(len(ip["ra"])),
                     "file": os.path.basename(p)})
        print(f"  mcmc{m}: +{len(ip['ra']):,} {tag} galaxies -> {os.path.basename(p)}", flush=True)

    manifest = {
        "product": f"local_2mpp_true3d_{args.mode}",
        "description": "True-3D ECHOES of the local neighbourhood: 2M++ at Manticore "
                       "peculiar-velocity distances + a per-realization completion drawn from "
                       "each Manticore posterior density field. mode='full' also restores the "
                       "faint galaxies below the flux limit everywhere (volume-limited to m_faint).",
        "mode": args.mode, "intensity": args.intensity, "frame": "equatorial comoving [Mpc]",
        "H0": 68.1, "dmax_mpc": args.dmax,
        "zoa_deg": args.zoa_deg, "m_faint": args.m_faint, "k_lim": args.k_lim,
        "base": "local_2mpp_observed.npz", "n_observed": int(cat.N_data), "realizations": meta,
        "note": "Completed catalog(realization i) = observed + local_2mpp_<mode>_mcmc<i>.npz. "
                "PROV 0=observed, 5=inpaint; full mode adds absmag/uncert/kind(zoa|faint). The "
                "ensemble over realizations carries the Manticore reconstruction's posterior "
                "uncertainty; per-galaxy uncert is the ensemble field scatter.",
    }
    with open(os.path.join(OUT, "local_2mpp_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote {OUT}/ ({len(reals)} realizations); "
          f"completed catalog(i) = observed + zoa_mcmc<i>.")


if __name__ == "__main__":
    main()
