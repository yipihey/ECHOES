"""RETIRED — this script was a JAX-vs-Julia equivalence diff for the local-volume GraphGP field.

Both of its products were validation checks against the now-removed JAX reference oracle:
  A. build_local_gp_field cube + the Julia generate_field vs the JAX generate_field engine
  B. field_posterior_vecchia inpaint + gp_posterior_dense (the JAX dense reference)
and the SUMMARY reported PASS/FAIL of that jax-vs-julia comparison. The echoes package is now
JAX-free (the GP engine runs on GraphGP.jl), so the JAX arm no longer exists and there is no
standalone Julia purpose left here. The production local-volume field build now lives in
pipeline/build_local_release.py.
"""
raise SystemExit(
    "retired: JAX-vs-Julia comparison; the field build now lives in build_local_release.py"
)
