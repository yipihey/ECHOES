# Build a Vecchia graph with GraphGP.jl and write the prebuilt-graph NPZ that the ECHOES drivers
# (run_graphgp.jl / run_kernel_mle.jl / run_posterior.jl) consume. This replaces the Python
# `graphgp.build_graph` (JAX) call in `graphgp_julia.build_graph_npz`, removing the JAX dependency
# from graph construction. GraphGP.jl `build_graph` is byte-for-byte identical to `gp.build_graph`
# (same permutation, split dims, neighbours, offsets), so downstream numerics are unchanged.
#
#   julia --project=bench run_build_graph.jl <in.npz> <out.npz>
#
# in.npz  : build_points (N,D) Float64 (already 21-bit-lattice quantised by the caller), n0, k,
#           cov_bins32, cov_vals32, [aniso_spatial_bins, aniso_z_bins, aniso_grid, aniso_alpha]
# out.npz : coords (N,D) UInt32, neighbors (M,K) Int32 0-based, offsets, n0, scale, cov_bins32,
#           cov_vals32, indices (0-based) [+ aniso_* passthrough]   — the exact gp.build_graph layout.

using GraphGP, NPZ

d = npzread(ARGS[1])
pts = Float64.(d["build_points"])                       # (N, D)
n0 = Int(d["n0"][]); k = Int(d["k"][])
bins = Float64.(d["cov_bins32"]); vals = Float64.(d["cov_vals32"])

p = build_graph(pts, n0, k, bins, vals)                 # byte-identical to gp.build_graph

# GraphGPProblem stores coords (D,N) and neighbors (K,M) 1-based; emit the (N,D) / (M,K) 0-based
# layout the bridge NPZ uses (run_graphgp.jl transposes + adds 1 on read).
out = Dict{String,Any}(
    "coords"     => UInt32.(permutedims(p.coords)),          # (N, D)
    "neighbors"  => Int32.(permutedims(p.neighbors) .- 1),   # (M, K) 0-based
    "offsets"    => Int.(p.offsets),
    "n0"         => Int(p.n0),
    "scale"      => Float64(p.scale),
    "cov_bins32" => Float32.(bins),
    "cov_vals32" => Float32.(vals),
)
p.indices === nothing || (out["indices"] = Int.(p.indices .- 1))   # 0-based
for kk in ("aniso_spatial_bins", "aniso_z_bins", "aniso_grid", "aniso_alpha")
    haskey(d, kk) && (out[kk] = d[kk])                  # passthrough (covariance is assembled downstream)
end
npzwrite(ARGS[2], out)
