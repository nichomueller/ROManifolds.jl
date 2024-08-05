function RBSteady.new_ttsvd(mat::AbstractTransientSnapshots{T,N},X::AbstractTProductArray;kwargs...) where {T,N}
  N_space = N-2
  cores = Vector{Array{T,3}}(undef,N-1)
  ranks = fill(1,N)
  sizes = size(mat)
  # preprocessing
  X′ = RBSteady.rank1_factors(X)
  mat′ = RBSteady.rescale_snapshots(mat,X)
  # routine on the spatial indices
  M = RBSteady.new_ttsvd!((cores,ranks,sizes),mat′,X′;ids_range=1:N_space,kwargs...)
  # routine on the temporal index
  _ = RBSteady.ttsvd!((cores,ranks,sizes),M;ids_range=N_space+1,kwargs...)
  return cores
end

function new_ttsvd(mat::AbstractArray{T,N},X::AbstractTProductArray;kwargs...) where {T,N}
  X′ = rank1_factors(X)
  mat′ = rescale_snapshots(mat,X)
  N_space = N-1
  cores = Vector{Array{T,3}}(undef,N-1)
  weights = Vector{Array{T,3}}(undef,N_space-1)
  ranks = fill(1,N)
  sizes = size(mat)
  # routine on the spatial indices
  new_ttsvd!((cores,weights,ranks,sizes),mat′,X′;ids_range=1:N_space,kwargs...)
  return cores
end
