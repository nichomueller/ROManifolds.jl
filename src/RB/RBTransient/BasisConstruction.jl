# tt-svd

function RBSteady.ttsvd(mat::AbstractTransientSnapshots{T,N},X=nothing;kwargs...) where {T,N}
  cores = Vector{Array{T,3}}(undef,N-1)
  ranks = fill(1,N)
  sizes = size(mat)
  cache = cores,ranks,sizes
  # routine on the spatial indexes
  M = ttsvd!(cache,mat,X;ids_range=1:N-2,kwargs...)
  # routine on the temporal index
  _ = ttsvd!(cache,M;ids_range=N-1,kwargs...)
  return cores
end

function RBSteady.ttsvd(mat::AbstractTransientSnapshots{T,N},X::AbstractTProductArray;kwargs...) where {T,N}
  N_space = N-2
  cores = Vector{Array{T,3}}(undef,N-1)
  weights = Vector{Array{T,3}}(undef,N_space-1)
  ranks = fill(1,N)
  sizes = size(mat)
  # routine on the spatial indexes
  M = ttsvd_and_weights!((cores,weights,ranks,sizes),mat,X;kwargs...)
  # routine on the temporal index
  _ = ttsvd!((cores,ranks,sizes),M;ids_range=N_space+1,kwargs...)
  return cores
end
