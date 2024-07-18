# tt-svd

function RBSteady.ttsvd(mat::AbstractTransientSnapshots{T,N};kwargs...) where {T,N}
  cores = Vector{Array{T,3}}(undef,N-1)
  ranks = fill(1,N)
  sizes = size(mat)
  cache = cores,ranks,sizes
  # routine on the spatial indices
  M = RBSteady.ttsvd!(cache,mat;ids_range=1:N-2,kwargs...)
  # routine on the temporal index
  _ = RBSteady.ttsvd!(cache,M;ids_range=N-1,kwargs...)
  return cores
end

function RBSteady.ttsvd(mat::AbstractTransientSnapshots{T,N},X::AbstractTProductArray;kwargs...) where {T,N}
  N_space = N-2
  cores = Vector{Array{T,3}}(undef,N-1)
  weights = Vector{Array{T,3}}(undef,N_space-1)
  ranks = fill(1,N)
  sizes = size(mat)
  # routine on the spatial indices
  M = RBSteady.ttsvd_and_weights!((cores,weights,ranks,sizes),mat,X;ids_range=1:N_space,kwargs...)
  # routine on the temporal index
  _ = RBSteady.ttsvd!((cores,ranks,sizes),M;ids_range=N_space+1,kwargs...)
  return cores
end

function check_orthogonality(cores::AbstractVector{<:AbstractArray{T,3}},X::AbstractTProductArray) where T
  Xglobal_space = kron(X)
  cores_space...,core_time = cores
  basis_space = dropdims(RBSteady._cores2basis(cores_space...);dims=1)
  isorth_space = norm(basis_space'*Xglobal_space*basis_space - I) ≤ 1e-10
  return isorth_space
end
