# tt-svd

function RBSteady.ttsvd(mat::AbstractTransientSnapshots{T,N};kwargs...) where {T,N}
  cores = Vector{Array{T,3}}(undef,N-1)
  ranks = fill(1,N)
  sizes = size(mat)
  cache = cores,ranks,sizes
  # routine on the spatial and temporal indices
  _ = RBSteady.ttsvd!(cache,mat;ids_range=1:N-1,kwargs...)
  return cores
end

function RBSteady.ttsvd(mat::AbstractTransientSnapshots{T,N},X::AbstractTProductTensor;kwargs...) where {T,N}
  N_space = N-2
  cores = Vector{Array{T,3}}(undef,N-1)
  ranks = fill(1,N)
  sizes = size(mat)
  # routine on the spatial indices
  M = RBSteady.ttsvd!((cores,ranks,sizes),mat,X;ids_range=1:N_space,kwargs...)
  # routine on the temporal index
  _ = RBSteady.ttsvd!((cores,ranks,sizes),M;ids_range=N_space+1,kwargs...)
  return cores
end

function check_orthogonality(cores::AbstractVector{<:AbstractArray{T,3}},X::AbstractTProductTensor) where T
  Xglobal_space = kron(X)
  cores_space...,core_time = cores
  basis_space = cores2basis(cores_space...)
  isorth_space = norm(basis_space'*Xglobal_space*basis_space - I) ≤ 1e-10
  num_times = size(core_time,2)
  Xglobal_spacetime = kron(Float64.(I(num_times)),Xglobal_space)
  basis_spacetime = cores2basis(cores...)
  isorth_spacetime = norm(basis_spacetime'*Xglobal_spacetime*basis_spacetime - I) ≤ 1e-10
  return isorth_space,isorth_spacetime
end
