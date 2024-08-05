function rescale_snapshots!(s::AbstractMatrix,X::AbstractTProductArray)
  X′ = rank1_approximation(X)
  Xk = kron(X)
  Ck = cholesky(Xk)
  C′ = cholesky(X′)
  ldiv!(Ck,s)
  lmul!(C,s)
  return s
end

function rescale_snapshots(s::AbstractSnapshots,X::AbstractTProductArray)
  sf = flatten_snapshots(s)
  sf′ = copy(sf)
  rescale_snapshots!(sf′,X)
  return sf
end

function rank1_factors(X::AbstractTProductArray)
  return X
end

function rank1_factors(X::TProductGradientArray;norm=:gradient)
  if norm == :gradient
    X.arrays_1d .+ X.gradients_1d
  else
    X.arrays_1d
  end
end

function rank1_approximation(X::AbstractTProductArray)
  factors = rank1_factors(X)
  TProduct._kron(factors...)
end

function new_ttsvd!(cache,mat::AbstractArray{T,N},X::AbstractVector{<:AbstractSparseMatrix};ids_range=1:N-1,kwargs...) where {T,N}
  cores,ranks,sizes = cache
  for k in ids_range
    mat_k = reshape(mat,ranks[k]*sizes[k],:)
    Ur,Σr,Vr = _tpod(mat_k,X[k];kwargs...)
    rank = size(Ur,2)
    ranks[k+1] = rank
    mat = reshape(Σr.*Vr',rank,sizes[k+1],:)
    cores[k] = reshape(Ur,ranks[k],sizes[k],rank)
  end
  return mat
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
