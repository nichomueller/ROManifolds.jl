function truncation(s::AbstractVector;ϵ=1e-4,rank=length(s))
  energies = cumsum(s.^2;dims=1)
  tolrank = findfirst(energies .>= (1-ϵ^2)*energies[end])
  min(rank,tolrank)
end

function _cholesky_factor_and_perm(mat::AbstractMatrix)
  C = cholesky(mat,RowMaximum();tol = 0.0,check=true)
  return C.L,C.p
end

function _cholesky_factor_and_perm(mat::AbstractSparseMatrix)
  C = cholesky(mat)
  return sparse(C.L),C.p
end

function tpod(mat::AbstractMatrix,args...;kwargs...)
  U,Σ,V = svd(mat)
  rank = truncation(Σ;kwargs...)
  U[:,1:rank]
end

function tpod(mat::AbstractMatrix,X::AbstractSparseMatrix;kwargs...)
  L,p = _cholesky_factor_and_perm(X)
  Xmat = L'*mat[p,:]
  U,Σ,V = svd(Xmat)
  rank = truncation(Σ;kwargs...)
  (L'\U[:,1:rank])[invperm(p),:]
end

# we are not interested in the last dimension (corresponds to the parameter)

function ttsvd!(cache,mat::AbstractArray{T,N},args...;ids_range=1:N-1,kwargs...) where {T,N}
  cores,ranks,sizes = cache
  for k in ids_range
    mat_k = reshape(mat,ranks[k]*sizes[k],:)
    U,Σ,V = svd(mat_k)
    rank = truncation(Σ;kwargs...)
    core_k = U[:,1:rank]
    ranks[k+1] = rank
    mat = reshape(Σ[1:rank].*V[:,1:rank]',rank,sizes[k+1],:)
    cores[k] = reshape(core_k,ranks[k],sizes[k],rank)
  end
  return mat
end

function ttsvd!(cache,mat::AbstractArray{T,N},X::AbstractMatrix;ids_range=1:N-1,kwargs...) where {T,N}
  L,p = _cholesky_factor_and_perm(X)
  Ip = invperm(p)
  cores,ranks,sizes = cache
  for k in ids_range
    Xmat_k = L'*reshape(mat,ranks[k]*sizes[k],:)[p,:]
    U,Σ,V = svd(Xmat_k)
    rank = truncation(Σ;kwargs...)
    core_k = (L'\U[:,1:rank])[Ip,:]
    ranks[k+1] = rank
    mat = reshape(Σ[1:rank].*V[:,1:rank]',rank,sizes[k+1],:)
    cores[k] = reshape(core_k,ranks[k],sizes[k],rank)
  end
  return mat
end

function ttsvd_and_weights!(cache,mat::AbstractArray,X::AbstractTProductArray;kwargs...)
  cores,weights,ranks,sizes = cache
  for k in 1:1:length(X.arrays_1d)-1
    mat_k = reshape(mat,ranks[k]*sizes[k],:)
    U,Σ,V = svd(mat_k)
    rank = truncation(Σ;kwargs...)
    core_k = U[:,1:rank]
    ranks[k+1] = rank
    mat = reshape(Σ[1:rank].*V[:,1:rank]',rank,sizes[k+1],:)
    cores[k] = reshape(core_k,ranks[k],sizes[k],rank)
    _weight_array!(weights,cores,X,Val(k))
  end
  XW = _get_norm_matrix_from_weights(X,weights)
  M = ttsvd!((cores,ranks,sizes),mat,XW;ids_range=1:length(X.arrays_1d),kwargs...)
  return M
end

function _get_norm_matrices(X::TProductArray,::Val{d}) where d
  return [X.arrays_1d[d]]
end

function _get_norm_matrices(X::TProductGradientArray,::Val{1},::Val{1})
  return [X.arrays_1d[1],X.gradients_1d[1]]
end
function _get_norm_matrices(X::TProductGradientArray,::Val{1},::Val{2})
  return [X.arrays_1d[1],X.gradients_1d[1],X.arrays_1d[1]]
end
function _get_norm_matrices(X::TProductGradientArray,::Val{2},::Val{2})
  return [X.arrays_1d[2],X.arrays_1d[2],X.gradients_1d[2]]
end
function _get_norm_matrices(X::TProductGradientArray,::Val{1},::Val{3})
  return [X.arrays_1d[1],X.gradients_1d[1],X.arrays_1d[1],X.arrays_1d[1]]
end
function _get_norm_matrices(X::TProductGradientArray,::Val{2},::Val{3})
  return [X.arrays_1d[2],X.arrays_1d[2],X.gradients_1d[2],X.arrays_1d[2]]
end
function _get_norm_matrices(X::TProductGradientArray,::Val{3},::Val{3})
  return [X.arrays_1d[3],X.arrays_1d[3],X.arrays_1d[3],X.gradients_1d[3]]
end

function _get_norm_matrices(X::TProductGradientArray,::Val{d}) where d
  _get_norm_matrices(X,Val(d),Val{length(X.arrays_1d)}())
end

function _weight_array!(weights,cores,X,::Val{1})
  X1 = _get_norm_matrices(X,Val(1))
  K = length(X1)
  core = cores[1]
  rank = size(core,3)
  W = zeros(rank,K,rank)
  @inbounds for k = 1:K
    X1k = X1[k]
    for i = 1:rank, i′ = 1:rank
      W[i,k,i′] = core[1,:,i]'*X1k*core[1,:,i′]
    end
  end
  weights[1] = W
  return
end

function _weight_array!(weights,cores,X,::Val{d}) where d
  Xd = _get_norm_matrices(X,Val(d))
  K = length(Xd)
  W_prev = weights[d-1]
  core = cores[d]
  rank = size(core,3)
  rank_prev = size(W_prev,3)
  W = zeros(rank,K,rank)
  @inbounds for k = 1:K
    Xdk = Xd[k]
    Wk = W[:,k,:]
    Wk_prev = W_prev[:,k,:]
    for i = 1:rank, i′ = 1:rank
      Wk_ii′ = Wk[i,i′]
      for i_prev = 1:rank_prev, i′_prev = 1:rank_prev
        Wk_ii′ += Wk_prev[i_prev,i′_prev]*core[i_prev,:,i]'*Xdk*core[i′_prev,:,i′]
      end
    end
  end
  weights[d] = W
  return
end

function _get_norm_matrix_from_weights(norms,weights)
  N_space = length(weights)+1
  X = _get_norm_matrices(norms,Val(N_space))
  W = weights[end]
  @check length(X) == size(W,2)
  iX = first(X)
  Tx = eltype(iX)
  Tw = eltype(W)
  T = promote_type(Tx,Tw)
  XW = zeros(T,size(iX,1)*size(W,1),size(iX,2)*size(W,3))
  @inbounds for k = eachindex(X)
    XW += kron(X[k],W[:,k,:])
  end
  @fastmath (XW+XW')/2 # needed to eliminate roundoff errors
end

function ttsvd(mat::AbstractArray{T,N},X=nothing;kwargs...) where {T,N}
  cores = Vector{Array{T,3}}(undef,N-1)
  ranks = fill(1,N)
  sizes = size(mat)
  cache = cores,ranks,sizes
  # routine on the spatial indexes
  ttsvd!(cache,mat,X;ids_range=1:N-1,kwargs...)
  return cores
end

function ttsvd(mat::AbstractArray{T,N},X::AbstractTProductArray;kwargs...) where {T,N}
  cores = Vector{Array{T,3}}(undef,N-1)
  weights = Vector{Array{T,3}}(undef,N-1)
  ranks = fill(1,N)
  sizes = size(mat)
  # routine on the spatial indexes
  ttsvd_and_weights!((cores,weights,ranks,sizes),mat,X;kwargs...)
  return cores
end

function orth_projection(
  v::AbstractVector,
  basis::AbstractMatrix)

  proj = similar(v)
  proj .= zero(eltype(proj))
  @inbounds for b = eachcol(basis)
    proj += b*dot(v,b)/dot(b,b)
  end
  proj
end

function orth_projection(
  v::AbstractVector,
  basis::AbstractMatrix,
  X::AbstractMatrix)

  proj = similar(v)
  proj .= zero(eltype(proj))
  @inbounds for b = eachcol(basis)
    proj += b*dot(v,X,b)/dot(b,X,b)
  end
  proj
end

function orth_complement!(
  v::AbstractVector,
  basis::AbstractMatrix,
  args...)

  v .= v-orth_projection(v,basis,args...)
end

_norm(v::AbstractVector,args...) = norm(v)
_norm(v::AbstractVector,X::AbstractMatrix) = sqrt(v'*X*v)

function gram_schmidt!(
  mat::AbstractMatrix,
  basis::AbstractMatrix,
  args...)

  @inbounds for i = axes(mat,2)
    mat_i = mat[:,i]
    orth_complement!(mat_i,basis,args...)
    if i > 1
      orth_complement!(mat_i,mat[:,1:i-1],args...)
    end
    mat_i /= _norm(mat_i,args...)
    mat[:,i] .= mat_i
  end
end
