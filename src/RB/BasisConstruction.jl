function truncation(s::AbstractVector;ϵ=1e-4,rank=length(s))
  energies = cumsum(s.^2;dims=1)
  tolrank = findfirst(energies .>= (1-ϵ^2)*energies[end])
  min(rank,tolrank)
end

function tpod(mat::AbstractMatrix,args...;kwargs...)
  U,Σ,V = svd(mat)
  rank = truncation(Σ;kwargs...)
  U[:,1:rank]
end

function tpod(mat::AbstractMatrix,X::AbstractMatrix;kwargs...)
  C = cholesky(X)
  L = sparse(C.L)
  Xmat = L'*mat[C.p,:]
  U,Σ,V = svd(Xmat)
  rank = truncation(Σ;kwargs...)
  (L'\U[:,1:rank])[invperm(C.p),:]
end

# we are not interested in the last dimension (corresponds to the parameter)

function ttsvd!(cache,mat::AbstractArray{T,N},args...;ids_range=1:N-1,kwargs...) where {T,N}
  cores,ranks = cache
  sizes = size(mat)
  for k in ids_range
    mat = reshape(mat,ranks[k]*sizes[k],:)
    U,Σ,V = svd(mat)
    rank = truncation(Σ;kwargs...)
    core_k = U[:,1:rank]
    ranks[k+1] = rank
    mat = reshape(Σ[1:rank].*V[:,1:rank]',rank,sizes[k+1],:)
    cores[k] = reshape(core_k,ranks[k],sizes[k],rank)
  end
  return
end

function ttsvd!(cache,mat::AbstractArray{T,N},X::AbstractMatrix;ids_range=1:N-1,kwargs...) where {T,N}
  C = cholesky(X)
  L = sparse(C.L)
  Ip = invperm(C.p)
  Xmat = L'*mat[C.p,:]
  cores,ranks = cache
  sizes = size(mat)
  for k in ids_range
    mat = reshape(Xmat,ranks[k]*sizes[k],:)
    U,Σ,V = svd(Xmat)
    rank = truncation(Σ;kwargs...)
    core_k = (L'\U[:,1:rank])[Ip,:]
    ranks[k+1] = rank
    mat = reshape(Σ[1:rank].*V[:,1:rank]',rank,sizes[k+1],:)
    cores[k] = reshape(core_k,ranks[k],sizes[k],rank)
  end
  return
end

function ttsvd_and_weights!(
  cache,
  mat::AbstractArray,X::FEM.AbstractTProductArray;
  ids_range=1:FEM.get_dim(X)-1,kwargs...)

  cores,weights,ranks,sizes = cache
  for k in ids_range
    mat = reshape(mat,ranks[k]*sizes[k],:)
    U,Σ,V = svd(mat)
    rank = truncation(Σ;kwargs...)
    core_k = U[:,1:rank]
    ranks[k+1] = rank
    mat = reshape(Σ[1:rank].*V[:,1:rank]',rank,sizes[k+1],:)
    cores[k] = reshape(core_k,ranks[k],sizes[k],rank)
    _weight_array!(weights,cores,X,Val(k))
  end
  return
end

function ttsvd(mat::AbstractArray{T,N},X=nothing;kwargs...) where {T,N}
  cores = Vector{Array{T,3}}(undef,N-1)
  ranks = fill(1,N)
  cache = cores,ranks
  ttsvd!(cache,copy(mat),X;kwargs...)
  return cores
end

function _get_norm_matrices(X::TProductArray,::Val{d}) where d
  return X.arrays_1d[d]
end

function _get_norm_matrices(X::TProductGradientArray,::Val{d}) where d
  return X.arrays_1d[d],X.gradients_1d[d]
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

function _get_norm_matrix_from_weights(weights)
  wend = weights[end]
  w = sum(wend,dims=2)
  dropdims(w,dims=2)
end

function ttsvd(mat::AbstractArray{T,N},X::FEM.AbstractTProductArray;kwargs...) where {T,N}
  N_space = N-2
  cores = Vector{Array{T,3}}(undef,N-1)
  weights = Vector{Array{T,3}}(undef,N_space)
  ranks = fill(1,N)
  cache = cores,weights,ranks
  M = copy(mat)
  # routine on the indexes from 1 to N_space - 1
  ttsvd_and_weights!(cache,M,X;ids_range=1:FEM.get_dim(X)-1,kwargs...)
  # routine on the indexes N_space to N_space + 1
  XW = _get_norm_matrix_from_weights(weights)
  ttsvd!(cache,M,XW;ids_range=FEM.get_dim(X),kwargs...)
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
