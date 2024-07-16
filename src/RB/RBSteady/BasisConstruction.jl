"""
    truncation(s::AbstractVector;ϵ=1e-4,rank::Integer=length(s)) -> Integer

Returns the minimum between `rank` and the first integer `tolrank` such that
(`s`[1]² + ... + `s`[`tolrank`]²) / (`s`[1]² + ... + `s`[end]²) ≥ `ϵ`²,
where `s` are the singular values of the SVD of a given matrix. The equation
above is the so-called relative energy criterion

"""
function truncation(s::AbstractVector;ϵ=1e-4,rank::Integer=length(s))
  energies = cumsum(s.^2;dims=1)
  tolrank = findfirst(energies .>= (1-ϵ^2)*energies[end])
  min(rank,tolrank)
end

function select_modes(U::AbstractMatrix,Σ::AbstractVector,V::AbstractMatrix;kwargs...)
  rank = truncation(Σ;kwargs...)
  return U[:,1:rank],Σ[1:rank],V[:,1:rank]
end

_size_condition(mat::AbstractMatrix) = (
  length(mat) > 1e6 && size(mat,1) ≥ size(mat,2) &&
  (size(mat,1) > 1e2*size(mat,2) || size(mat,2) > 1e2*size(mat,1))
  )


"""
    tpod(mat::AbstractMatrix,args...;kwargs...) -> AbstractMatrix

Truncated proper orthogonal decomposition. When a symmetric, positive definite
matrix `X` is provided as an argument, the output's columns are `X`-orthogonal,
otherwise they are ℓ²-orthogonal

"""
function tpod(mat::AbstractMatrix,args...;kwargs...)
  U,Σ,V = _tpod(mat,args...;kwargs...)
  return U
end

function _tpod(mat::AbstractMatrix,args...;kwargs...)
  if _size_condition(mat)
    massive_tpod(mat,args...;kwargs...)
  else
    standard_tpod(mat,args...;kwargs...)
  end
end

function standard_tpod(mat::AbstractMatrix,args...;kwargs...)
  U,Σ,V = svd(mat)
  Ur,Σr,Vr = select_modes(U,Σ,V;kwargs...)
  return Ur,Σr,Vr
end

function standard_tpod(mat::AbstractMatrix,X::AbstractMatrix;kwargs...)
  C = cholesky(X)
  L,p = sparse(C.L),C.p
  Xmat = L'*mat[p,:]
  Ũ,Σ,V = svd(Xmat)
  Ũr,Σr,Vr = select_modes(Ũ,Σ,V;kwargs...)
  Ur = (L'\Ũr)[invperm(p),:]
  return Ur,Σr,Vr
end

function massive_tpod(mat::AbstractMatrix,X::AbstractSparseMatrix;kwargs...)
  C = cholesky(X)
  L,p = sparse(C.L),C.p
  Xmat = L'*mat[p,:]
  mXmat = Xmat'*Xmat
  Vt,Σ²,V = svd(mXmat)
  Σ = sqrt.(Σ²)
  _,Σr,Vr = select_modes(Vt,Σ,V;kwargs...)
  Ũr = Xmat*Vr
  @inbounds for i = axes(Ũr,2)
    Ũr[:,i] /= Σr[i]+eps()
  end
  Ur = (L'\Ũr)[invperm(p),:]
  return Ur,Σr,Vr
end

function massive_tpod(mat::AbstractMatrix,args...;kwargs...)
  mmat = mat'*mat
  Vt,Σ²,V = svd(mat)
  Σ = sqrt.(Σ²)
  _,Σr,Vr = select_modes(Vt,Σ,V;kwargs...)
  Ur = mat*Vr
  @inbounds for i = axes(Ur,2)
    Ur[:,i] /= Σr[i]+eps()
  end
  return Ur,Σr,Vr
end

# We are not interested in the last dimension (corresponds to the parameter).
# Note: when X is provided as norm matrix, ids_range has length 1, so we actually
# are not computing a cholesky decomposition numerous times
function ttsvd!(cache,mat::AbstractArray{T,N},args...;ids_range=1:N-1,kwargs...) where {T,N}
  cores,ranks,sizes = cache
  for k in ids_range
    mat_k = reshape(mat,ranks[k]*sizes[k],:)
    Ur,Σr,Vr = _tpod(mat_k,args...;kwargs...)
    rank = size(Ur,2)
    ranks[k+1] = rank
    mat = reshape(Σr.*Vr',rank,sizes[k+1],:)
    cores[k] = reshape(Ur,ranks[k],sizes[k],rank)
  end
  return mat
end

function ttsvd_and_weights!(cache,mat::AbstractArray,X::AbstractTProductArray;ids_range=eachindex(X),kwargs...)
  cores,weights,ranks,sizes = cache
  for k in first(ids_range):last(ids_range)-1
    mat_k = reshape(mat,ranks[k]*sizes[k],:)
    Ur,Σr,Vr = _tpod(mat_k;kwargs...)
    rank = size(Ur,2)
    ranks[k+1] = rank
    mat = reshape(Σr.*Vr',rank,sizes[k+1],:)
    cores[k] = reshape(Ur,ranks[k],sizes[k],rank)
    _weight_array!(weights,cores,X,Val{k}())
  end
  XW = _get_norm_matrix_from_weights(X,weights)
  M = ttsvd!((cores,ranks,sizes),mat,XW;ids_range=last(ids_range),kwargs...)
  return M
end

function _weight_array!(weights,cores,X,::Val{1})
  X1 = tp_getindex(X,1)
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

function _weight_array!(weights,cores::Vector{<:BlockTTCore},X,::Val{1})
  X1 = tp_getindex(X,1)
  K = length(X1)
  bcore = cores[1]
  offset = bcore.offset3
  rank = size(bcore,3)
  W = zeros(rank,K,rank)
  for (i,core) in enumerate(blocks(bcore))
    range = offset[i]+1:offset[i+1]
    @inbounds for k = 1:K
      X1k = X1[k]
      for i = range, i′ = range
        W[i,k,i′] = core[1,:,i]'*X1k*core[1,:,i′]
      end
    end
  end
  weights[1] = W
  return
end

function _weight_array!(weights,cores,X,::Val{d}) where d
  Xd = tp_getindex(X,d)
  K = length(Xd)
  W_prev = weights[d-1]
  core = cores[d]
  rank = size(core,3)
  rank_prev = size(W_prev,3)
  W = zeros(rank,K,rank)
  @inbounds for k = 1:K
    Xdk = Xd[k]
    @views Wk = W[:,k,:]
    @views Wk_prev = W_prev[:,k,:]
    for i_prev = 1:rank_prev, i′_prev = 1:rank_prev
      Wk_prev′ = Wk_prev[i_prev,i′_prev]
      for i = 1:rank, i′ = 1:rank
        Wk[i,i′] += Wk_prev′*core[i_prev,:,i]'*Xdk*core[i′_prev,:,i′]
      end
    end
  end
  weights[d] = W
  return
end

function _weight_array!(weights,cores::Vector{<:BlockTTCore},X,::Val{d}) where d
  Xd = tp_getindex(X,d)
  K = length(Xd)
  W_prev = weights[d-1]
  bcore = cores[d]
  offset = bcore.offset3
  rank = size(bcore,3)
  bcore_prev = cores[d-1]
  offset_prev = bcore_prev.offset3
  rank_prev = size(bcore_prev,3)
  W = zeros(rank,K,rank)
  for (i,core) in enumerate(blocks(bcore))
    range = offset[i]+1:offset[i+1]
    range_prev = offset_prev[i]+1:offset_prev[i+1]
    @inbounds for k = 1:K
      Xdk = Xd[k]
      @views Wk = W[:,k,:]
      Wk_prev = W_prev[:,k,:]
      for i_prev = range_prev, i′_prev = range_prev
        Wk_prev′ = Wk_prev[i_prev,i′_prev]
        for i = range, i′ = range
          Wk[i,i′] += Wk_prev′*core[i_prev,:,i]'*Xdk*core[i′_prev,:,i′]
        end
      end
    end
  end
  weights[d] = W
  return
end

function _get_norm_matrix_from_weights(norms::AbstractTProductArray,weights)
  N_space = tp_length(norms)
  X = tp_getindex(norms,N_space)
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
  @. XW = (XW+XW')/2 # needed to eliminate roundoff errors
  return sparse(XW)
end

"""
    ttsvd(mat::AbstractMatrix,args...;kwargs...) -> Vector{<:AbstractArray}

Tensor train singular value decomposition. When a symmetric, positive definite
matrix `X` is provided as an argument, the columns of A₁ ⊗ ... ⊗ Aₙ, where
A₁, ..., Aₙ are the components of the outer vector, are `X`-orthogonal,
otherwise they are ℓ²-orthogonal

"""
function ttsvd(mat::AbstractArray{T,N};kwargs...) where {T,N}
  cores = Vector{Array{T,3}}(undef,N-1)
  ranks = fill(1,N)
  sizes = size(mat)
  cache = cores,ranks,sizes
  # routine on the spatial indices
  ttsvd!(cache,mat;ids_range=1:N-1,kwargs...)
  return cores
end

function ttsvd(mat::AbstractArray{T,N},X::AbstractTProductArray;kwargs...) where {T,N}
  N_space = N-1
  cores = Vector{Array{T,3}}(undef,N-1)
  weights = Vector{Array{T,3}}(undef,N_space-1)
  ranks = fill(1,N)
  sizes = size(mat)
  # routine on the spatial indices
  ttsvd_and_weights!((cores,weights,ranks,sizes),mat,X;ids_range=1:N_space,kwargs...)
  return cores
end

"""
    orth_projection(v::AbstractVector, basis::AbstractMatrix, args...) -> AbstractVector

Orthogonal projection of `v` on the column space of `basis`. When a symmetric,
positive definite matrix `X` is provided as an argument, the output is `X`-orthogonal,
otherwise it is ℓ²-orthogonal

"""
function orth_projection(
  v::AbstractVector,
  basis::AbstractMatrix)

  proj = similar(v)
  fill!(proj,zero(eltype(proj)))
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
  fill!(proj,zero(eltype(proj)))
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

"""
    gram_schmidt!(mat::AbstractMatrix, basis::AbstractMatrix, args...) -> AbstractMatrix

Gram-Schmidt algorithm for an abstract matrix `mat` with respect to the column
space of `basis`. When a symmetric, positive definite matrix `X` is provided as
an argument, the output is `X`-orthogonal, otherwise it is ℓ²-orthogonal

"""
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

function orthogonalize!(core::AbstractArray{T,3},X::AbstractTProductArray,weights) where T
  XW = _get_norm_matrix_from_weights(X,weights)
  L,p = _cholesky_factor_and_perm(XW)
  mat = reshape(core,:,size(core,3))
  XWmat = L'*mat[p,:]
  Q̃,R = qr(XWmat)
  core .= reshape((L'\Q̃)[invperm(p),:],size(core))
  return R
end
