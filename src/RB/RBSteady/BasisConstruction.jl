"""
    truncation(s::AbstractVector;ϵ=1e-4,rank::Integer=length(s)) -> Integer

Returns the minimum between `rank` and the first integer `tolrank` such that
(`s`[1]² + ... + `s`[`tolrank`]²) / (`s`[1]² + ... + `s`[end]²) ≥ `ϵ`²,
where `s` are the singular values of the SVD of a given matrix. The equation
above is the so-called relative energy criterion

"""
function truncation(s::AbstractVector;ϵ=1e-4,rank::Integer=length(s),squares=false)
  energies = squares ? cumsum(s;dims=1) : cumsum(s.^2;dims=1)
  tolrank = findfirst(energies .>= (1-ϵ^2)*energies[end])
  min(rank,tolrank)
end

function select_modes(U::AbstractMatrix,Σ::AbstractVector,V::AbstractMatrix;squares=false,kwargs...)
  Σ′ = squares ? sqrt.(Σ) : Σ
  rank = truncation(Σ;squares,kwargs...)
  return U[:,1:rank],Σ′[1:rank],V[:,1:rank]
end

function truncated_svd(mat::AbstractMatrix;randomized=false,kwargs...)
  if randomized
    smin = minimum(size(mat))
    rank = cld(smin,2)
    U,Σ,V = rsvd(mat,rank)
  else
    U,Σ,V = svd(mat)
  end
  select_modes(U,Σ,V;kwargs...)
end

 _size_condition(mat::AbstractMatrix) = false #(
#   length(mat) > 1e6 && (size(mat,1) > 1e2*size(mat,2) || size(mat,2) > 1e2*size(mat,1))
#   )


"""
    truncated_pod(mat::AbstractMatrix,args...;kwargs...) -> AbstractMatrix

Truncated proper orthogonal decomposition. When a symmetric, positive definite
matrix `X` is provided as an argument, the output's columns are `X`-orthogonal,
otherwise they are ℓ²-orthogonal

"""
function truncated_pod(mat::AbstractMatrix,args...;kwargs...)
  U,Σ,V = _tpod(mat,args...;kwargs...)
  return U
end

function _tpod(mat::AbstractMatrix,args...;kwargs...)
  if _size_condition(mat)
    if size(mat,1) > size(mat,2)
      massive_rows_tpod(mat,args...;kwargs...)
    else
      massive_cols_tpod(mat,args...;kwargs...)
    end
  else
    standard_tpod(mat,args...;kwargs...)
  end
end

function standard_tpod(mat::AbstractMatrix,args...;kwargs...)
  truncated_svd(mat;kwargs...)
end

function standard_tpod(mat::AbstractMatrix,X::AbstractMatrix;kwargs...)
  C = cholesky(X)
  L,p = sparse(C.L),C.p
  Xmat = L'*mat[p,:]
  Ũr,Σr,Vr = truncated_svd(Xmat;kwargs...)
  Ur = (L'\Ũr)[invperm(p),:]
  return Ur,Σr,Vr
end

function massive_rows_tpod(mat::AbstractMatrix,args...;kwargs...)
  mmat = mat'*mat
  _,Σr,Vr = truncated_svd(mmat;squares=true,kwargs...)
  Ur = mat*Vr
  @inbounds for i = axes(Ur,2)
    Ur[:,i] /= Σr[i]+eps()
  end
  return Ur,Σr,Vr
end

function massive_rows_tpod(mat::AbstractMatrix,X::AbstractSparseMatrix;kwargs...)
  C = cholesky(X)
  L,p = sparse(C.L),C.p
  Xmat = L'*mat[p,:]
  mXmat = Xmat'*Xmat
  _,Σr,Vr = truncated_svd(mXmat;squares=true,kwargs...)
  Ũr = Xmat*Vr
  @inbounds for i = axes(Ũr,2)
    Ũr[:,i] /= Σr[i]+eps()
  end
  Ur = (L'\Ũr)[invperm(p),:]
  return Ur,Σr,Vr
end

function massive_cols_tpod(mat::AbstractMatrix,args...;kwargs...)
  mmat = mat*mat'
  Ur,Σr,_ = truncated_svd(mmat;squares=true,kwargs...)
  Vr = Ur'*mat
  @inbounds for i = axes(Ur,2)
    Vr[:,i] /= Σr[i]+eps()
  end
  return Ur,Σr,Vr'
end

function massive_cols_tpod(mat::AbstractMatrix,X::AbstractSparseMatrix;kwargs...)
  C = cholesky(X)
  L,p = sparse(C.L),C.p
  Xmat = L'*mat[p,:]
  mXmat = Xmat*Xmat'
  Ũr,Σr,_ = truncated_svd(mXmat;squares=true,kwargs...)
  Ur = (L'\Ũr)[invperm(p),:]
  Vr = Ur'*Xmat
  @inbounds for i = axes(Ur,2)
    Vr[:,i] /= Σr[i]+eps()
  end
  return Ur,Σr,Vr'
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
  w = zeros(size(core,2))
  @inbounds for k = 1:K
    X1k = X1[k]
    for i′ = 1:rank
      mul!(w,X1k,core[1,:,i′])
      for i = 1:rank
        W[i,k,i′] = core[1,:,i]'*w
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
  w = zeros(size(core,2))
  @inbounds for k = 1:K
    Xdk = Xd[k]
    @views Wk = W[:,k,:]
    @views Wk_prev = W_prev[:,k,:]
    for i′_prev = 1:rank_prev
      for i′ = 1:rank
        mul!(w,Xdk,core[i′_prev,:,i′])
        for i_prev = 1:rank_prev
          Wk_prev′ = Wk_prev[i_prev,i′_prev]
          for i = 1:rank
            Wk[i,i′] += Wk_prev′*core[i_prev,:,i]'*w
          end
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
  w = similar(proj)
  @inbounds for b = eachcol(basis)
    mul!(w,X,b)
    proj += b*dot(v,w)/dot(b,w)
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

function _norm(A::AbstractArray{T,3} where T,X::AbstractVector{<:AbstractMatrix})
  @check length(X) == 3
  X12,X3... = X
  n2 = vec(A)*vec([_norm(M,X12)^2 for M in eachslice(A,dims=3,drop=true)]*X3)
  return sqrt(n2)
end

function _norm(M::AbstractMatrix,X::AbstractVector{<:AbstractMatrix})
  @check length(X) == 2
  n2 = vec(M)'*vec(X[1]'*M*X[2])
  return sqrt(n2)
end

function _norm(a::AbstractArray,X::AbstractVector{<:AbstractVector{<:AbstractMatrix}})
  vXv = 0.0
  for Xd in X
    vXv += _norm(a,Xd)^2
  end
  return sqrt(vXv)
end

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

function pivoted_qr(A;tol=1e-10)
  C = qr(A,ColumnNorm())
  r = findlast(abs.(diag(C.R)) .> tol)
  Q = C.Q[:,1:r]
  R = C.R[1:r,invperm(C.jpvt)]
  return Q,R
end

function orthogonalize!(core::AbstractArray{T,3},X::AbstractTProductArray,weights) where T
  XW = _get_norm_matrix_from_weights(X,weights)
  C = cholesky(XW)
  L,p = sparse(C.L),C.p
  mat = reshape(core,:,size(core,3))
  XWmat = L'*mat[p,:]
  Q̃,R = pivoted_qr(XWmat)
  core .= reshape((L'\Q̃)[invperm(p),axes(XWmat,2)],size(core))
  return R
end

function absorb!(core::AbstractArray{T,3},R::AbstractMatrix) where T
  Rcore = R*reshape(core,size(core,1),:)
  Q̃,_ = qr(reshape(Rcore,:,size(core,3)))
  core .= reshape(Q̃[:,axes(core,3)],size(core))
end

# for testing purposes

function check_orthogonality(cores::AbstractVector{<:AbstractArray{T,3}},X::AbstractTProductArray) where T
  Xglobal = kron(X)
  basis = dropdims(_cores2basis(cores...);dims=1)
  isorth = norm(basis'*Xglobal*basis - I) ≤ 1e-10
  return isorth
end
