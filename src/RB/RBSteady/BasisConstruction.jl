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
    #length(mat) > 1e6 && (size(mat,1) > 1e2*size(mat,2) || size(mat,2) > 1e2*size(mat,1))
    #)

"""
    truncated_pod(mat::AbstractMatrix,args...;kwargs...) -> AbstractMatrix

Truncated proper orthogonal decomposition. When a symmetric, positive definite
matrix `X` is provided as an argument, the output's columns are `X`-orthogonal,
otherwise they are ℓ²-orthogonal

"""
function truncated_pod(mat::AbstractMatrix,args...;kwargs...)
  U,Σ,V = _truncated_pod(mat,args...;kwargs...)
  return U
end

function _truncated_pod(mat::AbstractMatrix,args...;kwargs...)
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

function standard_tpod(mat::AbstractMatrix,X::AbstractSparseMatrix;kwargs...)
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

function _tt_truncated_pod(a::AbstractArray{T,3},args...;kwargs...) where T
  mat = reshape(a,size(a,1)*size(a,2),:)
  Ur,Σr,Vr = _truncated_pod(mat;kwargs...)
  core = reshape(Ur,size(a,1),size(a,2),:)
  remainder = Σr.*Vr'
  return core,remainder
end

function _tt_truncated_pod(a::AbstractArray{T,3},X::AbstractSparseMatrix;kwargs...) where T
  perm = (2,1,3)
  prev_rank = size(a,1)
  cur_size = size(a,2)

  C = cholesky(X)
  L,p = sparse(C.L),C.p

  Xa = _tt_mul(L,p,a)
  Xmat = reshape(Xa,:,size(Xa,3))
  Ũr,Σr,Vr = truncated_svd(Xmat;kwargs...)
  c̃ = reshape(Ũr,size(Xa,1),size(Xa,2),:)
  core = _tt_div(L,p,c̃)
  remainder = Σr.*Vr'

  return core,remainder
end

function _tt_mul(L::AbstractSparseMatrix{T},p::Vector{Int},a::AbstractArray{T,3}) where T
  @check size(L,2) == size(a,2)
  b = similar(a,T,(size(a,1),size(L,1),size(a,3)))
  ap = a[:,p,:]
  @inbounds for i1 in axes(b,1)
    ap1 = ap[i1,:,:]
    @inbounds for i3 in axes(b,3)
      b[i1,:,i3] = L'*ap1[:,i3]
    end
  end
  return b
end

function _tt_div(L::AbstractSparseMatrix{T},p::Vector{Int},a::AbstractArray{T,3}) where T
  @check size(L,1) == size(L,2) == size(a,2)
  b = similar(a)
  @inbounds for i1 in axes(b,1)
    a1 = a[i1,:,:]
    @inbounds for i3 in axes(b,3)
      b[i1,p,i3] = L'\a1[:,i3]
    end
  end
  return b
end

# We are not interested in the last dimension (corresponds to the parameter).
# Note: when X is provided as norm matrix, ids_range has length 1, so we actually
# are not computing a cholesky decomposition numerous times

function ttsvd!(cache,a::AbstractArray{T,N},args...;ids_range=1:N-1,kwargs...) where {T,N}
  cores,ranks,sizes = cache
  d1 = ids_range[1]
  a_d = reshape(a,ranks[d1],sizes[d1],:)
  for d in ids_range
    core_d,remainder_d = _tt_truncated_pod(a_d,args...;kwargs...)
    rank = size(core_d,3)
    ranks[d+1] = rank
    cores[d] = core_d
    a_d = reshape(remainder_d,rank,sizes[d+1],:)
  end
  return a_d
end

function ttsvd!(cache,a::AbstractArray{T,N},X::AbstractRank1Tensor;ids_range=1:N-1,kwargs...) where {T,N}
  for d in ids_range
    a = ttsvd!(cache,a,X[d];ids_range=d,kwargs...)
  end
  return a
end

function ttsvd!(cache,a::AbstractArray{T,N},X::AbstractRankTensor;ids_range=1:N,kwargs...) where {T,N}
  cores,ranks,sizes = cache
  cores_k,as = map(1:rank(X)) do k
    cores_k = copy(cores[ids_range])
    ranks_k = copy(ranks)
    a_k = ttsvd!((cores_k,ranks_k,sizes),a,X[k];ids_range,kwargs...)
    cores_k,a_k
  end |> tuple_of_arrays
  for d in ids_range
    touched = d == first(ids_range) ? fill(true,rank(X)) : I(rank(X))
    cores_d = getindex.(cores_k,d)
    cores[d] = BlockCore(cores_d,touched)
  end
  R = orthogonalize!(cores,ranks,X;ids_range)
  a = absorb(cat(as...;dims=1),R)
  return a
end

function pivoted_qr(A;tol=1e-10)
  C = qr(A,ColumnNorm())
  r = findlast(abs.(diag(C.R)) .> tol)
  Q = C.Q[:,1:r]
  R = C.R[1:r,invperm(C.jpvt)]
  return Q,R
end

function orthogonalize!(cores,ranks,X::AbstractTProductTensor;ids_range=eachindex(cores))
  weight = ones(1,rank(X),1)
  decomp = get_decomposition(X)
  for d in ids_range
    core = cores[d]
    if d == last(ids_range)
      XW = _get_norm_matrix_from_weight(X,weight)
      core′,R = reduce_rank(core,XW)
      cores[d] = core′
      ranks[d+1] = size(core′,3)
      return R
    end
    next_core = cores[d+1]
    Xd = getindex.(decomp,d)
    core′,R = reduce_rank(core)
    cores[d] = core′
    ranks[d+1] = size(core′,3)
    cores[d+1] = absorb(next_core,R)
    weight = _weight_array(weight,core′,Xd)
  end
end

function reduce_rank(core::AbstractArray{T,3},X::AbstractMatrix) where T
  C = cholesky(X)
  L,p = sparse(C.L),C.p
  mat = reshape(core,:,size(core,3))
  Xmat = L'*mat[p,:]
  Q̃,R = pivoted_qr(Xmat)
  Q = (L'\Q̃)[invperm(p),:]
  core′ = reshape(Q,size(core,1),size(core,2),:)
  return core′,R
end

function reduce_rank(core::AbstractArray{T,3}) where T
  mat = reshape(core,:,size(core,3))
  Q,R = pivoted_qr(mat)
  core′ = reshape(Q,size(core,1),size(core,2),:)
  return core′,R
end

function absorb(core::AbstractArray{T,3},R::AbstractMatrix) where T
  Rcore = R*reshape(core,size(core,1),:)
  return reshape(Rcore,size(Rcore,1),size(core,2),:)
end

function _weight_array(weight,core,X)
  @check length(X) == size(weight,2)
  K = size(weight,2)
  rank = size(core,3)
  rank_prev = size(weight,3)
  W = zeros(rank,K,rank)
  w = zeros(size(core,2))
  @inbounds for k = 1:K
    Xk = X[k]
    @views Wk = W[:,k,:]
    @views Wk_prev = weight[:,k,:]
    for i′_prev = 1:rank_prev
      for i′ = 1:rank
        mul!(w,Xk,core[i′_prev,:,i′])
        for i_prev = 1:rank_prev
          Wk_prev′ = Wk_prev[i_prev,i′_prev]
          for i = 1:rank
            Wk[i,i′] += Wk_prev′*core[i_prev,:,i]'*w
          end
        end
      end
    end
  end
  return W
end

function _get_norm_matrix_from_weight(X::AbstractRankTensor,WD)
  K = rank(X)
  XD = map(k -> X[k][end],1:K)
  XW = kron(XD[1],WD[:,1,:])
  @inbounds for k = 2:K
    XW += kron(XD[k],WD[:,k,:])
  end
  @. XW = (XW+XW')/2 # needed to eliminate roundoff errors
  return sparse(XW)
end

"""
    ttsvd(a::AbstractArray,args...;kwargs...) -> Vector{<:AbstractArray}

Tensor train singular value decomposition. When a symmetric, positive definite
matrix `X` is provided as an argument, the columns of A₁ ⊗ ... ⊗ Aₙ, where
A₁, ..., Aₙ are the components of the outer vector, are `X`-orthogonal,
otherwise they are ℓ²-orthogonal

"""
function ttsvd(a::AbstractArray{T,N};kwargs...) where {T,N}
  cores = Vector{Array{T,3}}(undef,N-1)
  ranks = fill(1,N)
  sizes = size(a)
  cache = cores,ranks,sizes
  # routine on the spatial indices
  ttsvd!(cache,a;ids_range=1:N-1,kwargs...)
  return cores
end

function ttsvd(a::AbstractArray{T,N},X::AbstractTProductTensor;kwargs...) where {T,N}
  N_space = N-1
  cores = Vector{Array{T,3}}(undef,N-1)
  ranks = fill(1,N)
  sizes = size(a)
  # routine on the spatial indices
  ttsvd!((cores,ranks,sizes),a,X;ids_range=1:N_space,kwargs...)
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
    mat_i /= induced_norm(mat_i,args...)
    mat[:,i] .= mat_i
  end
end

# for testing purposes

function check_orthogonality(cores::AbstractVector{<:AbstractArray{T,3}},X::AbstractTProductTensor) where T
  Xglobal = kron(X)
  basis = dropdims(_cores2basis(cores...);dims=1)
  isorth = norm(basis'*Xglobal*basis - I) ≤ 1e-10
  return isorth
end
