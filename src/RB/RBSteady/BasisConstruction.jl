function projection(red::AbstractReduction,s::AbstractSnapshots)
  @abstractmethod
end

function projection(red::PODReduction,s::AbstractSnapshots,args...)
  U,Σ,V = tpod(red,s)
  return U
end

function projection(red::TTSVDReduction,s::AbstractSnapshots,args...)
  cores,remainder = ttsvd(red,s)
  return cores
end

function _size_cond(M::AbstractMatrix)
  length(M) > 1e6 && (size(M,1) > 1e2*size(M,2) || size(M,2) > 1e2*size(M,1))
end

function _cholesky_decomp(X::AbstractSparseMatrix)
  C = cholesky(X)
  L = sparse(C.L)
  p = C.p
  return L,p
end

function select_rank(red::AbstractReduction,args...)
  @abstractmethod
end

function select_rank(red::SearchSVDRank,S::AbstractVector)
  tol = red.tol
  energies = cumsum(S.^2;dims=1)
  rank = findfirst(energies .>= (1-tol^2)*energies[end])
  return rank
end

function truncated_svd(red::AbstractReduction{SearchSVDRank},M::AbstractMatrix;issquare=false)
  U,S,V = svd(M)
  if issquare S = sqrt.(S) end
  rank = select_rank(red,S)
  return U[:,1:rank],S[1:rank],V[:,1:rank]
end

function truncated_svd(red::AbstractReduction{FixedSVDRank},M::AbstractMatrix;kwargs...)
  rank = red.rank
  return tsvd(M,rank)
end

function tpod(red::PODReduction,M::AbstractMatrix,args...)
  if ReductionStyle(red)==SearchSVDRank() && _size_cond(M)
    if size(M,1) > size(M,2)
      massive_rows_tpod(red,M,args...)
    else
      massive_cols_tpod(red,M,args...)
    end
  else
    standard_tpod(M,args...)
  end
end

function standard_tpod(red::AbstractReduction,M::AbstractMatrix)
  truncated_svd(red,M)
end

function standard_tpod(red::AbstractReduction,M::AbstractMatrix,X::AbstractSparseMatrix)
  L,p = _cholesky_decomp(X)
  XM = L'*M[p,:]
  Ũr,Σr,Vr = truncated_svd(XM)
  Ur = (L'\Ũr)[invperm(p),:]
  return Ur,Σr,Vr
end

function massive_rows_tpod(red::AbstractReduction,M::AbstractMatrix)
  MM = M'*M
  _,Σr,Vr = truncated_svd(MM;issquare=true)
  Ur = M*Vr
  @inbounds for i = axes(Ur,2)
    Ur[:,i] /= Σr[i]+eps()
  end
  return Ur,Σr,Vr
end

function massive_rows_tpod(red::AbstractReduction,M::AbstractMatrix,X::AbstractSparseMatrix)
  C = cholesky(X)
  L,p = sparse(C.L),C.p
  XM = L'*M[p,:]
  MXM = XM'*XM
  _,Σr,Vr = truncated_svd(MXM;issquare=true)
  Ũr = XM*Vr
  @inbounds for i = axes(Ũr,2)
    Ũr[:,i] /= Σr[i]+eps()
  end
  Ur = (L'\Ũr)[invperm(p),:]
  return Ur,Σr,Vr
end

function massive_cols_tpod(red::AbstractReduction,M::AbstractMatrix)
  MM = M*M'
  Ur,Σr,_ = truncated_svd(MM;issquare=true)
  Vr = Ur'*M
  @inbounds for i = axes(Ur,2)
    Vr[:,i] /= Σr[i]+eps()
  end
  return Ur,Σr,Vr'
end

function massive_cols_tpod(red::AbstractReduction,M::AbstractMatrix,X::AbstractSparseMatrix)
  C = cholesky(X)
  L,p = sparse(C.L),C.p
  XM = L'*M[p,:]
  MXM = XM*XM'
  Ũr,Σr,_ = truncated_svd(MXM;issquare=true)
  Ur = (L'\Ũr)[invperm(p),:]
  Vr = Ur'*XM
  @inbounds for i = axes(Ur,2)
    Vr[:,i] /= Σr[i]+eps()
  end
  return Ur,Σr,Vr'
end

function ttsvd_loop(red::AbstractReduction,A::AbstractArray{T,3}) where T
  M = reshape(A,size(A,1)*size(A,2),:)
  Ur,Σr,Vr = truncated_svd(red,M)
  core = reshape(Ur,size(A,1),size(A,2),:)
  remainder = Σr.*Vr'
  return core,remainder
end

function ttsvd_loop(red::AbstractReduction,A::AbstractArray{T,3},X::AbstractSparseMatrix) where T
  perm = (2,1,3)
  prev_rank = size(A,1)
  cur_size = size(A,2)

  L,p = _cholesky_decomp(X)

  XA = _tt_mul(L,p,A)
  XM = reshape(XA,:,size(XA,3))
  Ũr,Σr,Vr = truncated_svd(red,XM)
  c̃ = reshape(Ũr,size(XA,1),size(XA,2),:)
  core = _tt_div(L,p,c̃)
  remainder = Σr.*Vr'

  return core,remainder
end

function _tt_mul(L::AbstractSparseMatrix{T},p::Vector{Int},A::AbstractArray{T,3}) where T
  @check size(L,2) == size(A,2)
  B = similar(A,T,(size(A,1),size(L,1),size(A,3)))
  Ap = A[:,p,:]
  @inbounds for i1 in axes(B,1)
    Ap1 = Ap[i1,:,:]
    @inbounds for i3 in axes(B,3)
      B[i1,:,i3] = L'*Ap1[:,i3]
    end
  end
  return B
end

function _tt_div(L::AbstractSparseMatrix{T},p::Vector{Int},A::AbstractArray{T,3}) where T
  @check size(L,1) == size(L,2) == size(A,2)
  B = similar(A)
  @inbounds for i1 in axes(B,1)
    A1 = A[i1,:,:]
    @inbounds for i3 in axes(B,3)
      B[i1,p,i3] = L'\A1[:,i3]
    end
  end
  return B
end

# We are not interested in the last dimension (corresponds to the parameter)

function ttsvd(red::TTSVDReduction,A::AbstractArray{T,N}) where {T,N}
  cores = Array{T,3}[]
  oldrank = 1
  A_d = reshape(A,oldrank,size(A,1),:)
  for d in 1:N-1
    core_d,remainder_d = ttsvd_loop(red,A_d)
    oldrank = size(core_d,3)
    A_d = reshape(remainder_d,oldrank,size(A,d+1),:)
    push!(cores,core_d)
  end
  return cores,A_d
end

function ttsvd(red::TTSVDReduction,A::AbstractArray{T,N},X::AbstractRank1Tensor) where {T,N}
  Nspace = length(get_factors(X))
  @check Nspace ≤ N-1

  cores = Array{T,3}[]
  oldrank = 1
  A_d = reshape(A,oldrank,size(A,1),:)
  for d in 1:Nspace-1
    core_d,remainder_d = ttsvd_loop(red,A_d)
    oldrank = size(core_d,3)
    A_d = reshape(remainder_d,oldrank,size(A,d+1),:)
    push!(cores,core_d)
  end

  return cores,A_d
end

function ttsvd(red::TTSVDReduction,A::AbstractArray{T,N},X::AbstractRankTensor) where {T,N}
  cores_k,remainders_k = map(k -> ttsvd(red,A,X[k]),rank(X)) |> tuple_of_arrays
  cores = Array{T,3}[]
  for d in eachindex(first(cores_k))
    touched = d == 1 ? fill(true,rank(X)) : I(rank(X))
    cores_d = getindex.(cores_k,d)
    push!(cores,BlockCore(cores_d,touched))
  end
  R = orthogonalize!(cores,X)
  A_d = absorb(cat(remainder...;dims=1),R)
  return cores,A_d
end

function pivoted_qr(A;tol=1e-10)
  C = qr(A,ColumnNorm())
  r = findlast(abs.(diag(C.R)) .> tol)
  Q = C.Q[:,1:r]
  R = C.R[1:r,invperm(C.jpvt)]
  return Q,R
end

function orthogonalize!(cores,X::AbstractTProductTensor)
  weight = ones(1,rank(X),1)
  decomp = get_decomposition(X)
  for d in eachindex(cores)
    core_d = cores[d]
    if d == last(ids_range)
      XW = _get_norm_matrix_from_weight(X,weight)
      core_d′,R = reduce_rank(core_d,XW)
      cores[d] = core_d′
      return R
    end
    next_core = cores[d+1]
    X_d = getindex.(decomp,d)
    core_d′,R = reduce_rank(core_d,XW)
    cores[d] = core_d′
    cores[d+1] = absorb(next_core,R)
    weight = _weight_array(weight,core′,X_d)
  end
end

function reduce_rank(core::AbstractArray{T,3},X::AbstractMatrix) where T
  L,p = _cholesky_decomp(X)
  M = reshape(core,:,size(core,3))
  XM = L'*M[p,:]
  Q̃,R = pivoted_qr(XM)
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
