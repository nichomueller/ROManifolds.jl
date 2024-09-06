function projection(red::AbstractReduction,A::AbstractArray)
  @abstractmethod
end

function projection(red::PODReduction,A::AbstractArray,args...)
  red_style = ReductionStyle(red)
  U,S,V = tpod(red_style,A,args...)
  return U
end

function projection(red::TTSVDReduction,A::AbstractArray,args...)
  red_style = ReductionStyle(red)
  cores,remainder = ttsvd(red_style,A,args...)
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

function select_rank(red_style::ReductionStyle,args...)
  @abstractmethod
end

function select_rank(red_style::SearchSVDRank,S::AbstractVector)
  tol = red_style.tol
  energies = cumsum(S.^2;dims=1)
  rank = findfirst(energies .>= (1-tol^2)*energies[end])
  return rank
end

function truncated_svd(red_style::SearchSVDRank,M::AbstractMatrix;issquare=false)
  U,S,V = svd(M)
  if issquare S = sqrt.(S) end
  rank = select_rank(red_style,S)
  return U[:,1:rank],S[1:rank],V[:,1:rank]
end

function truncated_svd(red_style::FixedSVDRank,M::AbstractMatrix;issquare=false)
  rank = red_style.rank
  Ur,Sr,Vr = tsvd(M,rank)
  if issquare Sr = sqrt.(Sr) end
  return Ur,Sr,Vr
end

function tpod(red_style::ReductionStyle,M::AbstractMatrix,X::AbstractSparseMatrix)
  tpod(red_style,M,_cholesky_decomp(X)...)
end

function tpod(red_style::ReductionStyle,M::AbstractMatrix,args...)
  if isa(red_style,SearchSVDRank) && _size_cond(M)
    if size(M,1) > size(M,2)
      massive_rows_tpod(red_style,M,args...)
    else
      massive_cols_tpod(red_style,M,args...)
    end
  else
    standard_tpod(red_style,M,args...)
  end
end

function standard_tpod(red_style::ReductionStyle,M::AbstractMatrix)
  truncated_svd(red_style,M)
end

function standard_tpod(red_style::ReductionStyle,M::AbstractMatrix,L::AbstractSparseMatrix,p::AbstractVector{Int})
  XM = L'*M[p,:]
  Ũr,Sr,Vr = truncated_svd(red_style,XM)
  Ur = (L'\Ũr)[invperm(p),:]
  return Ur,Sr,Vr
end

function massive_rows_tpod(red_style::ReductionStyle,M::AbstractMatrix)
  MM = M'*M
  _,Sr,Vr = truncated_svd(red_style,MM;issquare=true)
  Ur = (M*Vr)*inv(Diagonal(Sr).+eps())
  return Ur,Sr,Vr
end

function massive_rows_tpod(red_style::ReductionStyle,M::AbstractMatrix,L::AbstractSparseMatrix,p::AbstractVector{Int})
  XM = L'*M[p,:]
  MXM = XM'*XM
  _,Sr,Vr = truncated_svd(red_style,MXM;issquare=true)
  Ũr = (XM*Vr)*inv(Diagonal(Sr).+eps())
  Ur = (L'\Ũr)[invperm(p),:]
  return Ur,Sr,Vr
end

function massive_cols_tpod(red_style::ReductionStyle,M::AbstractMatrix)
  MM = M*M'
  Ur,Sr,_ = truncated_svd(red_style,MM;issquare=true)
  Vr = inv(Diagonal(Sr).+eps())*(Ur'M)
  return Ur,Sr,Vr'
end

function massive_cols_tpod(red_style::ReductionStyle,M::AbstractMatrix,L::AbstractSparseMatrix,p::AbstractVector{Int})
  XM = L'*M[p,:]
  MXM = XM*XM'
  Ũr,Sr,_ = truncated_svd(red_style,MXM;issquare=true)
  Vr = inv(Diagonal(Sr).+eps())*(Ũr'XM)
  Ur = (L'\Ũr)[invperm(p),:]
  return Ur,Sr,Vr'
end

function ttsvd_loop(red_style::ReductionStyle,A::AbstractArray{T,3}) where T
  M = reshape(A,size(A,1)*size(A,2),:)
  Ur,Sr,Vr = tpod(red_style,M)
  core = reshape(Ur,size(A,1),size(A,2),:)
  remainder = Sr.*Vr'
  return core,remainder
end

# function ttsvd_loop(red_style::ReductionStyle,A::AbstractArray{T,3},X::AbstractSparseMatrix) where T
#   prev_rank = size(A,1)
#   cur_size = size(A,2)

#   L,p = _cholesky_decomp(X)

#   XA = _tt_mul(L,p,A)
#   XM = reshape(XA,:,size(XA,3))

#   if _size_cond(XM)
#     @check size(XM,2) > size(XM,1)
#     XMX = XM*XM'
#     Ũr,Sr,_ = truncated_svd(red_style,XMX;issquare=true)
#     c̃ = reshape(Ũr,prev_rank,cur_size,:)
#     core = _tt_div(L,p,c̃)
#     Ur = reshape(core,:,size(core,3))
#     Vrt = Ur'*XM
#     @inbounds for i = axes(Ur,2)
#       Vrt[:,i] /= Sr[i]+eps()
#     end
#     remainder = Sr.*Vrt
#   else
#     Ũr,Sr,Vr = standard_tpod(red_style,XM)
#     c̃ = reshape(Ũr,prev_rank,cur_size,:)
#     core = _tt_div(L,p,c̃)
#     remainder = Sr.*Vr'
#   end

#   return core,remainder
# end

function ttsvd_loop(red_style::ReductionStyle,A::AbstractArray{T,3},X::AbstractSparseMatrix) where T
  prev_rank = size(A,1)
  cur_size = size(A,2)
  M = reshape(A,prev_rank*cur_size,:)

  L,p = _cholesky_decomp(X)
  L′ = kron(I(prev_rank),L)
  p′ = vec(((collect(1:prev_rank).-1)*cur_size .+ p')')
  println(typeof(p′))
  Ur,Sr,Vr = tpod(red_style,M,L′,p′)

  core = reshape(Ur,prev_rank,cur_size,:)
  remainder = Sr.*Vr'
  return core,remainder
end

# function _tt_mul(L::AbstractSparseMatrix{T},p::Vector{Int},A::AbstractArray{T,3}) where T
#   @check size(L,1) == size(L,2) == size(A,2)
#   B = similar(A)
#   Ap = A[:,p,:]
#   @inbounds for i1 in axes(B,1)
#     B[i1,:,:] = L'*Ap[i1,:,:]
#   end
#   return B
# end

# function _tt_div(L::AbstractSparseMatrix{T},p::Vector{Int},A::AbstractArray{T,3}) where T
#   @check size(L,1) == size(L,2) == size(A,2)
#   B = similar(A)
#   @inbounds for i1 in axes(B,1)
#     B[i1,:,:] = L'\A[i1,:,:]
#   end
#   return B
# end

# We are not interested in the last dimension (corresponds to the parameter)

function ttsvd(red_style::ReductionStyle,A::AbstractArray{T,N}) where {T,N}
  cores = Array{T,3}[]
  oldrank = 1
  A_d = reshape(A,oldrank,size(A,1),:)
  for d in 1:N-1
    core_d,remainder_d = ttsvd_loop(red_style[d],A_d)
    oldrank = size(core_d,3)
    A_d = reshape(remainder_d,oldrank,size(A,d+1),:)
    push!(cores,core_d)
  end
  return cores,A_d
end

function ttsvd(red_style::ReductionStyle,A::AbstractArray{T,N},X::AbstractRank1Tensor) where {T,N}
  Nspace = length(get_factors(X))
  @check Nspace ≤ N-1

  cores = Array{T,3}[]
  oldrank = 1
  A_d = reshape(A,oldrank,size(A,1),:)
  for d in 1:Nspace
    core_d,remainder_d = ttsvd_loop(red_style[d],A_d,X[d])
    oldrank = size(core_d,3)
    A_d = reshape(remainder_d,oldrank,size(A,d+1),:)
    push!(cores,core_d)
  end

  return cores,A_d
end

function ttsvd(red_style::ReductionStyle,A::AbstractArray{T,N},X::AbstractRankTensor) where {T,N}
  cores_k,remainders_k = map(k -> ttsvd(red_style,A,X[k]),1:rank(X)) |> tuple_of_arrays
  cores = Array{T,3}[]
  for d in eachindex(first(cores_k))
    touched = d == 1 ? fill(true,rank(X)) : I(rank(X))
    cores_d = getindex.(cores_k,d)
    push!(cores,BlockCore(cores_d,touched))
  end
  R = orthogonalize!(cores,X)
  A_d = absorb(cat(remainders_k...;dims=1),R)
  return cores,A_d
end

function projection(
  red::TTSVDReduction,
  A::MultiValueSnapshots{T,N},
  X::AbstractRankTensor) where {T,N}

  red_style = ReductionStyle(red)
  cores,remainder = ttsvd(red_style,A,X)
  core_c,remainder_c = RBSteady.ttsvd_loop(red_style[N-1],remainder)
  push!(cores,core_c)
  return cores
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
    if d == length(cores)
      XW = _get_norm_matrix_from_weight(X,weight)
      core_d′,R = reduce_rank(core_d,XW)
      cores[d] = core_d′
      return R
    end
    next_core = cores[d+1]
    X_d = getindex.(decomp,d)
    core_d′,R = reduce_rank(core_d)
    cores[d] = core_d′
    cores[d+1] = absorb(next_core,R)
    weight = _weight_array(weight,core_d′,X_d)
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
