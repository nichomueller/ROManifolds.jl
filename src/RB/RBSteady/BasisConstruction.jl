function reduction(red::Reduction,A::AbstractArray)
  @abstractmethod
end

function reduction(red::PODReduction,A::AbstractArray,args...)
  red_style = ReductionStyle(red)
  U,S,V = tpod(red_style,A,args...)
  return U
end

function reduction(red::PODReduction,A::SparseSnapshots,args...)
  red_style = ReductionStyle(red)
  U,S,V = tpod(red_style,A,args...)
  return recast(U,A)
end

function reduction(red::TTSVDReduction,A::AbstractArray,args...)
  red_style = ReductionStyle(red)
  cores,remainder = ttsvd(red_style,A,args...)
  return cores
end

function reduction(red::TTSVDReduction,A::SparseSnapshots,args...)
  red_style = ReductionStyle(red)
  cores,remainder = ttsvd(red_style,A,args...)
  return recast(cores,A)
end

function _size_cond(M::AbstractMatrix)
  length(M) > 1e6 && (size(M,1) > 1e1*size(M,2) || size(M,2) > 1e2*size(M,1)) #false #
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

function truncated_svd(red_style::LRApproxRank,M::AbstractMatrix;kwargs...)
  psvd(M,red_style.opts)
end

function tpod(red_style::ReductionStyle,M::AbstractMatrix,X::AbstractSparseMatrix)
  tpod(red_style,M,_cholesky_decomp(X)...)
end

function tpod(red_style::ReductionStyle,M::AbstractMatrix,args...)
  if _size_cond(M) && !isa(red_style,LRApproxRank)
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
  # @warn "possibly incorrect V matrix"
  MM = M*M'
  Ur,Sr,_ = truncated_svd(red_style,MM;issquare=true)
  Vr = inv(Diagonal(Sr).+eps())*(Ur'M)
  return Ur,Sr,Vr'
end

function massive_cols_tpod(red_style::ReductionStyle,M::AbstractMatrix,L::AbstractSparseMatrix,p::AbstractVector{Int})
  # @warn "possibly incorrect V matrix"
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

function ttsvd_loop(red_style::ReductionStyle,A::AbstractArray{T,3},X::AbstractSparseMatrix) where T
  prev_rank = size(A,1)
  cur_size = size(A,2)
  M = reshape(A,prev_rank*cur_size,:)

  L,p = _cholesky_decomp(X)
  L′ = kron(I(prev_rank),L)
  p′ = vec(((collect(1:prev_rank).-1)*cur_size .+ p')')
  Ur,Sr,Vr = tpod(red_style,M,L′,p′)

  core = reshape(Ur,prev_rank,cur_size,:)
  remainder = Sr.*Vr'
  return core,remainder
end

# We are not interested in the last dimension (corresponds to the parameter)

function ttsvd(
  red_style::TTSVDRanks,
  A::AbstractArray{T,N}
  ) where {T,N}

  cores = Array{T,3}[]
  oldrank = 1
  remainder = reshape(A,oldrank,size(A,1),:)
  for d in 1:N-1
    core_d,remainder_d = ttsvd_loop(red_style[d],remainder)
    oldrank = size(core_d,3)
    remainder::Array{T,3} = reshape(remainder_d,oldrank,size(A,d+1),:)
    push!(cores,core_d)
  end
  return cores,remainder
end

function ttsvd(
  red_style::TTSVDRanks,
  A::AbstractArray{T,N},
  X::AbstractRankTensor{D}) where {T,N,D}

  @check D ≤ N-1
  if D == N - 1
    steady_ttsvd(red_style,A,X)
  else
    generalized_ttsvd(red_style,A,X)
  end
end

function steady_ttsvd(
  red_style::TTSVDRanks,
  A::AbstractArray{T,N},
  X::Rank1Tensor{D}
  ) where {T,N,D}

  cores = Array{T,3}[]
  oldrank = 1
  remainder = reshape(A,oldrank,size(A,1),:)
  for d in 1:D
    core_d,remainder_d = ttsvd_loop(red_style[d],remainder,X[d])
    oldrank = size(core_d,3)
    remainder::Array{T,3} = reshape(remainder_d,oldrank,size(A,d+1),:)
    push!(cores,core_d)
  end

  return cores,remainder
end

function steady_ttsvd(
  red_style::TTSVDRanks,
  A::AbstractArray{T,N},
  X::GenericRankTensor{D,K}
  ) where {T,N,D,K}

  # compute initial tt decompositions
  cores_k,remainders_k = map(k -> steady_ttsvd(red_style,A,X[k]),1:K) |> tuple_of_arrays

  # tt decomposition of the sum
  cores = block_cores(cores_k...)
  remainder::Array{T,3} = cat(remainders_k...;dims=1)

  # tt orthogonality
  remainder′ = orthogonalize!(red_style,cores,remainder,X)

  return cores,remainder′
end

function generalized_ttsvd(
  red_style::TTSVDRanks,
  A::AbstractArray{T,N},
  X::AbstractRankTensor{D}
  ) where {T,N,D}

  cores,remainder = steady_ttsvd(red_style,A,X)
  for d = D+1:N-1
    core_d,remainder_d = RBSteady.ttsvd_loop(red_style[d],remainder)
    remainder = reshape(remainder_d,size(core_d,3),size(A,d+1),:)
    push!(cores,core_d)
  end

  return cores,remainder
end

function orthogonalize!(red_style,cores,remainder,X::AbstractRankTensor)
  weight = ones(1,rank(X),1)
  decomp = get_decomposition(X)
  for d in eachindex(cores)
    core_d = cores[d]
    if d == length(cores)
      XW = ttnorm_array(X,weight)
      core_d′,R = reduce_rank(red_style[d],core_d,XW)
      cores[d] = core_d′
      remainder = absorb(remainder,R)
    else
      next_core = cores[d+1]
      X_d = getindex.(decomp,d)
      core_d′,R = reduce_rank(red_style[d],core_d)
      cores[d] = core_d′
      cores[d+1] = absorb(next_core,R)
      weight = weight_array(weight,core_d′,X_d)
    end
  end
  return remainder
end

function orthogonalize!(red_style,cores,remainder)
  for d in eachindex(cores)
    if d == length(cores)
      remainder = absorb(remainder,R)
    else
      cur_core = cores[d]
      next_core = cores[d+1]
      cur_core′,R = reduce_rank(red_style[d],cur_core)
      next_core′ = absorb(next_core,R)
      cores[d] = cur_core′
      cores[d+1] = next_core′
    end
  end
  return remainder
end

function reduce_rank(red_style,core::AbstractArray{T,3},args...) where T
  mat = reshape(core,:,size(core,3))
  Ur,Sr,Vr = tpod(red_style,mat,args...)
  core′ = reshape(Ur,size(core,1),size(core,2),:)
  R = Sr.*Vr'
  return core′,R
end

function absorb(core::AbstractArray{T,3},R::AbstractMatrix) where T
  Rcore = R*reshape(core,size(core,1),:)
  return reshape(Rcore,size(Rcore,1),size(core,2),:)
end

function weight_array(prev_weight,core,X)
  @check length(X) == size(prev_weight,2)
  @check size(core,1) == size(prev_weight,1) == size(prev_weight,3)

  K = length(X)
  rank_prev = size(core,1)
  rank = size(core,3)
  rrprev = rank_prev*rank
  N = size(core,2)

  cur_weight = zeros(rank,K,rank)
  core2D = reshape(permutedims(core,(2,1,3)),N,rrprev)
  cache_right = zeros(N,rrprev)
  cache_left = zeros(rrprev,rrprev)

  @inbounds for k = 1:K
    Xk = X[k]
    Wk = cur_weight[:,k,:]
    Wk_prev = prev_weight[:,k,:]
    mul!(cache_right,Xk,core2D)
    mul!(cache_left,core2D',cache_right)
    resh_weight = reshape(permutedims(reshape(cache_left,rank_prev,rank,rank_prev,rank),(2,4,1,3)),rank^2,:)
    cur_weight[:,k,:] = reshape(resh_weight*vec(Wk_prev),rank,rank)
  end
  return cur_weight
end

function ttnorm_array(X::AbstractRankTensor{D,K},WD) where {D,K}
  @check size(WD,1) == size(WD,3)
  @check size(WD,2) == K
  @check all(size(X[1][D]) == size(X[k][D]) for k = 2:K)

  s1 = size(WD,1)*size(X[1][D],1)
  s2 = size(WD,3)*size(X[1][D],2)
  XW = zeros(s1,s2)
  cache = zeros(s1,s2)

  for k = 1:rank(X)
    kron!(cache,X[k][D],WD[:,k,:])
    @. XW = XW + cache
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

for (f,g) in zip((:pivoted_qr,:pivoted_qr!),(:qr,:qr!))
  @eval begin
    function $f(A;tol=1e-10)
      C = $g(A,ColumnNorm())
      r = findlast(abs.(diag(C.R)) .> tol)
      Q = C.Q[:,1:r]
      R = C.R[1:r,invperm(C.jpvt)]
      return Q,R
    end
  end
end

for (f,g) in zip((:gram_schmidt,:gram_schmidt!),(:pivoted_qr,:pivoted_qr!))
  @eval begin
    function $f(M::AbstractMatrix)
      Q,R = $g(M)
      return Q,R
    end

    function $f(M::AbstractMatrix,X::AbstractSparseMatrix)
      L,p = _cholesky_decomp(X)
      XM = L'*M[p,:]
      Q̃,R = $g(XM)
      Q = (L'\Q̃)[invperm(p),:]
      return Q,R
    end
  end
end

for f in (:gram_schmidt,:gram_schmidt!)
  @eval begin
    function $f(M::AbstractMatrix,basis::AbstractMatrix,args...)
      Q,R = $f(hcat(basis,M),args...)
      return Q,R
    end
  end
end

# for testing purposes

function check_orthogonality(cores::AbstractVector{<:AbstractArray{T,3}},X::AbstractRankTensor) where T
  Xglobal = kron(X)
  basis = dropdims(_cores2basis(cores...);dims=1)
  isorth = norm(basis'*Xglobal*basis - I) ≤ 1e-10
  return isorth
end
