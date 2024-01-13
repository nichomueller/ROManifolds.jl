abstract type NnzArray{T,N} <: AbstractArray{T,N} end

Base.size(nza::NnzArray,idx...) = size(nza.nonzero_val,idx...)
Base.getindex(nza::NnzArray,idx...) = nza.nonzero_val[idx...]
Base.eachcol(nza::NnzArray) = eachcol(nza.nonzero_val)
get_nonzero_val(nza::NnzArray) = nza.nonzero_val
get_nonzero_idx(nza::NnzArray) = nza.nonzero_idx
get_nrows(nza::NnzArray) = nza.nrows

struct NnzVector{T} <: NnzArray{T,1}
  nonzero_val::Vector{T}
  nonzero_idx::Vector{Int}
  nrows::Int
end

function NnzVector(mat::SparseMatrixCSC{T,Int}) where T
  nonzero_idx,nonzero_val = findnz(mat[:])
  nrows = size(mat,1)
  NnzVector{T}(nonzero_val,nonzero_idx,nrows)
end

function NnzVector(mat::AbstractArray{<:SparseMatrixCSC})
  map(NnzVector,mat)
end

Base.length(nzv::NnzVector) = length(nzv.nonzero_val)

struct NnzMatrix{T,A} <: NnzArray{T,2}
  affinity::A
  nonzero_val::Matrix{T}
  nonzero_idx::Vector{Int}
  nrows::Int
  nparams::Int
end

function NnzMatrix(val::PArray{<:AbstractVector{T}};nparams=length(val),kwargs...) where T
  vals = get_array(val)
  nonzero_idx,nonzero_val = compress_array(stack(vals))
  nrows = size(testitem(val),1)
  NnzMatrix(Nonlinear(),nonzero_val,nonzero_idx,nrows,nparams)
end

function NnzMatrix(val::PArray{<:NnzVector{T}};nparams=length(val),kwargs...) where T
  vals = get_array(val)
  nonzero_idx = get_nonzero_idx(first(vals))
  nonzero_val = stack(map(get_nonzero_val,vals))
  nrows = get_nrows(first(vals))
  NnzMatrix(Nonlinear(),nonzero_val,nonzero_idx,nrows,nparams)
end

function NnzMatrix(val::AbstractVector{T};ntimes=1,kwargs...) where T
  nonzero_idx,nonzero_val = compress_array(val)
  nonzero_val = repeat(val,1,ntimes)
  nrows = size(val,1)
  NnzMatrix(Affine(),nonzero_val,nonzero_idx,nrows,1)
end

function NnzMatrix(val::NnzVector{T};ntimes=1,kwargs...) where T
  nonzero_idx = get_nonzero_idx(val)
  nonzero_val = repeat(get_nonzero_val(val),1,ntimes)
  nrows = get_nrows(val)
  NnzMatrix(Affine(),nonzero_val,nonzero_idx,nrows,1)
end

Base.length(nzm::NnzMatrix) = nzm.nparams
num_params(nzm::NnzMatrix) = length(nzm)
num_space_dofs(nzm::NnzMatrix) = size(nzm,1)
FEM.num_time_dofs(nzm::NnzMatrix) = Int(size(nzm,2)/length(nzm))

function Base.copy(nzm::NnzMatrix)
  NnzMatrix(nzm.affinity,copy(nzm.nonzero_val),nzm.nonzero_idx,nzm.nrows,nzm.nparams)
end

function Base.show(io::IO,nzm::NnzMatrix)
  print(io,"NnzMatrix storing $(length(nzm)) compressed transient snapshots")
end

function Base.prod(nzm1::T,nzm2::T) where {T<:NnzMatrix}
  @assert nzm1.nonzero_idx == nzm2.nonzero_idx
  @assert nzm1.nrows == nzm2.nrows
  @assert nzm1.nparams == nzm2.nparams

  nonzero_vals = nzm1.nonzero_val' * nzm2.nonzero_val
  NnzMatrix(nzm1.affinity,nonzero_vals,nzm1.nonzero_idx,nzm1.nrows,nzm1.nparams)
end

function Base.prod(nzm::NnzMatrix,a::AbstractArray)
  nonzero_vals = nzm.nonzero_val' * a
  NnzMatrix(nzm.affinity,nonzero_vals,nzm.nonzero_idx,nzm.nrows,nzm.nparams)
end

function Base.prod(a::AbstractArray,nzm::NnzMatrix)
  nonzero_vals = a' * nzm.nonzero_val
  NnzMatrix(nzm.affinity,nonzero_vals,nzm.nonzero_idx,nzm.nrows,nzm.nparams)
end

function Arrays.testitem(nzm::NnzMatrix)
  mode2_ndofs = Int(size(nzm,2)/nzm.nparams)
  NnzMatrix(nzm.affinity,nzm.nonzero_val[:,1:mode2_ndofs],nzm.nonzero_idx,nzm.nrows,1)
end

function recast(nzm::NnzMatrix{T}) where T
  m = zeros(T,nzm.nrows,size(nzm,2))
  m[nzm.nonzero_idx,:] = nzm.nonzero_val
  m
end

function recast(nzm::NnzMatrix,idx::Vector{Int})
  nonzero_idx = nzm.nonzero_idx
  entire_idx = nonzero_idx[idx]
  return entire_idx
end

function compress(a::AbstractMatrix,nzm::NnzMatrix{T}) where T
  m = zeros(T,nzm.nrows,size(nzm,2))
  m[nzm.nonzero_idx,:] = nzm.nonzero_val
  [a'*v for v in eachcol(m)]
end

function compress(a::AbstractMatrix,b::AbstractMatrix,nzm::NnzMatrix)
  irow,icol = vec_to_mat_idx(nzm.nonzero_idx,nzm.nrows)
  ncols = maximum(icol)
  map(eachcol(nzm)) do nzv
    m = sparse(irow,icol,nzv,nzm.nrows,ncols)
    a'*m*b
  end
end

function compress(nzm::NnzMatrix,args...;kwargs...)
  basis_space = tpod(nzm,args...;kwargs...)
  compressed_nzm = prod(basis_space,nzm)
  compressed_nzm_t = change_mode(compressed_nzm)
  basis_time = tpod(compressed_nzm_t;kwargs...)
  basis_space,basis_time
end

function Utils.tpod(nzm::NnzMatrix,args...;ϵ=1e-4,kwargs...)
  nonzero_val = tpod(nzm.nonzero_val,args...;ϵ)
  NnzMatrix(nzm.affinity,nonzero_val,nzm.nonzero_idx,nzm.nrows,nzm.nparams)
end

function Utils.change_mode(nzm::NnzMatrix)
  nparams = num_params(nzm)
  mode2 = change_mode(nzm.nonzero_val,nparams)
  return mode2
end

function collect_residuals_for_trian(op::NonlinearOperator)
  b = allocate_residual(op,op.u0)
  ress,trian = residual_for_trian!(b,op,op.u0)
  nzm = map(ress) do res
    NnzMatrix(res;nparams=length(op.μ),ntimes=length(op.t))
  end
  return nzm,trian
end

function collect_jacobians_for_trian(op::NonlinearOperator;i=1)
  A = allocate_jacobian(op,op.u0,i)
  jacs_i,trian = jacobian_for_trian!(A,op,op.u0,i)
  nzm_i = map(jacs_i) do jac_i
    NnzMatrix(NnzVector(jac_i);nparams=length(op.μ),ntimes=length(op.t))
  end
  return nzm_i,trian
end
