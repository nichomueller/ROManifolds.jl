abstract type AbstractNnzArray{T,N} <: AbstractArray{T,N} end
const AbstractNnzVector{T} = AbstractNnzArray{T,1}
const AbstractNnzMatrix{T} = AbstractNnzArray{T,2}

Base.size(nza::AbstractNnzArray,idx...) = size(nza.nonzero_val,idx...)
Base.getindex(nza::AbstractNnzArray,idx...) = getindex(nza.nonzero_val,idx...)
Base.eachcol(nza::AbstractNnzArray) = eachcol(nza.nonzero_val)
get_nonzero_val(nza::AbstractNnzArray) = nza.nonzero_val
get_nonzero_idx(nza::AbstractNnzArray) = nza.nonzero_idx
get_nrows(nza::AbstractNnzArray) = nza.nrows

function get_nonzero_val(nza::NTuple{N,AbstractNnzArray}) where N
  hcat(map(get_nonzero_val,nza)...)
end

function get_nonzero_idx(nza::NTuple{N,AbstractNnzArray}) where N
  nz_idx = map(get_nonzero_idx,nza)
  @check all([i == first(nz_idx) for i in nz_idx])
  first(nz_idx)
end

function get_nrows(nza::NTuple{N,AbstractNnzArray}) where N
  nrows = map(get_nrows,nza)
  @check all([r == first(nrows) for r in nrows])
  first(nrows)
end

struct NnzVector{T} <: AbstractNnzVector{T}
  nonzero_val::Vector{T}
  nonzero_idx::Vector{Int}
  nrows::Int

  function NnzVector(
    nonzero_val::Matrix{T},
    nonzero_idx::Vector{Int},
    nrows::Int) where T

    new{T}(nonzero_val,nonzero_idx,nrows)
  end

  function NnzVector(mat::SparseMatrixCSC{T,Int}) where T
    nonzero_idx,nonzero_val = findnz(mat[:])
    nrows = size(mat,1)
    new{T}(nonzero_val,nonzero_idx,nrows)
  end
end

Base.length(nzv::NnzVector) = length(nzv.nonzero_val)

struct NnzMatrix{T} <: AbstractNnzMatrix{T}
  nonzero_val::Matrix{T}
  nonzero_idx::Vector{Int}
  nrows::Int
  nparams::Int

  function NnzMatrix(
    nonzero_val::Matrix{T},
    nonzero_idx::Vector{Int},
    nrows::Int,
    nparams::Int) where T

    new{T}(nonzero_val,nonzero_idx,nrows,nparams)
  end

  function NnzMatrix(val::AbstractArray{T}...;nparams=length(val)) where T
    vals = hcat(val...)
    nonzero_idx,nonzero_val = compress_array(vals)
    nrows = size(vals,1)
    new{T}(nonzero_val,nonzero_idx,nrows,nparams)
  end

  function NnzMatrix(val::NnzVector{T}...;nparams=length(val)) where T
    nonzero_val = get_nonzero_val(val)
    nonzero_idx = get_nonzero_idx(val)
    nrows = get_nrows(val)
    new{T}(nonzero_val,nonzero_idx,nrows,nparams)
  end

  function NnzMatrix(val::PTArray;nparams=length(val))
    NnzMatrix(get_array(val)...;nparams)
  end

  function NnzMatrix(val::Vector{<:PTArray};nparams=length(testitem(val)))
    @check all([length(vali) == nparams for vali in val])
    NnzMatrix(get_array(hcat(val...))...;nparams)
  end
end

Base.length(nza::AbstractNnzArray) = length(nza.nparams)
num_params(nzm::NnzMatrix) = nzm.nparams
num_space_dofs(nzm::NnzMatrix) = size(nzm,1)
num_time_dofs(nzm::NnzMatrix) = Int(size(nzm,2)/nzm.nparams)

function Base.copy(nzm::NnzMatrix)
  NnzMatrix(
    copy(nzm.nonzero_val),
    copy(nzm.nonzero_idx),
    copy(nzm.nrows),
    copy(nzm.nparams))
end

function Base.show(io::IO,nzm::NnzMatrix)
  print(io,"NnzMatrix storing $(length(nzm)) compressed transient snapshots")
end

function Base.prod(nza1::NnzMatrix,nza2::NnzMatrix)
  nonzero_vals = nza1.nonzero_val' * nza2.nonzero_val
  NnzMatrix(nonzero_vals,nza1.nonzero_idx,nza1.nrows,nza1.nparams)
end

function Base.prod(nzm::NnzMatrix,a::AbstractArray)
  nonzero_vals = nzm.nonzero_vals' * a
  NnzMatrix(nonzero_vals,nzm.nonzero_idx,nzm.nrows,nzm.nparams)
end

function Base.prod(a::AbstractArray,nzm::NnzMatrix)
  nonzero_vals = a' * nzm.nonzero_vals
  NnzMatrix(nonzero_vals,nzm.nonzero_idx,nzm.nrows,nzm.nparams)
end

function recast(nzm::NnzMatrix{T}) where T
  m = zeros(T,nzm.nrows,size(nzm,2))
  m[nzm.nonzero_idx,:] = nzm.nonzero_val
  m
end

function reduce(a::AbstractMatrix,nzm::NnzMatrix{T}) where T
  m = zeros(T,nzm.nrows,size(nzm,2))
  m[nzm.nonzero_idx,:] = nzm.nonzero_val
  [a'*v for v in eachcol(m)]
end

function reduce(a::AbstractMatrix,b::AbstractMatrix,nzm::NnzMatrix)
  irow,icol = from_vec_to_mat_idx(nzm.nonzero_idx,nzm.nrows)
  ncols = maximum(icol)
  map(eachcol(nzm)) do nzv
    m = sparse(irow,icol,nzv,nzm.nrows,ncols)
    a'*m*b
  end
end

# function attach_initial_condition(nzm::NnzMatrix,a0::PTArray)
#   nt = num_time_dofs(nzm)
#   np = num_params(nzm)
#   minus_last_nnz = nzm.nonzero_val[:,1:nt*(np-1)]
#   a0_nnz = map(x->getindex(x,nzm.nonzero_idx),a0.array)

#   nonzero_val = hcat(hcat(a0_nnz...),minus_last_nnz)
#   NnzMatrix(nonzero_val,nzm.nonzero_idx,nzm.nrows,nzm.nparams)
# end

function recast_index(nzm::NnzMatrix,idx::Vector{Int})
  nonzero_idx = nzm.nonzero_idx
  nrows = nzm.nrows
  entire_idx = nonzero_idx[idx]
  entire_idx_rows,_ = from_vec_to_mat_idx(entire_idx,nrows)
  return entire_idx_rows
end

function change_mode(nzm::NnzMatrix{T}) where T
  space_ndofs = num_space_dofs(nzm)
  time_ndofs = num_time_dofs(nzm)
  nparams = num_params(nzm)
  mode2 = zeros(T,time_ndofs,space_ndofs*nparams)

  _mode2(n::Int) = nzm.nonzero_val[:,n:nparams:nparams*(time_ndofs-1)+n]'
  @inbounds for n = 1:time_ndofs
    mode2[n,:] = reshape(nzm.nonzero_val[:,(n-1)*nparams+1:n*nparams]',:)
  end

  return NnzMatrix(mode2,nzm.nonzero_idx,nzm.nrows,nparams)
end

function tpod(nzm::NnzMatrix,args...;kwargs...)
  nonzero_val = tpod(nzm.nonzero_val,args...;kwargs...)
  return NnzMatrix(nonzero_val,nzm.nonzero_idx,nzm.nrows,nzm.nparams)
end
