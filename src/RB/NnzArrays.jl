abstract type NnzArray{T,N} <: AbstractArray{T,N} end

Base.size(nza::NnzArray,idx...) = size(nza.nonzero_val,idx...)
Base.getindex(nza::NnzArray,idx...) = nza.nonzero_val[idx...]
Base.eachcol(nza::NnzArray) = eachcol(nza.nonzero_val)
get_nonzero_val(nza::NnzArray) = nza.nonzero_val
get_nonzero_idx(nza::NnzArray) = nza.nonzero_idx
get_nrows(nza::NnzArray) = nza.nrows

function get_nonzero_val(nza::NTuple{N,NnzArray}) where N
  hcat(map(get_nonzero_val,nza)...)
end

function get_nonzero_idx(nza::NTuple{N,NnzArray}) where N
  nz_idx = map(get_nonzero_idx,nza)
  @check all([i == first(nz_idx) for i in nz_idx])
  first(nz_idx)
end

function get_nrows(nza::NTuple{N,NnzArray}) where N
  nrows = map(get_nrows,nza)
  @check all([r == first(nrows) for r in nrows])
  first(nrows)
end

struct NnzVector{T} <: NnzArray{T,1}
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

struct NnzMatrix{T} <: NnzArray{T,2}
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
end

Base.length(nzm::NnzMatrix) = nzm.nparams
num_params(nzm::NnzMatrix) = length(nzm)
num_space_dofs(nzm::NnzMatrix) = size(nzm,1)
num_time_dofs(nzm::NnzMatrix) = Int(size(nzm,2)/length(nzm))

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

function Base.prod(nzm1::NnzMatrix,nzm2::NnzMatrix)
  @assert nzm1.nonzero_idx == nzm2.nonzero_idx
  @assert nzm1.nrows == nzm2.nrows
  @assert nzm1.nparams == nzm2.nparams

  nonzero_vals = nzm1.nonzero_val' * nzm2.nonzero_val
  NnzMatrix(nonzero_vals,nzm1.nonzero_idx,nzm1.nrows,nzm1.nparams)
end

function Base.prod(nzm::NnzMatrix,a::AbstractArray)
  nonzero_vals = nzm.nonzero_val' * a
  NnzMatrix(nonzero_vals,nzm.nonzero_idx,nzm.nrows,nzm.nparams)
end

function Base.prod(a::AbstractArray,nzm::NnzMatrix)
  nonzero_vals = a' * nzm.nonzero_val
  NnzMatrix(nonzero_vals,nzm.nonzero_idx,nzm.nrows,nzm.nparams)
end

function recast(nzm::NnzMatrix{T}) where T
  m = zeros(T,nzm.nrows,size(nzm,2))
  m[nzm.nonzero_idx,:] = nzm.nonzero_val
  m
end

function recast_idx(nzm::NnzMatrix,idx::Vector{Int})
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

abstract type PODStyle end
struct DefaultPOD <: PODStyle end
struct SteadyPOD <: PODStyle end

function compress(nzm::NnzMatrix,args...;kwargs...)
  steady = num_time_dofs(nzm) == 1 ? SteadyPOD() : DefaultPOD()
  compress(nzm,steady,args...;kwargs...)
end

function compress(nzm::NnzMatrix,::DefaultPOD,args...;kwargs...)
  basis_space = tpod(nzm,args...;kwargs...)
  compressed_nzm = prod(basis_space,nzm)
  compressed_nzm_t = change_mode(compressed_nzm)
  basis_time = tpod(compressed_nzm_t;kwargs...)
  basis_space,basis_time
end

function compress(nzm::NnzMatrix,::SteadyPOD,args...;kwargs...)
  basis_space = tpod(nzm,args...;kwargs...)
  basis_time = ones(eltype(nzm),1,1)
  basis_space,basis_time
end

function tpod(nzm::NnzMatrix,args...;kwargs...)
  nonzero_val = tpod(nzm.nonzero_val,args...;kwargs...)
  NnzMatrix(nonzero_val,nzm.nonzero_idx,nzm.nrows,nzm.nparams)
end

function change_mode(nzm::NnzMatrix{T}) where T
  nparams = num_params(nzm)
  mode2 = change_mode(nzm.nonzero_val,nparams)
  return mode2
end
