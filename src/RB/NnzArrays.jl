abstract type NnzArray{T,N} <: AbstractArray{T,N} end

Base.size(nza::NnzArray,idx...) = size(nza.nonzero_val,idx...)
Base.getindex(nza::NnzArray,idx...) = getindex(nza.nonzero_val,idx...)
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

function NnzArray(
  s::Snapshots{T};
  nparams=length(testitem(s.snaps))) where T

  @check all([length(vali) == nparams for vali in s.snaps])
  NnzMatrix(get_array(hcat(s.snaps...))...;nparams)
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

function Base.prod(nza1::NnzMatrix,nza2::NnzMatrix)
  nonzero_vals = nza1.nonzero_val' * nza2.nonzero_val
  NnzMatrix(nonzero_vals,nza1.nonzero_idx,nza1.nrows,nza1.nparams)
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
  nrows = nzm.nrows
  entire_idx = nonzero_idx[idx]
  entire_idx_rows,_ = from_vec_to_mat_idx(entire_idx,nrows)
  return entire_idx_rows
end

function compress(a::AbstractMatrix,nzm::NnzMatrix{T}) where T
  m = zeros(T,nzm.nrows,size(nzm,2))
  m[nzm.nonzero_idx,:] = nzm.nonzero_val
  [a'*v for v in eachcol(m)]
end

function compress(a::AbstractMatrix,b::AbstractMatrix,nzm::NnzMatrix)
  irow,icol = from_vec_to_mat_idx(nzm.nonzero_idx,nzm.nrows)
  ncols = maximum(icol)
  map(eachcol(nzm)) do nzv
    m = sparse(irow,icol,nzv,nzm.nrows,ncols)
    a'*m*b
  end
end

abstract type PODStyle end
struct DefaultPOD <: PODStyle end
struct SteadyPOD <: PODStyle end
struct TranposedPOD <: PODStyle end

function compress(nzm::NnzMatrix,norm_matrix=nothing;kwargs...)
  steady = num_time_dofs(nzm) == 1 ? SteadyPOD() : DefaultPOD()
  transposed = size(nzm,1) < size(nzm,2) ? TranposedPOD() : DefaultPOD()
  compress(nzm,steady,transposed,norm_matrix;kwargs...)
end

function compress(
  nzm::NnzMatrix,
  ::PODStyle,
  ::PODStyle,
  args...;
  kwargs...)

  basis_space,basis_time = transient_tpod(nzm,args...;kwargs...)
  basis_space,basis_time
end

function compress(
  nzm::NnzMatrix,
  ::DefaultPOD,
  ::TranposedPOD,
  args...;
  kwargs...)

  nzm_t = change_mode(nzm)
  basis_time,basis_space = transient_tpod(nzm_t,args...;kwargs...)
  basis_space,basis_time
end

for T in (:DefaultPOD,:TranposedPOD)
  @eval begin
    function compress(
      nzm::NnzMatrix,
      ::SteadyPOD,
      ::$T,
      args...;
      kwargs...)

      basis_space = tpod(nzm,args...;kwargs...)
      basis_time = ones(eltype(nzm),1,1)
      basis_space,basis_time
    end
  end
end

function tpod(nzm::NnzMatrix,args...;kwargs...)
  nonzero_val = tpod(nzm.nonzero_val,args...;kwargs...)
  NnzMatrix(nonzero_val,nzm.nonzero_idx,nzm.nrows,nzm.nparams)
end

function transient_tpod(nzm::NnzMatrix,args...;kwargs...)
  basis_axis1 = tpod(nzm,args...;kwargs...)
  compressed_nzm = prod(basis_axis1,nzm)
  compressed_nzm_t = change_mode(compressed_nzm)
  basis_axis2 = tpod(compressed_nzm_t;kwargs...)
  basis_axis1,basis_axis2
end

function change_mode(nzm::NnzMatrix{T}) where T
  time_ndofs = num_time_dofs(nzm)
  nparams = num_params(nzm)
  mode2 = change_mode(nzm.nonzero_val,time_ndofs,nparams)
  return NnzMatrix(mode2,nzm.nonzero_idx,nzm.nrows,nparams)
end

struct BlockNnzMatrix{T} <: AbstractVector{NnzMatrix{T}}
  blocks::Vector{NnzMatrix{T}}

  function BlockNnzMatrix(blocks::Vector{NnzMatrix{T}}) where T
    @check all([length(nzm) == length(blocks[1]) for nzm in blocks[2:end]])
    new{T}(blocks)
  end
end

function NnzArray(s::BlockSnapshots{T}) where T
  blocks = map(s.snaps) do val
    array = get_array(hcat(val...))
    NnzMatrix(array...)
  end
  BlockNnzMatrix(blocks)
end

Base.size(nzm::BlockNnzMatrix,idx...) = map(x->size(x,idx...),nzm.blocks)
Base.length(nzm::BlockNnzMatrix) = length(nzm.blocks[1])
Base.getindex(nzm::BlockNnzMatrix,idx...) = nzm.blocks[idx...]
Base.iterate(nzm::BlockNnzMatrix,args...) = iterate(nzm.blocks,args...)
get_nfields(nzm::BlockNnzMatrix) = length(nzm.blocks)
