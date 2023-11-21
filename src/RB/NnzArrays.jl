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

  function NnzVector(mat::SparseMatrixCSC{T,Int}) where T
    nonzero_idx,nonzero_val = findnz(mat[:])
    nrows = size(mat,1)
    new{T}(nonzero_val,nonzero_idx,nrows)
  end
end

Base.length(nzv::NnzVector) = length(nzv.nonzero_val)

struct NnzMatrix{T,A} <: NnzArray{T,2}
  nonzero_val::Matrix{T}
  nonzero_idx::Vector{Int}
  nrows::Int
  nparams::Int

  function NnzMatrix{A}(
    nonzero_val::Matrix{T},
    nonzero_idx::Vector{Int},
    nrows::Int,
    nparams::Int) where {T,A}

    new{T,A}(nonzero_val,nonzero_idx,nrows,nparams)
  end
end

function NnzMatrix(val::NonaffinePTArray{<:AbstractArray};nparams=length(val))
  vals = get_array(val)
  idx_val = map(compress_array,vals)
  nonzero_idx = first(first.(idx_val))
  nonzero_val = stack(last.(idx_val))
  nrows = size(testitem(val),1)
  NnzMatrix{Nonaffine}(nonzero_val,nonzero_idx,nrows,nparams)
end

function NnzMatrix(val::NonaffinePTArray{<:NnzVector};nparams=length(val))
  vals = get_array(val)
  nonzero_idx = get_nonzero_idx(first(vals))
  nonzero_val = stack(map(get_nonzero_val,vals))
  nrows = get_nrows(first(vals))
  NnzMatrix{Nonaffine}(nonzero_val,nonzero_idx,nrows,nparams)
end

function NnzMatrix(val::AffinePTArray{<:AbstractArray};nparams=length(val))
  v = get_array(val)
  nonzero_idx,nonzero_val = compress_array(v)
  nonzero_val = repeat(v,1,val.len)
  nrows = size(first(val),1)
  NnzMatrix{Affine}(nonzero_val,nonzero_idx,nrows,nparams)
end

function NnzMatrix(val::AffinePTArray{<:NnzVector};nparams=length(val))
  v = get_array(val)
  nonzero_idx = get_nonzero_idx(v)
  nonzero_val = repeat(get_nonzero_val(v),1,val.len)
  nrows = get_nrows(v)
  NnzMatrix{Affine}(nonzero_val,nonzero_idx,nrows,nparams)
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
  NnzMatrix{Nonaffine}(nonzero_vals,nzm1.nonzero_idx,nzm1.nrows,nzm1.nparams)
end

function Base.prod(nzm1::NnzMatrix{T,Affine} where T,nzm2::NnzMatrix{T,Affine} where T)
  @assert nzm1.nonzero_idx == nzm2.nonzero_idx
  @assert nzm1.nrows == nzm2.nrows
  @assert nzm1.nparams == nzm2.nparams

  nonzero_vals = nzm1.nonzero_val' * nzm2.nonzero_val
  NnzMatrix{Affine}(nonzero_vals,nzm1.nonzero_idx,nzm1.nrows,nzm1.nparams)
end

function Base.prod(nzm::NnzMatrix{T,A} where T,a::AbstractArray) where A
  nonzero_vals = nzm.nonzero_val' * a
  NnzMatrix{A}(nonzero_vals,nzm.nonzero_idx,nzm.nrows,nzm.nparams)
end

function Base.prod(a::AbstractArray,nzm::NnzMatrix{T,A} where T) where A
  nonzero_vals = a' * nzm.nonzero_val
  NnzMatrix{A}(nonzero_vals,nzm.nonzero_idx,nzm.nrows,nzm.nparams)
end

function Arrays.testitem(nzm::NnzMatrix{T,A} where T) where A
  mode2_ndofs = Int(size(nzm,2)/nzm.nparams)
  NnzMatrix{A}(nzm.nonzero_val[:,1:mode2_ndofs],nzm.nonzero_idx,nzm.nrows,1)
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

function compress(nzm::NnzMatrix,args...;kwargs...)
  basis_space = tpod(nzm,args...;kwargs...)
  compressed_nzm = prod(basis_space,nzm)
  compressed_nzm_t = change_mode(compressed_nzm)
  basis_time = tpod(compressed_nzm_t;kwargs...)
  basis_space,basis_time
end

function tpod(nzm::NnzMatrix{T,A} where T,args...;ϵ=1e-4,kwargs...) where A
  nonzero_val = tpod(nzm.nonzero_val,args...;ϵ)
  NnzMatrix{A}(nonzero_val,nzm.nonzero_idx,nzm.nrows,nzm.nparams)
end

function change_mode(nzm::NnzMatrix{T}) where T
  nparams = num_params(nzm)
  mode2 = change_mode(nzm.nonzero_val,nparams)
  return mode2
end
