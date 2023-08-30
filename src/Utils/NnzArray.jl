struct NnzArray{T,N,OT} <: AbstractArray{T,N}
  nonzero_val::AbstractArray{T,N}
  nonzero_idx::Vector{Int}
  nrows::Int

  function NnzArray{OT}(
    nonzero_val::AbstractArray{T,N},
    nonzero_idx::Vector{Int},
    nrows::Int) where {T,N,OT}

    new{T,N,OT}(nonzero_val,nonzero_idx,nrows)
  end
end

get_nonzero_val(nza::NnzArray) = nza.nonzero_val

Base.getindex(nza::NnzArray,idx...) = getindex(nza.nonzero_val,idx...)

function compress(entire_array::OT) where {OT<:AbstractArray}
  nonzero_idx,nonzero_val = compress_array(entire_array)
  nrows = size(entire_array,1)
  NnzArray{OT}(nonzero_val,nonzero_idx,nrows)
end

function compress(entire_arrays::Vector{OT}) where {OT<:AbstractArray}
  entire_array = reduce(hcat,entire_arrays)
  compress(entire_array)
end

function Base.show(io::IO,nza::NnzArray)
  print(io,"NnzArray storing $(length(nza.nonzero_idx)) nonzero values")
end

Base.size(nza::NnzArray,idx...) = size(nza.nonzero_val,idx...)

function Base.hcat(nza::NnzArray{T,N,OT}...) where {T,N,OT}
  msg = """\n
  Cannot hcat the given NnzArrays: the nonzero indices and/or the full
  order number of rows do not match one another.
  """
  nonzero_vals = map(x->getproperty(x,:nonzero_val),nza)
  nonzero_idxs = map(x->getproperty(x,:nonzero_idx),nza)
  nrowss = map(x->getproperty(x,:nrows),nza)

  @assert all([idx == first(nonzero_idxs) for idx in nonzero_idxs]) msg
  @assert all([nrow == first(nrowss) for nrow in nrowss]) msg

  NnzArray{OT}(reduce(hcat,nonzero_vals),first(nonzero_idxs),first(nrowss))
end

function recast(nza::NnzArray{T,N,<:AbstractMatrix}) where {T,N}
  entire_array = zeros(T,nza.nrows,size(nza,2))
  entire_array[nza.nonzero_idx,:] = nza.nonzero_val
  entire_array
end

function recast(nza::NnzArray{T,N,<:SparseMatrixCSC} where {T,N},col=1)
  sparse_rows,sparse_cols = from_vec_to_mat_idx(nza.nonzero_idx,nza.nrows)
  ncols = maximum(sparse_cols)
  sparse(sparse_rows,sparse_cols,nza.nonzero_val[:,col],nza.nrows,ncols)
end

function change_mode(nza::NnzArray{T,N,OT},nparams::Int) where {T,N,OT}
  mode1_ndofs = size(nza,1)
  mode2_ndofs = Int(size(nza,2)/nparams)
  mode2 = zeros(T,mode2_ndofs,mode1_ndofs*nparams)

  _mode2(k::Int) = nza.nonzero_val[:,(k-1)*mode2_ndofs+1:k*mode2_ndofs]'
  @inbounds for k = 1:nparams
    setindex!(mode2,_mode2(k),:,(k-1)*mode1_ndofs+1:k*mode1_ndofs)
  end

  return NnzArray{OT}(mode2,nza.nonzero_idx,nza.nrows)
end

function tpod(nza::NnzArray{T,N,OT};kwargs...) where {T,N,OT}
  nonzero_val = tpod(nza.nonzero_val;kwargs...)
  return NnzArray{OT}(nonzero_val,nza.nonzero_idx,nza.nrows)
end
