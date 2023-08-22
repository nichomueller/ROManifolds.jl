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

function recast(nza::NnzArray{T,N,<:AbstractMatrix} where {T,N})
  entire_array = zeros(nza.nrows,size(nza,2))
  entire_array[nza.nonzero_idx,:] = nza.nonzero_val
  entire_array
end

function recast(nza::NnzArray{T,N,<:SparseMatrixCSC} where {T,N},col=1)
  sparse_rows,sparse_cols = from_vec_to_mat_idx(nza.nonzero_idx,nza.nrows)
  ncols = maximum(sparse_cols)
  sparse(sparse_rows,sparse_cols,nza.nonzero_val[:,col],nza.nrows,ncols)
end

# function change_mode!(nza::NnzArray,nparams::Int)
#   mode1_ndofs = size(nza,1)
#   mode2_ndofs = Int(size(nza,2)/nparams)

#   mode2 = reshape(similar(nza.nonzero_val),mode2_ndofs,mode1_ndofs*nparams)
#   _mode2(k::Int) = nza.nonzero_val[:,(k-1)*mode2_ndofs+1:k*mode2_ndofs]'
#   @inbounds for k = 1:nparams
#     setindex!(mode2,_mode2(k),:,(k-1)*mode1_ndofs+1:k*mode1_ndofs)
#   end

#   nza.nonzero_val = mode2
#   return
# end

# function change_mode(nza::NnzArray,nparams::Int)
#   nzm_copy = copy(nza)
#   change_mode!(nzm_copy,nparams)
#   nzm_copy
# end

# function tpod!(nza::NnzArray;kwargs...)
#   nza.nonzero_val = tpod(nza.nonzero_val;kwargs...)
#   return
# end

# function tpod(nza::NnzArray;kwargs...)
#   nzm_copy = copy(nza)
#   tpod!(nzm_copy;kwargs...)
#   nzm_copy
# end
