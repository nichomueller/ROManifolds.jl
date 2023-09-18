get_nblocks(a::BlockArray) = length(a.blocks)

function Base.hcat(a::BlockVector,b::BlockVector)
  c = map(hcat,a.blocks,b.blocks)
  d = reshape(c,:,length(c))
  mortar(d)
end

function compress_array(entire_array::BlockMatrix)
  nblocks = get_nblocks(entire_array)
  nzm = map(compress_array,entire_array)

  nonzero_idx_blocks = first.(nzm)
  nonzero_idx = mortar(nonzero_idx_blocks)
  nonzero_val_blocks = last.(nzm)
  nonzero_val = mortar(reshape(nonzero_val_blocks,:,nblocks))

  nonzero_idx,nonzero_val
end

function tpod(bmat::BlockMatrix;kwags...)
  svd_blocks = map(tpod,bmat.blocks)
  mortar(reshape(svd_blocks,:,1))
end

struct BlockNnzArray{T,N,OT} <: AbstractArray{T,N}
  nonzero_val::BlockArray{T,N}
  nonzero_idx::BlockVector{T}
  nrows::Tuple{Vararg{Int}}

  function BlockNnzArray{OT}(
    nonzero_val::BlockArray{T,N},
    nonzero_idx::BlockVector{T},
    nrows::Vector{Int}) where {T,N,OT}

    new{T,N,OT}(nonzero_val,nonzero_idx,nrows)
  end
end

get_nonzero_val(nzb::BlockNnzArray) = nzb.nonzero_val.blocks

get_nblocks(nzb::BlockNnzArray) = length(nzb.nonzero_val.blocks)

function get_block(nzb::BlockNnzArray{T,N,OT},idx) where {T,N,OT}
  nonzero_val = nzb.nonzero_val.blocks[idx]
  nonzero_idx = nzb.nonzero_idx.blocks[idx]
  nrows = nzb.nrows[idx]
  NnzMatrix{OT}(nonzero_val,nonzero_idx,nrows)
end

Base.getindex(nzb::BlockNnzArray,idx) = get_block(nzb,idx)

function compress(entire_array::BlockArray{N,T,OT} where {N,T}) where {OT<:AbstractArray}
  nonzero_idx,nonzero_val = compress_array(entire_array)
  nrows = map(a->size(a,1),entire_array)
  BlockNnzArray{OT}(nonzero_val,nonzero_idx,nrows)
end

function Base.show(io::IO,nzb::BlockNnzArray)
  for nzm in nzb
    show(io,nzm)
  end
end

Base.size(nzb::BlockNnzArray,idx...) = map(nzm->size(nzm.nonzero_val,idx...),nzb)

function recast(nzb::BlockNnzArray)
  map(recast,nzb)
end

function tpod(nzb::BlockNnzArray{T,N,OT};kwargs...) where {T,N,OT}
  nonzero_val = tpod(nzm.nonzero_val;kwargs...)
  return BlockNnzArray{OT}(nonzero_val,nzm.nonzero_idx,nzm.nrows)
end
