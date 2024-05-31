struct TTBlockArray{T,N,A,B,C,I} <: AbstractTTArray{T,N}
  blockarrays::A
  blockmaps::B
  axes::C
  function TTBlockArray(
    blockarrays::A,
    blockmaps::B,
    axes::C
    ) where {T,N,I<:AbstractIndexMap,A<:AbstractArray{<:TTBlockArray{T,N}},B<:AbstractArray{I},C}

    new{T,N,A,B,C,I}(blockarrays,blockmaps,axes)
  end
end

const TTBlockVector = TTBlockArray{T,1,A,B,C,I} where {T,A,B,C,I}
const TTBlockMatrix = TTBlockArray{T,2,A,B,C,I} where {T,A,B,C,I}

get_values(a::TTBlockArray) = a.blockarrays
get_index_map(a::TTBlockArray) = a.index_map
get_dim(a::TTBlockArray) = get_dim(first(get_values(a)))

const ParamBlockTTArray = ParamBlockArray{T,N,L,A,B} where {T,N,L,A<:AbstractVector{<:ParamTTArray},B}
const ParamBlockTTVector = ParamBlockArray{T,1,L,A,B} where {T,L,A<:AbstractVector{<:TTVector},B}
const ParamBlockTTMatrix = ParamBlockArray{T,2,L,A,B} where {T,L,A<:AbstractVector{<:TTMatrix},B}
const ParamBlockTTSparseMatrix = ParamBlockArray{T,2,L,A,B} where {T,L,A<:AbstractVector{<:TTSparseMatrix},B}
const ParamBlockTTSparseMatrixCSC = ParamBlockArray{T,2,L,A,B} where {T,L,A<:AbstractVector{<:TTSparseMatrixCSC},B}

get_values(a::ParamBlockTTArray) = mortar(get_values.(get_array(a)))
get_index_map(a::ParamBlockTTArray) = get_index_map(first(a))

function BlockArrays.mortar(blocks::AbstractArray{<:TTBlockArray})
  block_index_map = map(get_index_map,blocks)
  block_sizes = BlockArrays.sizes_from_blocks(first.(blocks))
  block_axes = map(blockedrange,block_sizes)
  TTBlockArray(blocks,block_index_map,block_axes)
end

Base.axes(a::TTBlockArray) = a.blockaxes
Base.size(a::TTBlockArray) = map(length,axes(a))
BlockArrays.blocks(a::TTBlockArray) = get_values(a)

@inline function Base.getindex(a::TTBlockArray{T,N},i::Vararg{Integer,N}) where {T,N}
  @boundscheck checkbounds(a,i...)
  @inbounds v = a[findblockindex.(axes(a),i)...]
  return v
end

@inline function Base.setindex!(a::TTBlockArray{T,N},v,i::Vararg{Integer,N}) where {T,N}
  @boundscheck checkbounds(a,i...)
  @inbounds a[findblockindex.(axes(a),i)...] = v
  return a
end

function Base.copy(a::TTBlockArray)
  TTBlockArray(map(copy,get_values(a)),a.blockmaps,a.axes)
end

function Base.similar(a::TTBlockArray)
  TTBlockArray(map(similar,get_values(a)),a.blockmaps,a.axes)
end

function Base.fill!(a::TTBlockArray,v)
  for block in get_values(a)
    fill!(block,v)
  end
  a
end

function LinearAlgebra.fillstored!(a::TTBlockArray,v)
  for block in get_values(a)
    fillstored!(block,v)
  end
  a
end

for op in (:+,:-)
  @eval begin
    function ($op)(a::TTBlockArray,b::TTBlockArray)
      @check get_index_map(a) == get_index_map(b)
      TTBlockArray(($op)(get_values(a),get_values(b)),get_index_map(a),axes(a))
    end
  end
end

function Base.:*(a::TTBlockArray,b::Number)
  TTBlockArray(get_values(a)*b,get_index_map(a),axes(a))
end

function Base.:*(a::TTBlockMatrix,b::TTBlockVector)
  @check get_index_map(a) == get_index_map(b)
  TTBlockArray(get_values(a)*get_values(b),get_index_map(a),axes(a))
end

function Base.:\(a::TTBlockMatrix,b::TTBlockVector)
  @check get_index_map(a) == get_index_map(b)
  TTBlockArray(get_values(a)\get_values(b),get_index_map(a),axes(a))
end

function Base.transpose(a::TTBlockArray)
  TTBlockArray(transpose(get_values(a)),get_index_map(a),axes(a))
end

struct BlockTTBroadcast{V,M}
  values::V
  index_map::M
end

get_values(a::BlockTTBroadcast) = a.values
get_index_map(a::BlockTTBroadcast) = a.index_map

function Base.broadcasted(f,a::Union{TTBlockArray,BlockTTBroadcast}...)
  index_map = get_index_map(a[1])
  @check all(get_index_map.(a) == index_map)
  BlockTTBroadcast(Base.broadcasted(f,map(get_values,a)...),index_map)
end

function Base.broadcasted(f,a::Union{TTBlockArray,BlockTTBroadcast},b::Number)
  BlockTTBroadcast(Base.broadcasted(f,get_values(a),b),get_index_map(a))
end

function Base.broadcasted(f,a::Number,b::Union{TTBlockArray,BlockTTBroadcast})
  BlockTTBroadcast(Base.broadcasted(f,a,get_values(b)),get_index_map(a))
end

function Base.broadcasted(f,
  a::Union{TTBlockArray,BlockTTBroadcast},
  b::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}})
  Base.broadcasted(f,a,Base.materialize(b))
end

function Base.broadcasted(
  f,
  a::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}},
  b::Union{TTBlockArray,BlockTTBroadcast})
  Base.broadcasted(f,Base.materialize(a),b)
end

function Base.materialize(b::BlockTTBroadcast)
  a = Base.materialize(get_values(b))
  mortar(a,get_index_map(a))
end

function Base.materialize!(a::TTBlockArray,b::Broadcast.Broadcasted)
  Base.materialize!(get_values(a),b)
  a
end

function Base.materialize!(a::TTBlockArray,b::BlockTTBroadcast)
  Base.materialize!(get_values(a),get_values(b))
  a
end
