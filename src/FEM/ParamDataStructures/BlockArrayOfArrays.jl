struct BlockArrayOfArrays{T,N,L,A<:AbstractArray{<:ParamArray{T,N,L},N},B<:NTuple{N,AbstractUnitRange{Int}}} <: ParamArray{T,N,L}
  data::A
  axes::B
end

const BlockVectorOfVectors = BlockArrayOfArrays{T,1,L,A,B} where {T,L,A,B}
const BlockMatrixOfMatrices = BlockArrayOfArrays{T,2,L,A,B} where {T,L,A,B}

function BlockArrays._BlockArray(data::AbstractArray{<:ParamArray},axes<:NTuple{N,AbstractUnitRange{Int}})
  BlockArrayOfArrays(data,axes)
end

@inline Base.size(A::BlockArrayOfArrays) = map(length,axes(A))
Base.axes(A::BlockArrayOfArrays) = A.axes

@inline function ArraysOfArrays.innersize(A::BlockArrayOfArrays) where {T,N}
  map(innersize,A.data)
end

BlockArrays.blocks(A::BlockArrayOfArrays) = A.data

all_data(A::BlockArrayOfArrays) = A.data
param_getindex(A::BlockArrayOfArrays,i::Integer) = mortar(map(a->param_getindex(a,i),A.data))
param_view(A::BlockArrayOfArrays{T,N},i::Integer) where {T,N} = BlockParamView(A,i)
param_entry(A::BlockArrayOfArrays{T,N},i::Vararg{Integer,N}) where {T,N} = mortar(map(a->param_entry(a,i),A.data))

Base.@propagate_inbounds function param_setindex!(
  A::BlockArrayOfArrays{T,N},
  v::BlockArray{T,N},
  i::Integer
  ) where {T,N}

  @check blocksize(A) == blocksize(v)
  for j in eachblock(A)
    map(param_setindex!(A.data[j],blocks(v)[j],i))
  end
  A
end

Base.@propagate_inbounds function Base.getindex(A::BlockArrayOfArrays{T,N},i::Vararg{Integer,N}) where {T,N}
  @boundscheck checkbounds(A,i...)
  v = A[findblockindex.(axes(A),i)...]
  return v
end

@propagate_inbounds function Base.getindex(A::BlockArrayOfArrays{T,N},i::BlockIndex{N}) where {T,N}
  BlockArrays._blockindex_getindex(A,i)
end

@propagate_inbounds function Base.getindex(A::BlockArrayOfArrays{T},i::BlockIndex{1}) where T
  BlockArrays._blockindex_getindex(A,i)
end

Base.@propagate_inbounds function Base.setindex!(A::BlockArrayOfArrays{T,N},v,i::Vararg{Integer,N}) where {T,N}
  @boundscheck checkbounds(A,i...)
  @inbounds A[findblockindex.(axes(A),i)...] = v
  return A
end

Base.@propagate_inbounds function Base.setindex!(A::BlockArrayOfArrays{T,N},v,block::Vararg{Block{1},N}) where {T,N}
  blks = Int.(block)
  @inbounds A.data[blks...] = v
  return A
end

struct BlockParamView{T,N,L,A} <: ParamArray{T,N,L}
  data::BlockArrayOfArrays{T,N,L,A}
  i::Int
end

# Base.size(A::BlockParamView) =
# Base.getindex(A::BlockParamView) =
