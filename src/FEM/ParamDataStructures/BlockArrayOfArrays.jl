struct BlockArrayOfArrays{T,N,L,A<:AbstractArray{<:AbstractParamArray{T,N,L},N},B<:NTuple{N,AbstractUnitRange{Int}}} <: ParamArray{T,N,L}
  data::A
  axes::B
end

const BlockVectorOfVectors{T,L,A} = BlockArrayOfArrays{T,1,L,A}
const BlockMatrixOfMatrices{T,L,A} = BlockArrayOfArrays{T,2,L,A}

function BlockArrays._BlockArray(data::AbstractArray{<:AbstractParamArray,N},axes::NTuple{N,AbstractUnitRange{Int}}) where N
  BlockArrayOfArrays(data,axes)
end

@inline Base.size(A::BlockArrayOfArrays) = map(length,axes(A))
Base.axes(A::BlockArrayOfArrays) = A.axes

@inline function ArraysOfArrays.innersize(A::BlockArrayOfArrays)
  map(innersize,A.data)
end

@inline function inneraxes(A::BlockArrayOfArrays{T,N}) where {T,N}
  is = innersize(A)
  _inneraxes(is)
end

_inneraxes(i) = @abstractmethod

@inline function _inneraxes(i::AbstractVector{Tuple{Int}})
  (blockedrange(getindex.(i,1)),)
end

@inline function _inneraxes(i::AbstractMatrix{Tuple{Int,Int}})
  diagi = diag(i)
  ntuple(j->blockedrange(getindex.(diagi,j)),Val{length(diagi)}())
end

BlockArrays.blocks(A::BlockArrayOfArrays) = A.data

param_getindex(A::BlockArrayOfArrays,i::Integer) = BlockParamView(A,i)#mortar(map(a->param_getindex(a,i),A.data))

function param_entry(A::BlockArrayOfArrays{T,N},i::Vararg{Integer,N}) where {T,N}
  param_entry(A,findblockindex.(inneraxes(A),i)...)
end

function param_entry(A::BlockArrayOfArrays{T,N},i::BlockIndex{N}) where {T,N}
  @boundscheck blockcheckbounds(A,Block(i.I))
  @inbounds bl = A.data[i.I...]
  @inbounds ParamNumber(bl.data[i.α...,:])
end

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

Base.@propagate_inbounds function Base.getindex(A::BlockArrayOfArrays{T,N},i::BlockIndex{N}) where {T,N}
  BlockArrays._blockindex_getindex(A,i)
end

Base.@propagate_inbounds function Base.getindex(A::BlockVectorOfVectors{T},i::BlockIndex{1}) where {T}
  BlockArrays._blockindex_getindex(A,i)
end

@inline function BlockArrays._blockindex_getindex(A::BlockArrayOfArrays{T,N},i::BlockIndex{N}) where {T,N}
  @boundscheck blockcheckbounds(A,Block(i.I))
  @inbounds bl = A.data[i.I...]
  @boundscheck checkbounds(bl,i.α...)
  @inbounds v = bl[i.α...]
  return v
end

Base.@propagate_inbounds function Base.setindex!(A::BlockArrayOfArrays{T,N},v,i::Vararg{Integer,N}) where {T,N}
  @boundscheck checkbounds(A,i...)
  @inbounds A[findblockindex.(axes(A),i)...] = v
  return A
end

Base.@propagate_inbounds function Base.setindex!(A::BlockArrayOfArrays{T,N},v,i::Vararg{BlockIndex{1},N}) where {T,N}
  _blockindex_setindex!(A,v,BlockIndex(i))
  A
end

Base.@propagate_inbounds function Base.setindex!(A::BlockArrayOfArrays{T,N},v,i::BlockIndex{N}) where {T,N}
  _blockindex_setindex!(A,v,i)
  A
end

Base.@propagate_inbounds function Base.setindex!(A::BlockVectorOfVectors{T},v,i::BlockIndex{1}) where {T}
  _blockindex_setindex!(A,v,i)
  A
end

@inline function _blockindex_setindex!(A::BlockArrayOfArrays{T,N},v,i::BlockIndex{N}) where {T,N}
  @boundscheck blockcheckbounds(A,Block(i.I))
  @inbounds bl = A.data[i.I...]
  @boundscheck checkbounds(bl,i.α...)
  @inbounds bl[i.α...] = v
  return A
end

function Base.similar(A::BlockArrayOfArrays{T,N},::Type{<:AbstractArray{T′,N}}) where {T,T′,N}
  BlockArrayOfArrays(map(a->similar(a,Array{T′,N}),A.data),A.axes)
end

function Base.similar(A::BlockArrayOfArrays{T,N},::Type{<:AbstractArray{T′,N}},axes::BlockedUnitRange) where {T,T′,N}
  sz = map(length,blocks(axes))
  BlockArrayOfArrays(map((a,s)->similar(a,Array{T′,N},s),A.data,sz),A.axes)
end

function Base.copyto!(A::BlockArrayOfArrays,B::BlockArrayOfArrays)
  @check size(A) == size(B) && innersize(A) == innersize(B)
  map(copyto!,A.data,B.data)
  A
end

function Base.zero(A::BlockArrayOfArrays)
  BlockArrayOfArrays(map(zero,A.data),A.axes)
end

for f in (:(Base.fill!),:(LinearAlgebra.fillstored!))
  @eval begin
    function $f(A::BlockArrayOfArrays,z::Number)
      map(a -> $f(a,z),param_data(A))
      return A
    end
  end
end

for f in (:(Base.maximum),:(Base.minimum))
  @eval begin
    $f(g,A::BlockArrayOfArrays) = $f(map(a -> $f(g,a),A.data))
  end
end

struct BlockParamView{T,N,L,A} <: AbstractBlockArray{T,N}
  data::BlockArrayOfArrays{T,N,L,A}
  index::Int
end

Base.axes(A::BlockParamView) = inneraxes(A.data)

Base.@propagate_inbounds function Base.getindex(A::BlockParamView{T,N},i::Vararg{Integer,N}) where {T,N}
  @boundscheck checkbounds(A,i...)
  v = A[findblockindex.(axes(A),i)...]
  return v
end

Base.@propagate_inbounds function Base.getindex(A::BlockParamView{T,N},i::BlockIndex{N}) where {T,N}
  BlockArrays._blockindex_getindex(A,i)
end

Base.@propagate_inbounds function Base.getindex(A::BlockParamView{T},i::BlockIndex{1}) where {T}
  BlockArrays._blockindex_getindex(A,i)
end

@inline function BlockArrays._blockindex_getindex(A::BlockParamView{T,N},i::BlockIndex{N}) where {T,N}
  @boundscheck blockcheckbounds(A,Block(i.I))
  @inbounds bl = A.data.data[i.I...]
  @boundscheck checkbounds(bl.data,i.α...,A.index)
  @inbounds v = bl.data[i.α...,A.index]
  return v
end

Base.@propagate_inbounds function Base.setindex!(A::BlockParamView{T,N},v,i::Vararg{Integer,N}) where {T,N}
  @boundscheck checkbounds(A,i...)
  @inbounds A[findblockindex.(axes(A),i)...] = v
  return A
end

Base.@propagate_inbounds function Base.setindex!(A::BlockParamView{T,N},v,i::BlockIndex{N}) where {T,N}
  _blockindex_setindex!(A,v,i)
  A
end

Base.@propagate_inbounds function Base.setindex!(A::BlockParamView{T},v,i::BlockIndex{1}) where {T}
  _blockindex_setindex!(A,v,i)
  A
end

@inline function _blockindex_setindex!(A::BlockParamView{T,N},v,i::BlockIndex{N}) where {T,N}
  @boundscheck blockcheckbounds(A,Block(i.I))
  @inbounds bl = A.data.data[i.I...]
  @boundscheck checkbounds(bl.data,i.α...,A.index)
  @inbounds bl.data[i.α...,A.index] = v
  return A
end
