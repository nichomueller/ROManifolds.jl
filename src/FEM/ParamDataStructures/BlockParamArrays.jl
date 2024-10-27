ParamArray(A::AbstractArray{<:AbstractParamArray};kwargs...) = mortar(A)

function param_array(a::BlockArray,l::Integer;kwargs...)
  mortar(b -> param_array(b,l;kwargs...),blocks(a))
end

"""
    struct BlockParamArray{T,N,A<:AbstractArray{<:AbstractParamArray{T,N},N},
      B<:NTuple{N,AbstractUnitRange{Int}}} <: ParamArray{T,N} end

Is to a [`ParamArray`](@ref) as a BlockArray is to a regular AbstractArray.
Instances of BlockParamArray are obtained by extending the function `_BlockArray`
in the package BlockArrays.

"""
struct BlockParamArray{T,N,A<:AbstractArray{<:AbstractParamArray{T,N},N},B<:NTuple{N,AbstractUnitRange{Int}}} <: ParamArray{T,N}
  data::A
  axes::B
end

const BlockParamVector{T} = BlockParamArray{T,1,<:AbstractVector{<:AbstractParamVector{T}}}
const BlockParamMatrix{T} = BlockParamArray{T,2,<:AbstractMatrix{<:AbstractParamMatrix{T}}}

const BlockConsecutiveParamVector{T} = BlockParamArray{T,1,<:AbstractVector{<:ConsecutiveParamVector{T}}}
const BlockConsecutiveParamMatrix{T} = BlockParamArray{T,2,<:AbstractMatrix{<:ConsecutiveParamMatrix{T}}}

function BlockArrays._BlockArray(data::AbstractArray{<:AbstractParamArray,N},axes::NTuple{N,AbstractUnitRange{Int}}) where N
  @assert all(param_length(d)==param_length(first(data)) for d in data)
  BlockParamArray(data,axes)
end

nblocks(A) = @abstractmethod
nblocks(A::Union{BlockedUnitRange,BlockArray,BlockParamArray}) = length(blocks(A))

MemoryLayoutStyle(::Type{<:BlockParamArray{T,N,A}}) where {T,N,A} = MemoryLayoutStyle(eltype(A))

param_length(A::BlockParamArray) = param_length(first(blocks(A)))

@inline Base.size(A::BlockParamArray) = map(length,axes(A))
Base.axes(A::BlockParamArray) = A.axes

@inline function ArraysOfArrays.innersize(A::BlockParamArray)
  map(innersize,A.data)
end

@inline function inneraxes(A::BlockParamArray{T,N}) where {T,N}
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

BlockArrays.blocks(A::BlockParamArray) = A.data

function param_getindex(A::BlockParamArray,i::Integer)
  mortar(map(a->param_getindex(a,i),blocks(A)))
end

Base.@propagate_inbounds function Base.getindex(A::BlockParamArray{T,N},i::Vararg{Integer,N}) where {T,N}
  @boundscheck checkbounds(A,i...)
  v = A[findblockindex.(axes(A),i)...]
  return v
end

Base.@propagate_inbounds function Base.getindex(A::BlockParamArray{T,N},i::BlockIndex{N}) where {T,N}
  BlockArrays._blockindex_getindex(A,i)
end

Base.@propagate_inbounds function Base.getindex(A::BlockParamVector{T},i::BlockIndex{1}) where {T}
  BlockArrays._blockindex_getindex(A,i)
end

@inline function BlockArrays._blockindex_getindex(A::BlockParamArray{T,N},i::BlockIndex{N}) where {T,N}
  @boundscheck blockcheckbounds(A,Block(i.I))
  @inbounds bl = A.data[i.I...]
  @boundscheck checkbounds(bl,i.α...)
  @inbounds v = bl[i.α...]
  return v
end

Base.@propagate_inbounds function Base.setindex!(A::BlockParamArray{T,N},v,i::Vararg{Integer,N}) where {T,N}
  @boundscheck checkbounds(A,i...)
  @inbounds A[findblockindex.(axes(A),i)...] = v
  return A
end

Base.@propagate_inbounds function Base.setindex!(A::BlockParamArray{T,N},v,i::Vararg{BlockIndex{1},N}) where {T,N}
  _blockindex_setindex!(A,v,BlockIndex(i))
  A
end

Base.@propagate_inbounds function Base.setindex!(A::BlockParamArray{T,N},v,i::BlockIndex{N}) where {T,N}
  _blockindex_setindex!(A,v,i)
  A
end

Base.@propagate_inbounds function Base.setindex!(A::BlockParamVector{T},v,i::BlockIndex{1}) where {T}
  _blockindex_setindex!(A,v,i)
  A
end

@inline function _blockindex_setindex!(A::BlockParamArray{T,N},v,i::BlockIndex{N}) where {T,N}
  @boundscheck blockcheckbounds(A,Block(i.I))
  @inbounds bl = A.data[i.I...]
  @boundscheck checkbounds(bl,i.α...)
  @inbounds bl[i.α...] = v
  return A
end

Base.@propagate_inbounds function Base.getindex(A::BlockParamVector{T},i::Block{1}) where T
  @boundscheck blockcheckbounds(A,i)
  getindex(A.data,i.n...)
end

Base.@propagate_inbounds function Base.getindex(A::BlockParamArray{T,N},i::Block{N}) where {T,N}
  @boundscheck blockcheckbounds(A,i)
  getindex(A.data,i.n...)
end

Base.@propagate_inbounds function Base.setindex!(A::BlockParamVector{T},v,i::Block{1}) where T
  @boundscheck blockcheckbounds(A,i)
  setindex!(A.data,v,i.n...)
end

Base.@propagate_inbounds function Base.setindex!(A::BlockParamArray{T,N},v,i::Block{N}) where {T,N}
  @boundscheck blockcheckbounds(A,i)
  setindex!(A.data,v,i.n...)
end

function Base.similar(A::BlockParamArray{T,N},::Type{<:AbstractArray{T′,N}}) where {T,T′,N}
  BlockParamArray(map(a->similar(a,Array{T′,N}),blocks(A)),A.axes)
end

function Base.similar(A::BlockParamArray{T,N},::Type{S},axes::Vararg{BlockedUnitRange}) where {T,T′,N,S<:AbstractArray{T′,N}}
  A′ = map(eachindex(blocks(A))) do i
    ai = blocks(A)[i]
    axi = map(ax -> blocks(ax)[i],axes)
    si = length.(axi)
    similar(ai,S,si)
  end
  BlockParamArray(A′,A.axes)
end

function Base.copy(A::BlockParamArray)
  A′ = map(copy,blocks(A))
  BlockParamArray(A′,A.axes)
end

function Base.copyto!(A::BlockParamArray,B::BlockParamArray)
  @check size(A) == size(B) && innersize(A) == innersize(B)
  map(copyto!,A.data,B.data)
  A
end

for f in (:(Base.fill!),:(LinearAlgebra.fillstored!))
  @eval begin
    function $f(A::BlockParamArray,b::Number)
      map(a -> $f(a,b),blocks(A))
      return A
    end
  end
end

function LinearAlgebra.rmul!(A::BlockParamArray,b::Number)
  map(a -> rmul!(a,b),blocks(A))
  return A
end

function LinearAlgebra.axpy!(α::Number,A::BlockParamArray,B::BlockParamArray)
  for (a,b) in zip(blocks(A),blocks(B))
    (iszero(α) || iszero(a)) && continue
    axpy!(α,a,b)
  end
  return B
end

function LinearAlgebra.norm(A::BlockParamArray)
  n = zeros(param_length(A))
  for b in blocks(A)
    n .+= norm(b).^2
  end
  return sqrt.(n)
end

function get_param_entry(A::BlockParamArray{T},i...) where T
  entries = map(a -> get_param_entry(a,i...),blocks(A))
  mortar(entries)
end
