ParamArray(A::AbstractArray{<:ParamArray}) = mortar(A)

function param_array(a::BlockArray,l::Integer) where N
  mortar(b -> param_array(b,l),blocks(a))
end

"""
    struct BlockParamArray{T,N,L,A<:AbstractArray{<:AbstractParamArray{T,N,L},N},
      B<:NTuple{N,AbstractUnitRange{Int}}} <: ParamArray{T,N,L} end

Is to a [`ParamArray`](@ref) as a BlockArray is to a regular AbstractArray.
Instances of BlockParamArray are obtained by extending the function `_BlockArray`
in the package BlockArrays.

"""
struct BlockParamArray{T,N,L,A<:AbstractArray{<:AbstractParamArray{T,N,L},N},B<:NTuple{N,AbstractUnitRange{Int}}} <: ParamArray{T,N,L}
  data::A
  axes::B
end

const BlockParamVector{T,L} = BlockParamArray{T,1,L,<:AbstractVector{<:AbstractParamVector{T,L}}}
const BlockParamMatrix{T,L} = BlockParamArray{T,2,L,<:AbstractMatrix{<:AbstractParamMatrix{T,L}}}

const BlockConsecutiveParamVector{T,L} = BlockParamArray{T,1,L,<:AbstractVector{<:ConsecutiveParamVector{T,L}}}
const BlockConsecutiveParamMatrix{T,L} = BlockParamArray{T,2,L,<:AbstractMatrix{<:ConsecutiveParamMatrix{T,L}}}

function BlockArrays._BlockArray(data::AbstractArray{<:AbstractParamArray,N},axes::NTuple{N,AbstractUnitRange{Int}}) where N
  BlockParamArray(data,axes)
end

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
  @warn "Should avoid using this function for performance reasons"
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

function Base.similar(A::BlockParamArray{T,N},::Type{<:AbstractArray{T′,N}},axes::BlockedUnitRange) where {T,T′,N}
  BlockParamArray(map((a,ax)->similar(a,Array{T′,N},ax),blocks(A),blocks(axes)),A.axes)
end

function Base.copyto!(A::BlockParamArray,B::BlockParamArray)
  @check size(A) == size(B) && innersize(A) == innersize(B)
  map(copyto!,A.data,B.data)
  A
end

for f in (:(Base.fill!),:(LinearAlgebra.fillstored!))
  @eval begin
    function $f(A::BlockParamArray,z::Number)
      map(a -> $f(a,z),blocks(A))
      return A
    end
  end
end

function LinearAlgebra.norm(A::BlockParamArray)
  n = 0.0
  for b in blocks(A)
    n += norm(b)^2
  end
  return sqrt(n)
end

function param_entry(A::BlockParamArray{T,N},i::Vararg{Integer}) where {T,N}
  n = length(blocks(A))
  L = param_length(A)
  entries = Vector{Vector{T}}(undef,n)
  @inbounds for (k,Ak) in enumerate(blocks(A))
    entries[k] = param_entry(Ak,i...)
  end
  mortar(entries)
end
