function IndexMaps.recast(i::SparseIndexMap,a::AbstractVector{<:AbstractArray{T,3}}) where T
  us = IndexMaps.get_univariate_sparsity(i)
  @check length(us) ≤ length(a)
  if length(us) == length(a)
    return map(SparseCore,a,us)
  else
    asparse = map(i ->SparseCore(a[i],us[i]),eachindex(us))
    afull = a[length(us)+1:end]
    return [asparse...,afull...]
  end
end

"""
    abstract type AbstractTTCore{T,N} <: AbstractArray{T,N} end

Type for nonstandard representations of tensor train cores.

Subtypes:
- [`SparseCore`](@ref)

"""
abstract type AbstractTTCore{T,N} <: AbstractArray{T,N} end

"""
    abstract type SparseCore{T,N} <: AbstractTTCore{T,N} end

Tensor train cores for sparse matrices.

Subtypes:
- [`SparseCoreCSC`](@ref)

"""
abstract type SparseCore{T,N} <: AbstractTTCore{T,N} end

"""
    struct SparseCoreCSC{T,Ti} <: SparseCore{T,3} end

Tensor train cores for sparse matrices in CSC format

"""
struct SparseCoreCSC{T,Ti} <: SparseCore{T,3}
  array::Array{T,3}
  sparsity::SparsityPatternCSC{T,Ti}
end

function SparseCore(array::Array{T,3},sparsity::SparsityPatternCSC{T}) where T
  SparseCoreCSC(array,sparsity)
end

Base.size(a::SparseCoreCSC) = size(a.array)
Base.getindex(a::SparseCoreCSC,i::Vararg{Integer,3}) = getindex(a.array,i...)

num_space_dofs(a::SparseCoreCSC) = IndexMaps.num_rows(a.sparsity)*IndexMaps.num_cols(a.sparsity)

# block cores

struct BlockCore{T,D,A<:AbstractArray{T,D},BS} <: AbstractArray{T,D}
  array::Vector{A}
  axes::BS
  function BlockCore(array::Vector{A},axes::BS) where {T,D,A<:AbstractArray{T,D},BS<:NTuple}
    @assert all((size(a,2)==size(first(array),2) for a in array))
    new{T,D,A,BS}(array,axes)
  end
end

function BlockCore(array::Vector{<:AbstractArray},touched::AbstractArray{Bool}=I(length(array)))
  block_sizes = _sizes_from_blocks(array,touched)
  axes = map(blockedrange,block_sizes)
  BlockCore(array,axes)
end

function BlockCore(array::Vector{<:Vector{<:AbstractArray}})
  N = length(array)
  map(eachindex(first(array))) do n
    touched = n == 1 ? fill(true,N) : I(N)
    arrays_n = getindex.(array,n)
    BlockCore(arrays_n,touched)
  end
end

const BlockCore3D{T} = BlockCore{T,3,Array{T,3}}

Base.size(a::BlockCore) = map(length,axes(a))
Base.axes(a::BlockCore) = a.axes

function Base.getindex(a::BlockCore3D,i::Vararg{Integer,3})
  if size(a,1) == 1
    _getindex_vector(a,i...)
  else
    _getindex_matrix(a,i...)
  end
end

function _getindex_vector(a::BlockCore3D,i::Vararg{Integer,3})
  i1,i2,i3 = i
  @assert i1 == 1
  b3 = BlockArrays.findblockindex(axes(a,3),i3)
  a.array[b3.I...][i1,i2,b3.α...]
end

function _getindex_matrix(a::BlockCore3D,i::Vararg{Integer,3})
  i1,i2,i3 = i
  b1 = BlockArrays.findblockindex(axes(a,1),i1)
  b3 = BlockArrays.findblockindex(axes(a,3),i3)
  if b1.I == b3.I
    a.array[b1.I...][b1.α...,i2,b3.α...]
  else
    zero(eltype(a))
  end
end

function _sizes_from_blocks(a::Vector{<:AbstractArray},touched::AbstractVector{Bool})
  s1 = fill(1,length(a))
  s2 = fill(size(a[1],2),length(a))
  s3 = map(a -> size(a,3),a)
  for i in 1:length(a)-1
    s1[i] = 0
    s2[i] = 0
  end
  return (s1,s2,s3)
end

function _sizes_from_blocks(a::Vector{<:AbstractArray},touched::AbstractMatrix{Bool})
  s1 = map(a -> size(a,1),a)
  s2 = fill(size(a[1],2),length(a))
  s3 = map(a -> size(a,3),a)
  for i in 1:length(a)-1
    s2[i] = 0
  end
  return (s1,s2,s3)
end
