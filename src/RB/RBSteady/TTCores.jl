"""
    abstract type AbstractTTCore{T,N} <: AbstractArray{T,N} end

Type for nonstandard representations of tensor train cores.

Subtypes:
- [`SparseCore`](@ref)

"""
abstract type AbstractTTCore{T,N} <: AbstractArray{T,N} end

Base.IndexStyle(::Type{<:AbstractTTCore}) = IndexCartesian()
Base.getindex(a::AbstractTTCore,i::Integer...) = getindex(a,CartesianIndex(i))

"""
    abstract type SparseCore{T} <: AbstractTTCore{T,4} end

Tensor train cores for sparse matrices. In contrast with standard (3-D) tensor train
cores, a SparseCore is a 4-D abstract array. Information on the sparsity pattern
of the matrices must be provided for indexing purposes.

Subtypes:
- [`SparseCoreCSC`](@ref)

"""
abstract type SparseCore{T} <: AbstractTTCore{T,4} end

"""
    struct SparseCoreCSC{T,Ti} <: SparseCore{T} end

Tensor train cores for sparse matrices in CSC format

"""
struct SparseCoreCSC{T,Ti} <: SparseCore{T}
  array::Array{T,3}
  sparsity::SparsityPatternCSC{T,Ti}
  sparse_indexes::Vector{CartesianIndex{2}}
end

function SparseCore(
  array::AbstractArray{T},
  sparsity::SparsityPatternCSC{T}) where T

  irows,icols,_ = findnz(sparsity)
  SparseCoreCSC(array,sparsity,CartesianIndex.(irows,icols))
end

Base.size(a::SparseCoreCSC) = (size(a.array,1),IndexMaps.num_rows(a.sparsity),
  IndexMaps.num_cols(a.sparsity),size(a.array,3))

function Base.getindex(a::SparseCoreCSC,i::CartesianIndex{4})
  is_nnz = CartesianIndex(i.I[2:3]) ∈ a.sparse_indexes
  if is_nnz
    sparse_getindex(a,i)
  else
    zero(eltype(a))
  end
end

function sparse_getindex(a::SparseCoreCSC{T},i::CartesianIndex{4}) where T
  i2 = findfirst(a.sparse_indexes .== [CartesianIndex(i.I[2:3])])
  i1,i3 = i.I[1],i.I[4]
  getindex(a.array,CartesianIndex((i1,i2,i3)))
end

function _cores2basis(a::SparseCoreCSC{S},b::SparseCoreCSC{T}) where {S,T}
  @notimplemented "Need to provide a sparse index map for the construction of the global basis"
end

function _cores2basis(I::SparseIndexMap,a::SparseCoreCSC{S},b::SparseCoreCSC{T}) where {S,T}
  @check size(a,4) == size(b,1)
  Is = get_index_map(I)
  TS = promote_type(T,S)
  nrows = size(a,2)*size(b,2)
  ncols = size(a,3)*size(b,3)
  ab = zeros(TS,size(a,1),nrows*ncols,size(b,4))
  for i = axes(a,1), j = axes(b,4)
    for α = axes(a,4)
      @inbounds @views ab[i,vec(Is),j] += kronecker(b.array[α,:,j],a.array[i,:,α])
    end
  end
  return ab
end

function _cores2basis(I::SparseIndexMap,a::SparseCoreCSC{S},b::SparseCoreCSC{T},c::SparseCoreCSC{U}) where {S,T,U}
  @check size(a,4) == size(b,1) && size(b,4) == size(c,1)
  Is = get_index_map(I)
  TSU = promote_type(T,S,U)
  nrows = size(a,2)*size(b,2)*size(c,2)
  ncols = size(a,3)*size(b,3)*size(c,3)
  abc = zeros(TSU,size(a,1),nrows*ncols,size(c,4))
  for i = axes(a,1), j = axes(c,4)
    for α = axes(a,4), β = axes(b,4)
      @inbounds @views abc[i,vec(Is),j] += kronecker(c.array[β,:,j],b.array[α,:,β],a.array[i,:,α])
    end
  end
  return abc
end
