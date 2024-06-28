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
    abstract type SparseCore{T,N} <: AbstractTTCore{T,N} end

Tensor train cores for sparse matrices. In contrast with standard (3-D) tensor train
cores, a SparseCore is either a 4-D array (in scalar problems) or a 5-D array (in
multivalued problems). Information on the sparsity pattern of the matrices must
be provided for indexing purposes.

Subtypes:
- [`SparseCoreCSC`](@ref)

"""
abstract type SparseCore{T,N} <: AbstractTTCore{T,N} end

function _cores2basis(a::SparseCore{S},b::SparseCore{T}) where {S,T}
  @notimplemented "Need to provide a sparse index map for the construction of the global basis"
end

"""
    struct SparseCoreCSC{T,Ti} <: SparseCore{T,4} end

Tensor train cores for sparse matrices in CSC format

"""
struct SparseCoreCSC{T,Ti} <: SparseCore{T,4}
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

function Base.getindex(a::SparseCoreCSC,i::Vararg{Integer,4})
  if CartesianIndex(i[2:3]) ∈ a.sparse_indexes
    core_getindex(a,i)
  else
    zero(eltype(a))
  end
end

function core_getindex(a::SparseCoreCSC{T},i::Vararg{Integer,4}) where T
  i2 = findfirst(a.sparse_indexes .== [CartesianIndex(i[2:3])])
  i1,i3 = i[1],i[4]
  getindex(a.array,i1,i2,i3)
end

function cores2basis(core::SparseCoreCSC{T}) where T
  pcore = permutedims(core,(2,3,1,4))
  return reshape(pcore,size(pcore,1)*size(pcore,2),:)
end

# when we multiply two SparseCoreCSC objects, the result is a 3-D core that stacks
# the matrices' rows and columns
function _cores2basis(
  I::SparseIndexMap,
  a::SparseCoreCSC{S},
  b::SparseCoreCSC{T}
  ) where {S,T}

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

function _cores2basis(
  I::SparseIndexMap,
  a::SparseCoreCSC{S},
  b::SparseCoreCSC{T},
  c::SparseCoreCSC{U}
  ) where {S,T,U}

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

"""
    struct MultiValueSparseCoreCSC{T,Ti} <: SparseCore{T,5} end

Tensor train cores for multivalued sparse matrices in CSC format

"""
struct MultiValueSparseCoreCSC{T,Ti} <: SparseCore{T,5}
  array::Array{T,4}
  sparsity::MultiValueSparsityPatternCSC{T,Ti}
  sparse_indexes::Vector{CartesianIndex{2}}
end

function SparseCore(
  array::AbstractArray{T},
  sparsity::MultiValueSparsityPatternCSC{T}) where T

  irows,icols,_ = findnz(sparsity)
  SparseCoreCSC(array,sparsity,CartesianIndex.(irows,icols))
end

Base.size(a::MultiValueSparseCoreCSC) = (size(a.array,1),IndexMaps.num_rows(a.sparsity),
  IndexMaps.num_cols(a.sparsity),num_components(a.sparsity),size(a.array,3))

function Base.getindex(a::MultiValueSparseCoreCSC,i::Vararg{Integer,5})
  if CartesianIndex(i[2:3]) ∈ a.sparse_indexes
    core_getindex(a,i)
  else
    zero(eltype(a))
  end
end

function core_getindex(a::MultiValueSparseCoreCSC{T},i::Vararg{Integer,5}) where T
  i2 = findfirst(a.sparse_indexes .== [CartesianIndex(i[2:3])])
  i1,i3,i4 = i[1],i[5]
  getindex(a.array,i1,i2,i3,i4)
end

function cores2basis(core::MultiValueSparseCoreCSC{T}) where T
  pcore = permutedims(core,(2,3,4,1,5))
  return reshape(pcore,size(pcore,1)*size(pcore,2)*size(pcore,3),:)
end

# when we multiply two MultiValueSparseCoreCSC objects, the result is a 3-D core that stacks
# the matrices' rows, columns and components
function _cores2basis(
  I::SparseIndexMap,
  a::MultiValueSparseCoreCSC{S},
  b::MultiValueSparseCoreCSC{T}
  ) where {S,T}

  error("stop here")
  @check size(a,5) == size(b,1)
  @check size(a,4) == size(b,4)
  Is = get_index_map(I)
  TS = promote_type(T,S)
  nrows = size(a,2)*size(b,2)
  ncols = size(a,3)*size(b,3)
  ncomp = size(a,4)
  ab = zeros(TS,size(a,1),nrows*ncols*ncomp,size(b,4))
  for c = 1:ncomp
    for i = axes(a,1), j = axes(b,4)
      for α = axes(a,4)
        @inbounds @views ab[i,vec(Is),j] += kronecker(b.array[α,:,c,j],a.array[i,:,c,α])
      end
    end
  end
  return ab
end

# function _cores2basis(
#   I::SparseIndexMap,
#   a::MultiValueSparseCoreCSC{S},
#   b::MultiValueSparseCoreCSC{T},
#   c::MultiValueSparseCoreCSC{U}
#   ) where {S,T,U}

#   @check size(a,4) == size(b,1) && size(b,4) == size(c,1)
#   Is = get_index_map(I)
#   TSU = promote_type(T,S,U)
#   nrows = size(a,2)*size(b,2)*size(c,2)
#   ncols = size(a,3)*size(b,3)*size(c,3)
#   abc = zeros(TSU,size(a,1),nrows*ncols,size(c,4))
#   for i = axes(a,1), j = axes(c,4)
#     for α = axes(a,4), β = axes(b,4)
#       @inbounds @views abc[i,vec(Is),j] += kronecker(c.array[β,:,j],b.array[α,:,β],a.array[i,:,α])
#     end
#   end
#   return abc
# end
