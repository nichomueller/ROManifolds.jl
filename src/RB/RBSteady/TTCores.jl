function IndexMaps.recast(i::SparseIndexMap,a::AbstractVector{<:AbstractArray{T,3}}) where T
  us = IndexMaps.get_univariate_sparsity(i)
  @check length(us) == length(a)
  asparse = map(SparseCore,a,us)
  return asparse
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

Tensor train cores for sparse matrices. In contrast with standard (3-D) tensor train
cores, a SparseCore is a 4-D array. Information on the sparsity pattern of the matrices must
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
    core_getindex(a,i...)
  else
    zero(eltype(a))
  end
end

function core_getindex(a::SparseCoreCSC{T},i::Vararg{Integer,4}) where T
  i2 = findfirst(a.sparse_indexes .== [CartesianIndex(i[2:3])])
  i1,i3 = i[1],i[4]
  getindex(a.array,i1,i2,i3)
end

struct BlockArrayTTCores{T,D,N,A<:AbstractArray{T,D}} <: AbstractArray{AbstractArray{T,D},N}
  array::Vector{A}
  touched::Array{Bool,N}
end

const BlockVectorTTCores{T,D,A<:AbstractArray{T,D}} = BlockArrayTTCores{T,D,1,A}
const BlockMatrixTTCores{T,D,A<:AbstractArray{T,D}} = BlockArrayTTCores{T,D,2,A}

Base.size(a::BlockArrayTTCores) = size(a.touched)

size_of_block(a::BlockVectorTTCores,irow) = size(a.array[irow])
function size_of_block(a::BlockMatrixTTCores{T,3} where T,irow,icol)
  brow = a.array[irow]
  bcol = a.array[icol]
  @check size(brow,2) == size(bcol,2)
  (size(brow,1),size(brow,2),size(bcol,3))
end

function Base.getindex(a::BlockArrayTTCores{T,D,N},i::Vararg{Integer,N}) where {T,D,N}
  iblock = first(i)
  if all(i.==iblock)
    a.array[iblock]
  else
    fill(zero(T),size_of_block(a,i...))
  end
end

function get_touched_blocks(a::BlockArrayTTCores)
  findall(a.touched)
end

# core operations

function Base.:*(a::AbstractMatrix{T},b::AbstractArray{S,3}) where {T,S}
  @check size(a,2) == size(b,2)
  TS = promote_type(T,S)
  ab = zeros(TS,size(b,1),size(a,1),size(b,3))
  @inbounds for i = axes(b,1), j = axes(b,3)
    ab[i,:,j] = a*b[i,:,j]
  end
  return ab
end

function Base.:*(a::AbstractArray{T,3},b::AbstractMatrix{S}) where {T,S}
  @check size(a,2) == size(b,1)
  TS = promote_type(T,S)
  ab = zeros(TS,size(a,1),size(b,2),size(a,3))
  @inbounds for i = axes(a,1), j = axes(a,3)
    ab[i,:,j] = a[i,:,j]*b
  end
  return ab
end

"""
    cores2basis(index_map::AbstractIndexMap,cores::AbstractArray...) -> AbstractMatrix
    cores2basis(index_map::AbstractIndexMap,cores::ArrayBlock...) -> ArrayBlock

Computes the kronecker product of the suitably indexed input cores

"""
function cores2basis(index_map::AbstractIndexMap,cores::AbstractArray...)
  cores2basis(_cores2basis(index_map,cores...))
end

function cores2basis(cores::AbstractArray...)
  c2m = _cores2basis(cores...)
  return dropdims(c2m;dims=1)
end

function cores2basis(core::AbstractArray{T,3}) where T
  pcore = permutedims(core,(2,1,3))
  return reshape(pcore,size(pcore,1),:)
end

function cores2basis(core::SparseCoreCSC{T}) where T
  pcore = permutedims(core,(2,3,1,4))
  return reshape(pcore,size(pcore,1)*size(pcore,2),:)
end

function _cores2basis(a::AbstractArray{S,3},b::AbstractArray{T,3}) where {S,T}
  @check size(a,3) == size(b,1)
  TS = promote_type(T,S)
  nrows = size(a,2)*size(b,2)
  ab = zeros(TS,size(a,1),nrows,size(b,3))
  for i = axes(a,1), j = axes(b,3)
    for α = axes(a,3)
      @inbounds @views ab[i,:,j] += kronecker(b[α,:,j],a[i,:,α])
    end
  end
  return ab
end

# when we multiply a 4-D spatial core with a 3-D temporal core
function _cores2basis(a::AbstractArray{S,4},b::AbstractArray{T,3}) where {S,T}
  @check size(a,4) == size(b,1)
  TS = promote_type(T,S)
  nrows = size(a,2)*size(b,2)
  ncols = size(a,3)
  ab = zeros(TS,size(a,1),nrows*ncols,size(b,3)) # returns a 3-D array
  for i = axes(a,1), j = axes(b,3)
    for α = axes(a,4)
      @inbounds @views ab[i,:,j] += vec(kronecker(b[α,:,j],a[i,:,:,α]))
    end
  end
  return ab
end

function _cores2basis(a::AbstractArray{S,3},b::AbstractArray{T,4}) where {S,T}
  @notimplemented "Usually the spatial cores are computed before the temporal ones"
end

function _cores2basis(a::AbstractArray{S,N},b::AbstractArray{T,N}) where {S,T,N}
  @abstractmethod
end

function _cores2basis(a::AbstractArray,b::AbstractArray...)
  c,d... = b
  return _cores2basis(_cores2basis(a,c),d...)
end

function _cores2basis(i::AbstractIndexMap,a::AbstractArray{T,3}...) where T
  basis = _cores2basis(a...)
  invi = inv_index_map(i)
  return view(basis,:,vec(invi),:)
end

# when we multiply two SparseCoreCSC objects, the result is a 3-D core that stacks
# the matrices' rows and columns
function _cores2basis(
  I::SparseIndexMap,
  a::SparseCoreCSC{S},
  b::SparseCoreCSC{T}
  ) where {S,T}

  @check size(a,4) == size(b,1)
  Is = get_sparse_index_map(I)
  TS = promote_type(T,S)
  nrows = size(a,2)*size(b,2)
  ncols = size(a,3)*size(b,3)
  ab = zeros(TS,size(a,1),nrows*ncols,size(b,4))
  return _cores2basis!(ab,Is,a,b)
end

function _cores2basis(
  I::SparseIndexMap,
  a::SparseCoreCSC{S},
  b::SparseCoreCSC{T},
  c::SparseCoreCSC{U}
  ) where {S,T,U}

  @check size(a,4) == size(b,1) && size(b,4) == size(c,1)
  Is = get_sparse_index_map(I)
  TSU = promote_type(T,S,U)
  nrows = size(a,2)*size(b,2)*size(c,2)
  ncols = size(a,3)*size(b,3)*size(c,3)
  abc = zeros(TSU,size(a,1),nrows*ncols,size(c,4))
  return _cores2basis!(abc,Is,a,b,c)
end

function _cores2basis!(ab,I::AbstractIndexMap,a,b)
  for i = axes(a,1), j = axes(b,4)
    for α = axes(a,4)
      @inbounds @views ab[i,vec(I),j] += kronecker(b.array[α,:,j],a.array[i,:,α])
    end
  end
  return ab
end

function _cores2basis!(abc,I::AbstractIndexMap,a,b,c)
  for i = axes(a,1), j = axes(c,4)
    for α = axes(a,4), β = axes(b,4)
      @inbounds @views abc[i,vec(I),j] += kronecker(c.array[β,:,j],b.array[α,:,β],a.array[i,:,α])
    end
  end
  return abc
end

# Fixed dofs

function _cores2basis!(ab,I::FixedDofsIndexMap,a,b)
  nz_indices = findall(I[:].!=0)
  for i = axes(a,1), j = axes(b,4)
    for α = axes(a,4)
      @inbounds @views ab[i,vec(I),j] += kronecker(b.array[α,:,j],a.array[i,:,α]
        )[nz_indices]
    end
  end
  return view(ab,:,nz_indices,:)
end

function _cores2basis!(abc,I::FixedDofsIndexMap,a,b,c)
  nz_indices = findall(I[:].!=0)
  for i = axes(a,1), j = axes(c,4)
    for α = axes(a,4), β = axes(b,4)
      @inbounds @views abc[i,vec(I),j] += kronecker(c.array[β,:,j],b.array[α,:,β],a.array[i,:,α]
        )[nz_indices]
    end
  end
  return view(abc,:,nz_indices,:)
end
