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

struct BlockArrayTTCores{T,D,N,A<:AbstractArray{T,D}} <: AbstractArray{AbstractArray{T,D},N}
  array::Vector{A}
  touched::Array{Bool,N}
end

const BlockVectorTTCores{T,D,A} = BlockArrayTTCores{T,1,D,A}
const BlockMatrixTTCores{T,D,A} = BlockArrayTTCores{T,2,D,A}

Base.size(a::BlockArrayTTCores) = size(a.touched)

function Base.getindex(a::BlockArrayTTCores{T,D,N},i::Vararg{Integer,N}) where {T,D,N}
  iblock = first(i)
  if all(i.==iblock)
    a.array[iblock]
  else
    fill(zero(T),first(a.array))
  end
end

function get_touched_blocks(a::BlockArrayTTCores)
  findall(a.touched)
end

# core operations

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

function cores2basis(core::MultiValueSparseCoreCSC{T}) where T
  pcore = permutedims(core,(2,3,4,1,5))
  return reshape(pcore,size(pcore,1)*size(pcore,2)*size(pcore,3),:)
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

function _cores2basis(i::FixedDofsIndexMap,a::AbstractArray{T,3}...) where T
  basis = _cores2basis(a...)
  invi = inv_index_map(i)
  return view(basis,:,remove_fixed_dof(invi),:)
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

function union_cores(a::AbstractVector{<:AbstractArray}...)
  @check all(length(ai) == length(first(a)) for ai in a)
  [union_first_cores(first.(a)...),union_cores.(getindex.(a,2:length(a)))...]
end

function union_first_cores(a::A...) where {T,D,A<:AbstractArray{T,D}} # -> BlockVectorTTCores
  @check all(size(ai,2) == size(first(a),2) for ai in a)
  array = [a...]
  touched = fill(true,s)
  BlockArrayTTCores(array,touched)
end

function union_cores(a::A...) where {T,D,A<:AbstractArray{T,D}} # -> BlockMatrixTTCores
  @check all(size(ai,2) == size(first(a),2) for ai in a)
  array = [a...]
  s = (length(a),length(a))
  touched = fill(false,k.size)
  for i in diag(CartesianIndices(s))
    touched[i] = true
  end
  BlockArrayTTCores(array,touched)
end

function cores2basis(cores::BlockArrayTTCores...)
  c2m = _cores2basis(cores...)
  array = dropdims.(c2m.array;dims=1)
  touched = c2m.touched
  BlockArrayTTCores(array,touched)
end

function _cores2basis(a::BlockVectorTTCores,b::BlockMatrixTTCores)
  @check length(a) == size(b,1)
  @check (a.touched[i] && b.touched[i,i] for i in eachindex(a))
  array = [_cores2basis(a[i],b[i,i]) for i in eachindex(a)]
  touched = a.touched
  BlockArrayTTCores(array,touched)
end

function _cores2basis(a::BlockMatrixTTCores,b::BlockMatrixTTCores)
  @check (a.touched[i,i] && b.touched[i,i] for i in eachindex(a))
  array = [_cores2basis(a[i],b[i,i]) for i in eachindex(a)]
  touched = a.touched
  BlockArrayTTCores(array,touched)
end

function _cores2basis(i::AbstractIndexMap,a::BlockArrayTTCores...)
  basis = _cores2basis(a...)
  invi = inv_index_map(i)
  array = map(b -> view(b,:,vec(invi),:),basis.array)
  touched = basis.touched
  ArrayBlock(array,touched)
end
