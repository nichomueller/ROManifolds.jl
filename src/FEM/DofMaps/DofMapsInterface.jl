"""
    abstract type AbstractDofMap{D,Ti} <: AbstractArray{Ti,D} end

Type representing an indexing strategy. Subtypes:

- `VectorDofMap`
- `TrivialSparseMatrixDofMap`
- `SparseMatrixDofMap`
"""
abstract type AbstractDofMap{D,Ti} <: AbstractArray{Ti,D} end

"""
    const TrivialDofMap{Ti} = AbstractDofMap{1,Ti}
"""
const TrivialDofMap{Ti} = AbstractDofMap{1,Ti}

Base.IndexStyle(i::AbstractDofMap) = IndexLinear()

function remove_masked_dofs(i::AbstractDofMap)
  i′ = collect(vec(i))
  deleteat!(i′,findall(iszero,i′))
  i′
end

"""
    vectorize(i::AbstractDofMap) -> AbstractVector

Reshapes `i` as a vector, and removes the masked dofs in `i` (if any)
"""
function vectorize(i::AbstractDofMap)
  remove_masked_dofs(i)
end

"""
    invert(i::AbstractDofMap) -> AbstractArray

Calls the function `invperm` on the nonzero entries of `i`, and places
zeros on the remaining zero entries of `i`. The output has the same size as `i`
"""
function invert(i::AbstractDofMap)
  i′ = remove_masked_dofs(i)
  inz = findall(!iszero,i′)
  i′[inz] = invperm(i[inz])
  i′
end

"""
    flatten(i::AbstractDofMap) -> TrivialDofMap

Flattens `i`, the output will be a dof map with ndims == 1
"""
function flatten(i::AbstractDofMap)
  @abstractmethod
end

"""
    struct VectorDofMap{D} <: AbstractDofMap{D,Int32}
      size::Dims{D}
      dof_to_mask::Vector{Bool}
      cumulative_masks::Vector{Int}
    end

Dof map intended for FE vectors (e.g. solutions or residuals). The field `size`
denotes the shape of the vector, and `dof_to_mask` tracks the dofs to hide, and
`cumulative_masks` simply enables the correct indexing of the map
"""
struct VectorDofMap{D} <: AbstractDofMap{D,Int32}
  size::Dims{D}
  dof_to_mask::Vector{Bool}
  cumulative_masks::Vector{Int}
end

function VectorDofMap(s::Dims{D},dof_to_mask::Vector{Bool}=fill(false,prod(s))) where D
  @check length(dof_to_mask) == prod(s)
  cumulative_masks = cumsum(dof_to_mask)
  VectorDofMap(s,dof_to_mask,cumulative_masks)
end

VectorDofMap(i::VectorDofMap) = i
VectorDofMap(l::Integer,args...) = VectorDofMap((l,),args...)

function VectorDofMap(i::VectorDofMap,_mask_to_add::Vector{Bool})
  @check length(i) == length(_mask_to_add)
  dof_to_mask = copy(i.dof_to_mask)
  for j in eachindex(i)
    dof_to_mask[j] = dof_to_mask[j] || _mask_to_add[j]
  end
  VectorDofMap(i.size,dof_to_mask)
end

Base.size(i::VectorDofMap) = i.size

function Base.getindex(i::VectorDofMap,j::Integer)
  i.dof_to_mask[j] ? zero(eltype(i)) : j-i.cumulative_masks[j]
end

function Base.setindex!(i::VectorDofMap,v,j::Integer)
  !i.dof_to_mask[j] && j-i.cumulative_masks[j]
end

function Base.reshape(i::VectorDofMap,s::Vararg{Int})
  @assert prod(s) == length(i)
  VectorDofMap(s,i.dof_to_mask,i.cumulative_masks)
end

function flatten(i::VectorDofMap)
  VectorDofMap((prod(i.size),),i.dof_to_mask,i.cumulative_masks)
end

abstract type SparseDofMapStyle end
struct SparseDofMapIndexing <: SparseDofMapStyle end
struct FullDofMapIndexing <: SparseDofMapStyle end

"""
    struct TrivialSparseMatrixDofMap{A<:SparsityPattern} <: TrivialDofMap{Int}
      sparsity::A
    end

Index map used to select the nonzero entries of a sparse matrix of sparsity `sparsity`
"""
struct TrivialSparseMatrixDofMap{A<:SparsityPattern} <: TrivialDofMap{Int}
  sparsity::A
end

function TrivialDofMap(i::TrivialSparseMatrixDofMap)
  TrivialSparseMatrixDofMap(i.sparsity)
end

SparseDofMapStyle(i::TrivialSparseMatrixDofMap) = SparseDofMapIndexing()

Base.size(i::TrivialSparseMatrixDofMap) = (nnz(i.sparsity),)
Base.getindex(i::TrivialSparseMatrixDofMap,j::Integer) = j

function flatten(i::TrivialSparseMatrixDofMap)
  i
end

function recast(a::AbstractArray,i::TrivialSparseMatrixDofMap)
  recast(a,i.sparsity)
end

"""
    struct SparseMatrixDofMap{D,Ti,A<:SparsityPattern,B<:SparseDofMapStyle} <: AbstractDofMap{D,Ti}
      d_sparse_dofs_to_sparse_dofs::Array{Ti,D}
      d_sparse_dofs_to_full_dofs::Array{Ti,D}
      sparsity::A
      index_style::B
    end

Index map used to select the nonzero entries of a sparse matrix of sparsity `sparsity`
"""
struct SparseMatrixDofMap{D,Ti,A<:SparsityPattern,B<:SparseDofMapStyle} <: AbstractDofMap{D,Ti}
  d_sparse_dofs_to_sparse_dofs::Array{Ti,D}
  d_sparse_dofs_to_full_dofs::Array{Ti,D}
  sparsity::A
  index_style::B
end

function SparseMatrixDofMap(
  d_sparse_dofs_to_sparse_dofs::AbstractArray,
  d_sparse_dofs_to_full_dofs::AbstractArray,
  sparsity::SparsityPattern)

  index_style = SparseDofMapIndexing()
  SparseMatrixDofMap(d_sparse_dofs_to_sparse_dofs,d_sparse_dofs_to_full_dofs,sparsity,index_style)
end

SparseDofMapStyle(i::SparseMatrixDofMap) = i.index_style

function FullDofMapIndexing(i::SparseMatrixDofMap)
  SparseMatrixDofMap(
    i.d_sparse_dofs_to_sparse_dofs,
    i.d_sparse_dofs_to_full_dofs,
    i.sparsity,
    FullDofMapIndexing()
    )
end

function SparseDofMapIndexing(i::SparseMatrixDofMap)
  SparseMatrixDofMap(
    i.d_sparse_dofs_to_sparse_dofs,
    i.d_sparse_dofs_to_full_dofs,
    i.sparsity,
    SparseDofMapIndexing()
    )
end

Base.size(i::SparseMatrixDofMap) = size(i.d_sparse_dofs_to_sparse_dofs)

function Base.getindex(i::SparseMatrixDofMap{D,Ti,A,SparseDofMapIndexing},j::Integer) where {D,Ti,A}
  getindex(i.d_sparse_dofs_to_sparse_dofs,j)
end

function Base.setindex!(i::SparseMatrixDofMap{D,Ti,A,SparseDofMapIndexing},v,j::Integer) where {D,Ti,A}
  setindex!(i.d_sparse_dofs_to_sparse_dofs,v,j)
end

function Base.getindex(i::SparseMatrixDofMap{D,Ti,A,FullDofMapIndexing},j::Integer) where {D,Ti,A}
  getindex(i.d_sparse_dofs_to_full_dofs,j)
end

function Base.setindex!(i::SparseMatrixDofMap{D,Ti,A,FullDofMapIndexing},v,j::Integer) where {D,Ti,A}
  setindex!(i.d_sparse_dofs_to_full_dofs,v,j)
end

function flatten(i::SparseMatrixDofMap)
  TrivialSparseMatrixDofMap(i.sparsity)
end

function recast(a::AbstractArray,i::SparseMatrixDofMap)
  recast(a,i.sparsity)
end

# utils

# i1 ∘ i2
function compose_maps(i1::AbstractArray{Ti,D},i2::AbstractArray{Ti,D}) where {Ti,D}
  @assert size(i1) == size(i2)
  i12 = zeros(Ti,size(i1))
  for (i,m2i) in enumerate(i2)
    iszero(m2i) && continue
    i12[i] = i1[m2i]
  end
  return i12
end
