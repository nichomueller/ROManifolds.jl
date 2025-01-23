"""
    abstract type AbstractDofMap{D,Ti} <: AbstractArray{Ti,D} end

Type representing an indexing strategy. Subtypes:

- `VectorDofMap`
- `TrivialSparseMatrixDofMap`
- `SparseMatrixDofMap`
- `ConstrainedDofMap`
"""
abstract type AbstractDofMap{D,Ti} <: AbstractArray{Ti,D} end

"""
    const TrivialDofMap{Ti} = AbstractDofMap{1,Ti}
"""
const TrivialDofMap{Ti} = AbstractDofMap{1,Ti}

Base.IndexStyle(i::AbstractDofMap) = IndexLinear()

FESpaces.ConstraintStyle(i::I) where I<:AbstractDofMap = ConstraintStyle(I)
FESpaces.ConstraintStyle(::Type{<:AbstractDofMap}) = UnConstrained()

function remove_constrained_dofs(i::AbstractDofMap)
  remove_constrained_dofs(i,ConstraintStyle(i))
end

function remove_constrained_dofs(i::AbstractDofMap,::UnConstrained)
  collect(i)
end

function remove_constrained_dofs(i::AbstractDofMap{Ti},::Constrained) where Ti
  i′ = collect(vec(i))
  deleteat!(i′,get_dof_to_constraints(i))
  i′
end

get_dof_to_constraints(i::AbstractDofMap) = @abstractmethod

"""
    vectorize(i::AbstractDofMap) -> AbstractVector

Reshapes `i` as a vector, and removes the constrained dofs in `i` (if any)
"""
function vectorize(i::AbstractDofMap)
  vec(remove_constrained_dofs(i))
end

"""
    invert(i::AbstractDofMap;kwargs...) -> AbstractArray

Calls the function `invperm` on the nonzero entries of `i`, and places
zeros on the remaining zero entries of `i`. The output has the same size as `i`
"""
function invert(i::AbstractDofMap)
  invert(i,ConstraintStyle(i))
end

function invert(i::AbstractArray,::UnConstrained)
  i′ = similar(i)
  inz = findall(!iszero,i)
  i′[inz] = invperm(i[inz])
  i′
end

function invert(i::AbstractDofMap,::Constrained)
  i′ = remove_constrained_dofs(i)
  invert(i′,UnConstrained())
end

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

"""
    flatten(i::AbstractDofMap) -> TrivialDofMap

Flattens `i`, the output will be a dof map with ndims == 1
"""
function flatten(i::AbstractDofMap)
  @abstractmethod
end

struct VectorDofMap{D} <: AbstractDofMap{D,Int32}
  size::NTuple{D,Int}
end

VectorDofMap(l::Integer) = VectorDofMap((l,))

Base.size(i::VectorDofMap) = i.size
Base.getindex(i::VectorDofMap,j::Integer) = j
Base.setindex!(i::VectorDofMap,v,j::Integer) = v

function Base.reshape(i::VectorDofMap,s::Int...)
  @assert prod(s) == length(i)
  VectorDofMap(s)
end

function flatten(i::VectorDofMap)
  VectorDofMap((prod(i.size),))
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

"""
    struct ConstrainedDofMap{D,Ti,I<:AbstractDofMap{D,Ti}} <: AbstractDofMap{D,Ti}
      indices::I
      dof_to_constraint_mask::Vector{Bool}
    end

Contains an additional field `dof_to_constraint_mask`
which tracks the constrained dofs. If a dof is constrained, a zero is shown at its
place
"""
struct ConstrainedDofMap{D,Ti,I<:AbstractDofMap{D,Ti}} <: AbstractDofMap{D,Ti}
  indices::I
  dof_to_constraint_mask::Vector{Bool}
end

FESpaces.ConstraintStyle(::Type{<:ConstrainedDofMap}) = Constrained()

get_dof_to_constraints(i::ConstrainedDofMap) = i.dof_to_constraint_mask

Base.size(i::ConstrainedDofMap) = size(i.indices)

function Base.getindex(i::ConstrainedDofMap,j::Integer)
  i.dof_to_constraint_mask[j] ? zero(eltype(i)) : getindex(i.indices,j)
end

function Base.setindex!(i::ConstrainedDofMap,v,j::Integer)
  !i.dof_to_constraint_mask[j] && setindex!(i.indices,v,j)
end

function Base.reshape(i::ConstrainedDofMap,s::Int...)
  ConstrainedDofMap(reshape(i.indices,s...),i.dof_to_constraint_mask)
end

function flatten(i::ConstrainedDofMap)
  ConstrainedDofMap(flatten(i.indices),i.dof_to_constraint_mask)
end

function recast(a::AbstractArray,i::ConstrainedDofMap)
  recast(a,i.indices)
end
