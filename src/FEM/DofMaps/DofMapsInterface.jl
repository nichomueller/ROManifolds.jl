"""
    abstract type AbstractDofMap{D,Ti} <: AbstractArray{Ti,D} end

Type representing an indexing strategy. Subtypes:

- `InverseDofMap`
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
    flatten(i::AbstractDofMap) -> TrivialDofMap

Flattens `i`, the output will be a dof map with ndims == 1
"""
function flatten(i::AbstractDofMap)
  @abstractmethod
end

"""
    change_dof_map(i::AbstractDofMap,args...) -> AbstractDofMap

Creates a new dof map from an old dof map `i`
"""
function change_dof_map(i::AbstractDofMap,args...)
  @abstractmethod
end

"""
    invert(i::AbstractDofMap) -> InverseDofMap

Retruns an InverseDofMap object out of an existing dof map `i`
"""
function invert(i::AbstractDofMap)
  InverseDofMap(i)
end

"""
    struct InverseDofMap{D,Ti,I<:AbstractDofMap{D,Ti}} <: AbstractDofMap{D,Ti}
      dof_map::I
    end

Inverse dof map of a given AbstractDofMap object `dof_map`
"""
struct InverseDofMap{D,Ti,I<:AbstractDofMap{D,Ti}} <: AbstractDofMap{D,Ti}
  dof_map::I
end

Base.size(i::InverseDofMap) = size(i.dof_map)

function Base.getindex(i::InverseDofMap,j::Integer)
  index = i.dof_map[j]
  iszero(index) ? index : j
end

function Base.setindex!(i::InverseDofMap,v,j::Integer)
  index = i.dof_map[j]
  !iszero(value) && setindex!(i.dof_map,v,index)
end

function Base.reshape(i::InverseDofMap,s::Vararg{Int})
  InverseDofMap(reshape(i.dof_map,s...))
end

function flatten(i::InverseDofMap)
  InverseDofMap(flatten(i.dof_map))
end

function change_dof_map(i::InverseDofMap,args...)
  InverseDofMap(change_dof_map(i.dof_map,args...))
end

for f in (:vectorize,:flatten,:invert,:change_dof_map)
  @eval begin
    function $f(i::AbstractVector{<:AbstractDofMap},args...)
      map(i -> $f(i,args...),i)
    end
  end
end

"""
    struct VectorDofMap{D,I<:AbstractVector{<:Integer}} <: AbstractDofMap{D,Int32}
      size::Dims{D}
      bg_dof_to_act_dof::I
    end

Dof map intended for the reindexing of FE vectors (e.g. solutions or residuals).
Fields:
  - `size`: denotes the shape of the array deriving from the reindexing of FE vectors
  - `bg_dof_to_act_dof`: vector of size equals to `prod(size)` that has the following
  purpose. When the FE mesh is Cartesian, `bg_dof_to_act_dof that` is the identity
  map, i.e. `bg_dof_to_act_dof that[i] == i`. When the mesh is not Cartesian, the
  field `size` denotes the size of a bounding (Cartesian) box, and `bg_dof_to_act_dof that`
  creates a correspondence between the set of background DOFs (defined on the bounding box),
  and the active ones (defined on the actual geometry). If the `i`th background DOF
  is equal to the `j`th active DOF, then `bg_dof_to_act_dof that[i] == j`; if
  instead it corresponds to an inactive DOF (e.g. situated in a hole), then
  `bg_dof_to_act_dof that[i] == 0`. Custom data structures that are indexed
  according to an `AbstractDofMap` are designed to return zero when indexed by a
  zero index.

---
**NOTE**

This map only returns the DOFs in lexicographical order. This means that the
entries of the FE vector are not permuted in order, they are simply reshaped and,
in non-Cartesian applications, expanded.

---
"""
struct VectorDofMap{D,I<:AbstractVector{<:Integer}} <: AbstractDofMap{D,Int32}
  size::Dims{D}
  bg_dof_to_act_dof::I
end

VectorDofMap(s::Dims{D}) = VectorDofMap(s,IdentityVector(prod(s)))
VectorDofMap(l::Integer,args...) = VectorDofMap((l,),args...)

function VectorDofMap(s::Dims{D},bg_dof_to_mask::AbstractVector{<:Bool})
  bg_dof_to_act_dof = get_mask_to_act_dof(bg_dof_to_mask,prod(s))
  VectorDofMap(s,bg_dof_to_act_dof)
end

VectorDofMap(i::VectorDofMap) = i
VectorDofMap(i::VectorDofMap,args...) = VectorDofMap(i.size,args...)

Base.size(i::VectorDofMap) = i.size

Base.getindex(i::VectorDofMap,j::Integer) = getindex(i.bg_dof_to_act_dof,j)

Base.setindex!(i::VectorDofMap,v,j::Integer) = setindex!(i.bg_dof_to_act_dof,v,j)

function Base.reshape(i::VectorDofMap,s::Vararg{Int})
  @assert prod(s) == length(i)
  VectorDofMap(s,i.bg_dof_to_act_dof)
end

function flatten(i::VectorDofMap)
  VectorDofMap((prod(i.size),),i.bg_dof_to_act_dof)
end

function change_dof_map(i::VectorDofMap,args...)
  VectorDofMap(i,args...)
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

Index map used to select the nonzero entries of a sparse matrix of sparsity `sparsity`.
The nonzero entries are sorted according to the field `d_sparse_dofs_to_sparse_dofs`
by default. For more details, check the function [`get_d_sparse_dofs_to_full_dofs`](@ref)
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
