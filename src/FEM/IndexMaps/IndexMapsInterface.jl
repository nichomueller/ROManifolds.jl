function _minimum_dir_d(i::AbstractVector{CartesianIndex{D}},d::Integer) where D
  mind = Inf
  for ii in i
    if ii.I[d] < mind
      mind = ii.I[d]
    end
  end
  return mind
end

function _maximum_dir_d(i::AbstractVector{CartesianIndex{D}},d::Integer) where D
  maxd = 0
  for ii in i
    if ii.I[d] > maxd
      maxd = ii.I[d]
    end
  end
  return maxd
end

function _shape_per_dir(i::AbstractVector{CartesianIndex{D}}) where D
  function _admissible_shape(d::Integer)
    mind = _minimum_dir_d(i,d)
    maxd = _maximum_dir_d(i,d)
    @assert all([ii.I[d] ≥ mind for ii in i]) && all([ii.I[d] ≤ maxd for ii in i])
    return maxd - mind + 1
  end
  ntuple(d -> _admissible_shape(d),D)
end

function _shape_per_dir(i::AbstractVector{<:Integer})
  min1 = minimum(i)
  max1 = maximum(i)
  (max1 - min1 + 1,)
end

"""
    abstract type AbstractIndexMap{D,Ti} <: AbstractArray{Ti,D}

Index mapping operator that serves to view a finite element function according to
a nonstandard indexing strategy. Just like the connectivity matrix, the entries of
the index maps are positive when the corresponding dof is free, and negative when
the corresponding dof is Dirichlet.
"""

abstract type AbstractIndexMap{D,Ti} <: AbstractArray{Ti,D} end

Base.view(i::AbstractIndexMap,locations) = IndexMapView(i,locations)

"""
    free_dofs_map(i::AbstractIndexMap) -> AbstractIndexMap

Removes the negative indices from the index map, which correspond to Dirichlet
boundary conditions.
"""

function free_dofs_map(i::AbstractIndexMap)
  free_dofs_locations = findall(i.>zero(eltype(i)))
  s = _shape_per_dir(free_dofs_locations)
  free_dofs_locations′ = reshape(free_dofs_locations,s)
  view(i,free_dofs_locations′)
end

"""
    dirichlet_dofs_map(i::AbstractIndexMap) -> AbstractVector

Removes the positive indices from the index map, which correspond to free dofs.
"""

function dirichlet_dofs_map(i::AbstractIndexMap)
  dir_dofs_locations = findall(i.<zero(eltype(i)))
  i.indices[dir_dofs_locations]
end

"""
    inv_index_map(i::AbstractIndexMap) -> AbstractIndexMap

Returns the inverse index map.
"""

function inv_index_map(i::AbstractIndexMap)
  invi = reshape(invperm(vec(i)),size(i))
  IndexMap(invi)
end

"""
    change_index_map(f,i::AbstractIndexMap) -> AbstractIndexMap

Returns an index map given by f∘i, where f is a function encoding an index map.
"""

function change_index_map(f,i::AbstractIndexMap)
  i′::AbstractIndexMap = f(vec(collect(i)))
  i′
end

"""
    fix_dof_index_map(i::AbstractIndexMap,dof_to_fix::Integer) -> FixedDofIndexMap
"""

function fix_dof_index_map(i::AbstractIndexMap,dof_to_fix::Integer)
  FixedDofIndexMap(i,dof_to_fix)
end

"""
    TrivialIndexMap{Ti,I<:AbstractVector{Ti}} <: AbstractIndexMap{1,Ti}

Represents an index map that does not change the indexing strategy of the FEM function.
In other words, this is simply a wrapper for a LinearIndices list. In the case of sparse
matrices, the indices in a TrivialIndexMap are those of the nonzero elements.
"""

struct TrivialIndexMap{Ti,I<:AbstractVector{Ti}} <: AbstractIndexMap{1,Ti}
  indices::I
end

function TrivialIndexMap(i::AbstractIndexMap)
  TrivialIndexMap(LinearIndices((length(i),)))
end

Base.size(i::TrivialIndexMap) = (length(i.indices),)
Base.getindex(i::TrivialIndexMap,j::Integer) = getindex(i.indices,j)

"""
    IndexMap{D,Ti} <: AbstractIndexMap{D,Ti}

Most standard implementation of an index map.
"""

struct IndexMap{D,Ti} <: AbstractIndexMap{D,Ti}
  indices::Array{Ti,D}
end

Base.size(i::IndexMap) = size(i.indices)
Base.getindex(i::IndexMap{D},j::Vararg{Integer,D}) where D = getindex(i.indices,j...)

"""
    IndexMapView{D,Ti,I<:AbstractIndexMap{D,Ti},L} <: AbstractIndexMap{D,Ti}

View of an AbstractIndexMap at selected locations. Both the index map and the set of locations
have the same dimension. Therefore, this map cannot be used to view boundary indices,
only portions of the domain.
"""

struct IndexMapView{D,Ti,I<:AbstractIndexMap{D,Ti},L} <: AbstractIndexMap{D,Ti}
  indices::I
  locations::L
  function IndexMapView(indices::I,locations::L) where {I,L}
    msg = "The index map and the view locations must have the same dimension"
    @check ndims(indices) == ndims(locations) msg
    D = ndims(indices)
    Ti = eltype(indices)
    new{D,Ti,I,L}(indices,locations)
  end
end

Base.size(i::IndexMapView) = size(i.locations)
Base.getindex(i::IndexMapView{D},j::Vararg{Integer,D}) where D = i.indices[i.locations[j...]]

"""
    FixedDofIndexMap{D,Ti,I<:AbstractIndexMap{D,Ti}} <: AbstractIndexMap{D,Ti}

Index map to be used when imposing a zero mean constraint on a given FE space. The fixed
dof is seen as a CartesianIndex of dimension D.
"""

struct FixedDofIndexMap{D,Ti,I<:AbstractIndexMap{D,Ti}} <: AbstractIndexMap{D,Ti}
  indices::I
  dof_to_fix::CartesianIndex{D}
end

function FixedDofIndexMap(indices::AbstractIndexMap,dof_to_fix::Integer)
  FixedDofIndexMap(indices,CartesianIndices(size(indices))[dof_to_fix])
end

Base.size(i::FixedDofIndexMap) = size(i.indices)

function Base.getindex(i::FixedDofIndexMap{D},j::Vararg{Integer,D}) where D
  if CartesianIndex(j) == i.dof_to_fix
    getindex(i.indices,j...)
  else
    -one(eltype(i))
  end
end

Base.view(i::FixedDofIndexMap,locations) = FixedDofIndexMap(IndexMapView(i.indices,locations),i.dof_to_fix)

"""
    TProductIndexMap{D,Ti,I<:AbstractIndexMap{D,Ti}} <: AbstractIndexMap{D,Ti}

Index map to be used when defining a tensor product FE space on a CartesianDiscreteModel.

"""

struct TProductIndexMap{D,Ti,I<:AbstractIndexMap{D,Ti}} <: AbstractIndexMap{D,Ti}
  indices::I
  indices_1d::Vector{Vector{Ti}}
end

function TProductIndexMap(indices::AbstractArray,indices_1d::AbstractVector{<:AbstractVector})
  TProductIndexMap(IndexMap(indices),indices_1d)
end

Base.size(i::TProductIndexMap) = size(i.indices)
Base.getindex(i::TProductIndexMap{D},j::Vararg{Integer,D}) where D = getindex(i.indices,j...)
get_tp_indices(i::TProductIndexMap) = i.indices
get_univariate_indices(i::TProductIndexMap) = i.indices_1d

struct SparseIndexMap{D,Ti,A,B} <: AbstractIndexMap{D,Ti}
  indices::A
  sparsity::B
  function SparseIndexMap(
    indices::A,
    sparsity::B
    ) where {D,Ti,A<:AbstractIndexMap{D,Ti},B<:TProductSparsityPattern}
    new{D,Ti,A,B}(indices,sparsity)
  end
end

Base.size(i::SparseIndexMap) = size(i.indices)
Base.getindex(i::SparseIndexMap{D},j::Vararg{Integer,D}) where D = getindex(i.indices,j...)
get_index_map(i::SparseIndexMap) = i.indices
get_sparsity(i::SparseIndexMap) = get_sparsity(i.sparsity)
get_univariate_sparsity(i::SparseIndexMap) = get_univariate_sparsity(i.sparsity)

function inv_index_map(i::SparseIndexMap)
  invi = IndexMap(reshape(sortperm(i[:]),size(i)))
  SparseIndexMap(invi,i.sparsity)
end
