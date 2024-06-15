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
  view(i,free_dofs_locations)
end

"""
    dirichlet_dofs_map(i::AbstractIndexMap) -> AbstractVector

Removes the positive indices from the index map, which correspond to Dirichlet
boundary conditions.
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
  i′::AbstractIndexMap = f(collect(i))
  i′
end

"""
    fix_dof_index_map(i::AbstractIndexMap,dof_to_fix::Integer) -> FixedDofIndexMap
"""

function fix_dof_index_map(i::AbstractIndexMap,dof_to_fix::Integer)
  FixedDofIndexMap(i,dof_to_fix)
end

struct TrivialIndexMap{Ti,I<:AbstractVector{Ti}} <: AbstractIndexMap{1,Ti}
  indices::I
end

function TrivialIndexMap(i::AbstractIndexMap)
  TrivialIndexMap(LinearIndices((length(i),)))
end

Base.size(i::TrivialIndexMap) = (length(i.indices),)
Base.getindex(i::TrivialIndexMap,j::Integer) = getindex(i.indices,j)

struct IndexMap{D,Ti} <: AbstractIndexMap{D,Ti}
  indices::Array{Ti,D}
end

Base.size(i::IndexMap) = size(i.indices)
Base.getindex(i::IndexMap,j...) = getindex(i.indices,j...)

struct IndexMapView{D,Ti,I,L} <: AbstractIndexMap{D,Ti}
  indices::I
  locations::L
  function IndexMapView(indices::I,locations::L) where {D,Ti,I<:AbstractIndexMap{D,Ti},L}
    new{D,Ti,I,L}(indices,locations)
  end
end

Base.size(i::IndexMapView) = _shape_per_dir(i.locations)
Base.getindex(i::IndexMapView,j::Integer) = i.indices[i.locations[j]]

struct FixedDofIndexMap{D,Ti,I} <: AbstractIndexMap{D,Ti}
  indices::I
  dof_to_fix::Int
  function FixedDofIndexMap(indices::I,dof_to_fix::Int) where {D,Ti,I<:AbstractIndexMap{D,Ti}}
    new{D,Ti,I}(indices,dof_to_fix)
  end
end

Base.size(i::FixedDofIndexMap) = size(i.indices)
Base.getindex(i::FixedDofIndexMap,j::Integer) = fixed_getindex(Val(j==i.dof_to_fix),i.indices,j)
fixed_getindex(::Val{false},i::AbstractArray,j::Int) = getindex(i,j)
fixed_getindex(::Val{true},i::AbstractArray,j::Int) = -one(eltype(i))

Base.view(i::FixedDofIndexMap,locations) = FixedDofIndexMap(IndexMapView(i.indices,locations),i.dof_to_fix)

struct TProductIndexMap{D,Ti,I} <: AbstractIndexMap{D,Ti}
  indices::I
  indices_1d::Vector{Vector{Ti}}
  function TProductIndexMap(
    indices::I,
    indices_1d::Vector{Vector{Ti}}
    ) where {D,Ti,I<:AbstractIndexMap{D,Ti}}
    new{D,Ti,I}(indices,indices_1d)
  end
end

function TProductIndexMap(indices::AbstractArray,indices_1d::AbstractVector{<:AbstractVector})
  TProductIndexMap(IndexMap(indices),indices_1d)
end

Base.size(i::TProductIndexMap) = size(i.indices)
Base.getindex(i::TProductIndexMap,j::Integer) = getindex(i.indices,j...)
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
Base.getindex(i::SparseIndexMap,j...) = getindex(i.indices,j...)
get_index_map(i::SparseIndexMap) = i.indices
get_sparsity(i::SparseIndexMap) = get_sparsity(i.sparsity)
get_univariate_sparsity(i::SparseIndexMap) = get_univariate_sparsity(i.sparsity)

function inv_index_map(i::SparseIndexMap)
  invi = IndexMap(reshape(sortperm(i[:]),size(i)))
  SparseIndexMap(invi,i.sparsity)
end
