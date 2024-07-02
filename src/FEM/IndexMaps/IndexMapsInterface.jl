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
    abstract type AbstractIndexMap{D,Ti} <: AbstractArray{Ti,D} end

Index mapping operator that serves to view a finite element function according to
a nonstandard indexing strategy. Just like the connectivity matrix, the entries of
the index maps are positive when the corresponding dof is free, and negative when
the corresponding dof is dirichlet.

Subtypes:
- [`TrivialIndexMap`](@ref)
- [`IndexMap`](@ref)
- [`IndexMapView`](@ref)
- [`FixedDofsIndexMap`](@ref)
- [`TProductIndexMap`](@ref)
- [`SparseIndexMap`](@ref)
- [`AbstractMultiValueIndexMap`](@ref)

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

Returns the inverse of the index map defined by `i`.

"""
function inv_index_map(i::AbstractIndexMap)
  invi = reshape(invperm(vec(i)),size(i))
  IndexMap(invi)
end

"""
    change_index_map(f,i::AbstractIndexMap) -> AbstractIndexMap

Returns an index map given by `f`∘`i`, where f is a function encoding an index map.

"""
function change_index_map(f,i::AbstractIndexMap)
  i′::AbstractIndexMap = f(vec(collect(i)))
  i′
end

"""
    fix_dof_index_map(i::AbstractIndexMap,dofs_to_fix) -> FixedDofsIndexMap

"""
function fix_dof_index_map(i::AbstractIndexMap,dofs_to_fix)
  FixedDofsIndexMap(i,dofs_to_fix)
end

"""
    recast(i::AbstractIndexMap,a::AbstractArray) -> AbstractArray

Recasting operation of an array according to the index map `i`
"""
function recast(i::AbstractIndexMap,a::AbstractArray)
  return a
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
    FixedDofsIndexMap{D,Ti,I<:AbstractIndexMap{D,Ti}} <: AbstractIndexMap{D,Ti}

Index map to be used when imposing a zero mean constraint on a given `FESpace`. The fixed
dofs are stored as a vector of CartesianIndex of dimension D; when indexing vectors
defined on the zero mean `FESpace`, the length of the vector is 1

"""
struct FixedDofsIndexMap{D,Ti,I<:AbstractIndexMap{D,Ti}} <: AbstractIndexMap{D,Ti}
  indices::FixedEntriesArray{Ti,D,I}
end

function FixedDofsIndexMap(indices::AbstractIndexMap,dofs_to_fix)
  indices′ = FixedEntriesArray(indices,dofs_to_fix)
  FixedDofsIndexMap(indices′)
end

Base.size(i::FixedDofsIndexMap) = size(i.indices)

Base.getindex(i::FixedDofsIndexMap{D},j::Vararg{Integer,D}) where D = getindex(i.indices,j...)

Base.view(i::FixedDofsIndexMap,locations) = FixedDofsIndexMap(view(i.indices,locations))

for f in (:free_dofs_map,:inv_index_map)
  @eval begin
    function $f(a::FixedEntriesArray)
      FixedEntriesArray($f(a.array),a.fixed_entries)
    end

    function $f(i::FixedDofsIndexMap)
      indices′ = $f(i.indices)
      FixedDofsIndexMap(indices′)
    end
  end
end

function remove_fixed_dof(i::FixedDofsIndexMap)
  filter(x -> x != 0,i)
end

function permute_sparsity(a::SparsityPatternCSC,i::FixedDofsIndexMap,j::AbstractIndexMap)
  permute_sparsity(a,remove_fixed_dof(i),vec(j))
end

function permute_sparsity(a::SparsityPatternCSC,i::AbstractIndexMap,j::FixedDofsIndexMap)
  permute_sparsity(a,vec(i),remove_fixed_dof(j))
end

function permute_sparsity(a::SparsityPatternCSC,i::FixedDofsIndexMap,j::FixedDofsIndexMap)
  permute_sparsity(a,remove_fixed_dof(i),remove_fixed_dof(j))
end

"""
    TProductIndexMap{D,Ti,I<:AbstractIndexMap{D,Ti}} <: AbstractIndexMap{D,Ti}

Index map to be used when defining a [`TProductFESpace`](@ref) on a CartesianDiscreteModel.

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

"""
    SparseIndexMap{D,Ti,A<:AbstractIndexMap{D,Ti},B<:TProductSparsityPattern} <: AbstractIndexMap{D,Ti}

Index map used to select the nonzero entries of a sparse matrix. The field `sparsity`
contains the tensor product sparsity of the matrix to be indexed.

"""
struct SparseIndexMap{D,Ti,A<:AbstractIndexMap{D,Ti},B<:TProductSparsityPattern} <: AbstractIndexMap{D,Ti}
  indices::A
  sparsity::B
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

# MultiField interface

abstract type AbstractMultiValueIndexMap{D,Ti} <: AbstractIndexMap{D,Ti} end

Base.view(i::AbstractMultiValueIndexMap,locations) = MultiValueIndexMapView(i,locations)

function compose_indices(index::AbstractArray{Ti,D},ncomps::Integer) where {Ti,D}
  # indices = zeros(Ti,size(index)...,ncomps)
  # @inbounds for comp = 1:ncomps
  #   selectdim(indices,D+1,comp) .= (index.-1).*ncomps .+ comp
  # end
  indices = repeat(index;outer=(ntuple(_->1,Val{D}())...,ncomps))
  return MultiValueIndexMap(indices)
end

function _to_scalar_values!(indices::AbstractArray,D::Integer,d::Integer)
  indices .= (indices .- d) ./ D .+ 1
end

function get_component(i::AbstractMultiValueIndexMap{D},d;multivalue=true) where D
  ncomps = num_components(i)
  indices = collect(selectdim(i,D,d))
  !multivalue && _to_scalar_values!(indices,ncomps,d)
  return IndexMap(indices)
end

function split_components(i::AbstractMultiValueIndexMap{D}) where D
  indices = collect(eachslice(i;dims=D))
  return IndexMaps.(indices)
end

function merge_components(i::AbstractVector{<:AbstractArray{Ti}}) where Ti
  sizes = map(size,i)
  @check all(sizes .== [first(sizes)])
  indices = stack(i)
  return indices
end

function permute_sparsity(a::SparsityPatternCSC,i::AbstractMultiValueIndexMap,j::AbstractMultiValueIndexMap)
  ncomps_i = num_components(i)
  ncomps_j = num_components(j)
  @check ncomps_i == ncomps_j
  i1 = get_component(i,1)
  j1 = get_component(j,1)
  pa = permute_sparsity(a,i1,j1)
  return MultiValueSparsityPatternCSC(pa.matrix,ncomps_i)
end

for T in (:AbstractIndexMap,:FixedDofsIndexMap)
  @eval begin
    function permute_sparsity(a::SparsityPatternCSC,i::AbstractMultiValueIndexMap,j::$T)
      i1 = get_component(i,1)
      permute_sparsity(a,i1,j)
    end

    function permute_sparsity(a::SparsityPatternCSC,i::$T,j::AbstractMultiValueIndexMap)
      j1 = get_component(j,1)
      permute_sparsity(a,i,j1)
    end
  end
end

function TrivialIndexMap(i::AbstractMultiValueIndexMap)
  TrivialIndexMap(merge_components(i))
end

struct MultiValueIndexMap{D,Ti,I<:AbstractArray{Ti,D}} <: AbstractMultiValueIndexMap{D,Ti}
  indices::I
end

function MultiValueIndexMap(indices::AbstractVector{<:AbstractArray})
  mindices = merge_components(indices)
  return MultiValueIndexMap(mindices)
end

Base.size(i::MultiValueIndexMap) = size(i.indices)
Base.getindex(i::MultiValueIndexMap,j...) = getindex(i.indices,j...)
TensorValues.num_components(i::MultiValueIndexMap{D}) where D = size(i.indices,D)

struct MultiValueIndexMapView{D,Ti,I<:AbstractMultiValueIndexMap{D,Ti},L} <: AbstractMultiValueIndexMap{D,Ti}
  indices::I
  locations::L
end

Base.size(i::MultiValueIndexMapView) = size(i.locations)
Base.getindex(i::MultiValueIndexMapView{D},j::Vararg{Integer,D}) where D = i.indices[i.locations[j...]]
TensorValues.num_components(i::MultiValueIndexMapView{D}) where D = size(i,D)
