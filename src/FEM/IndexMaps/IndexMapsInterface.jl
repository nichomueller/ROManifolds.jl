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
  i′::AbstractIndexMap = f(collect(vec(i)))
  i′
end

"""
    fix_dof_index_map(i::AbstractIndexMap,dofs_to_fix) -> FixedDofsIndexMap

"""
function fix_dof_index_map(i::AbstractIndexMap,dofs_to_fix)
  FixedDofsIndexMap(i,dofs_to_fix)
end

abstract type AbstractTrivialIndexMap <: AbstractIndexMap{1,Int} end

Base.getindex(i::AbstractTrivialIndexMap,j::Integer) = j
Base.setindex!(i::AbstractTrivialIndexMap,v::Integer,j::Integer) = nothing
Base.copy(i::AbstractTrivialIndexMap) = i

"""
    TrivialIndexMap <: AbstractTrivialIndexMap

Represents an index map that does not change the indexing strategy of the FEM function.
In other words, this is simply a wrapper for a LinearIndices list. In the case of sparse
matrices, the indices in a TrivialIndexMap are those of the nonzero elements.

"""
struct TrivialIndexMap <: AbstractTrivialIndexMap
  length::Int
end

TrivialIndexMap(i::AbstractArray) = TrivialIndexMap(length(i))
Base.size(i::TrivialIndexMap) = (length(i.length),)

struct TrivialSparseIndexMap{A<:SparsityPattern} <: AbstractTrivialIndexMap
  sparsity::A
end

TrivialIndexMap(sparsity::SparsityPattern) = TrivialSparseIndexMap(sparsity)
Base.size(i::TrivialSparseIndexMap) = (nnz(i.sparsity),)

recast(a::AbstractArray,i::TrivialSparseIndexMap) = recast(a,i.sparsity)

"""
    IndexMap{D,Ti} <: AbstractIndexMap{D,Ti}

Most standard implementation of an index map.

"""
struct IndexMap{D,Ti} <: AbstractIndexMap{D,Ti}
  indices::Array{Ti,D}
end

Base.size(i::IndexMap) = size(i.indices)
Base.getindex(i::IndexMap{D},j::Vararg{Integer,D}) where D = getindex(i.indices,j...)
Base.setindex!(i::IndexMap{D},v,j::Vararg{Integer,D}) where D = setindex!(i.indices,v,j...)
Base.copy(i::IndexMap) = IndexMap(copy(i.indices))
Base.stack(i::AbstractArray{<:IndexMap}) = IndexMap(stack(get_array.(i)))
Arrays.get_array(i::IndexMap) = i.indices

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
Base.setindex!(i::IndexMapView{D},v,j::Vararg{Integer,D}) where D = setindex!(i.indices,v,i.locations[j...])

"""
    FixedDofsIndexMap{D,Ti,I<:AbstractIndexMap{D,Ti}} <: AbstractIndexMap{D,Ti}

Index map to be used when imposing a zero mean constraint on a given `FESpace`. The fixed
dofs are stored as a vector of CartesianIndex of dimension D; when indexing vectors
defined on the zero mean `FESpace`, the length of the vector is 1

"""
struct FixedDofsIndexMap{D,Ti,I} <: AbstractIndexMap{D,Ti}
  indices::FixedEntriesArray{Ti,D,I}
end

function FixedDofsIndexMap(indices::AbstractIndexMap,dofs_to_fix)
  indices′ = FixedEntriesArray(indices,dofs_to_fix)
  FixedDofsIndexMap(indices′)
end

Arrays.get_array(i::FixedDofsIndexMap) = i.indices
get_fixed_dofs(i::FixedDofsIndexMap) = get_fixed_entries(i.indices)

Base.size(i::FixedDofsIndexMap) = size(i.indices)
Base.getindex(i::FixedDofsIndexMap{D},j::Vararg{Integer,D}) where D = getindex(i.indices,j...)
Base.setindex!(i::FixedDofsIndexMap{D},v,j::Vararg{Integer,D}) where D = setindex!(i.indices,v,j...)
Base.copy(i::FixedDofsIndexMap) = FixedDofsIndexMap(copy(i.indices))
Base.view(i::FixedDofsIndexMap,locations) = FixedDofsIndexMap(view(i.indices,locations))
Base.vec(i::FixedDofsIndexMap) = remove_fixed_dof(i)
Base.stack(i::AbstractArray{<:FixedDofsIndexMap}) = FixedDofsIndexMap(stack(get_array.(i)))

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

abstract type AbstractMultiValueIndexMap{D,Ti} <: AbstractIndexMap{D,Ti} end

Base.view(i::AbstractMultiValueIndexMap,locations) = MultiValueIndexMapView(i,locations)

function compose_indices(index::AbstractArray{Ti,D},ncomps::Integer) where {Ti,D}
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
  IndexMap(indices)
end

function split_components(i::AbstractMultiValueIndexMap{D}) where D
  indices = collect(eachslice(i;dims=D))
  IndexMaps.(indices)
end

function merge_components(i::AbstractVector{<:AbstractArray{Ti}}) where Ti
  sizes = map(size,i)
  @check all(sizes .== [first(sizes)])
  indices = stack(i)
  return indices
end

function permute_sparsity(
  a::SparsityPatternCSC,
  i::AbstractMultiValueIndexMap{D},
  j::AbstractMultiValueIndexMap{D}
  ) where D

  @check size(i,D) == size(j,D)
  ncomps = size(i,D)
  i1 = get_component(i,1)
  j1 = get_component(j,1)
  pa = permute_sparsity(a,j1,i1)
  MultiValueSparsityPatternCSC(pa.matrix,ncomps)
end

function permute_sparsity(a::SparsityPatternCSC,i::AbstractMultiValueIndexMap,j::AbstractIndexMap)
  i1 = get_component(i,1)
  permute_sparsity(a,i1,j)
end

function permute_sparsity(a::SparsityPatternCSC,i::AbstractIndexMap,j::AbstractMultiValueIndexMap)
  j1 = get_component(j,1)
  permute_sparsity(a,i,j1)
end

struct MultiValueIndexMap{D,Ti,I} <: AbstractMultiValueIndexMap{D,Ti}
  indices::I
  function MultiValueIndexMap(indices::I) where {D,Ti,I<:AbstractArray{Ti,D}}
    new{D,Ti,I}(indices)
  end
end

function MultiValueIndexMap(indices::AbstractVector{<:AbstractArray})
  mindices = merge_components(indices)
  return MultiValueIndexMap(mindices)
end

Base.size(i::MultiValueIndexMap) = size(i.indices)
Base.getindex(i::MultiValueIndexMap{D},j::Vararg{Integer,D}) where D = getindex(i.indices,j...)
Base.setindex!(i::MultiValueIndexMap{D},v,j::Vararg{Integer,D}) where D = setindex!(i.indices,v,j...)
Base.copy(i::MultiValueIndexMap) = MultiValueIndexMap(copy(i.indices))
TensorValues.num_components(i::MultiValueIndexMap{D}) where D = size(i.indices,D)

struct MultiValueIndexMapView{D,Ti,I,L} <: AbstractMultiValueIndexMap{D,Ti}
  indices::I
  locations::L
  function MultiValueIndexMapView(indices::I,locations::L) where {D,Ti,I<:AbstractMultiValueIndexMap{D,Ti},L}
    new{D,Ti,I,L}(indices,locations)
  end
end

Base.size(i::MultiValueIndexMapView) = size(i.locations)
Base.getindex(i::MultiValueIndexMapView{D},j::Vararg{Integer,D}) where D = i.indices[i.locations[j...]]
Base.setindex!(i::MultiValueIndexMapView{D},v,j::Vararg{Integer,D}) where D = setindex!(i.indices,v,i.locations[j...])
Base.copy(i::MultiValueIndexMapView) = MultiValueIndexMap(copy(i.indices),i.locations)
TensorValues.num_components(i::MultiValueIndexMapView{D}) where D = size(i,D)

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
Base.setindex!(i::TProductIndexMap{D},v,j::Vararg{Integer,D}) where D = setindex!(i.indices,v,j...)
Base.copy(i::TProductIndexMap) = TProductIndexMap(copy(i.indices),i.indices_1d)
Base.vec(i::TProductIndexMap) = vec(i.indices)
get_tp_indices(i::TProductIndexMap) = i.indices
get_univariate_indices(i::TProductIndexMap) = i.indices_1d

"""
    SparseIndexMap{D,Ti,A<:AbstractIndexMap{D,Ti},B<:TProductSparsityPattern} <: AbstractIndexMap{D,Ti}

Index map used to select the nonzero entries of a sparse matrix. The field `sparsity`
contains the tensor product sparsity of the matrix to be indexed. The field `indices`
refers to the nonzero entries of the sparse matrix, whereas `indices_sparse` is
used to access the corresponding sparse entries

"""
struct SparseIndexMap{D,Ti,A<:AbstractIndexMap{D,Ti},B<:TProductSparsityPattern} <: AbstractIndexMap{D,Ti}
  indices::A
  indices_sparse::A
  sparsity::B
end

Base.size(i::SparseIndexMap) = size(i.indices)
Base.getindex(i::SparseIndexMap{D},j::Vararg{Integer,D}) where D = getindex(i.indices,j...)
Base.setindex!(i::SparseIndexMap{D},v,j::Vararg{Integer,D}) where D = setindex!(i.indices,v,j...)
Base.copy(i::SparseIndexMap) = SparseIndexMap(copy(i.indices),copy(i.indices_sparse),i.sparsity)
Base.vec(i::SparseIndexMap) = vec(i.indices)
get_index_map(i::SparseIndexMap) = i.indices
get_sparse_index_map(i::SparseIndexMap) = i.indices_sparse
get_sparsity(i::SparseIndexMap) = i.sparsity
get_univariate_sparsity(i::SparseIndexMap) = get_univariate_sparsity(i.sparsity)

function inv_index_map(i::SparseIndexMap)
  invi = IndexMap(reshape(sortperm(vec(i.indices)),size(i)))
  invi_sparse = IndexMap(reshape(sortperm(vec(i.indices_sparse)),size(i)))
  SparseIndexMap(invi,invi_sparse,i.sparsity)
end

recast(a::AbstractArray,i::SparseIndexMap) = recast(a,i.sparsity)

const MultiValueSparseIndexMap{D,Ti,A<:AbstractMultiValueIndexMap{D,Ti},B} = SparseIndexMap{D,Ti,A,B}

Base.view(i::MultiValueSparseIndexMap,locations) = MultiValueIndexMapView(i,locations)

function get_component(i::MultiValueSparseIndexMap{D},d;multivalue=true) where D
  ncomps = num_components(i)
  indices = collect(selectdim(i,D,d))
  !multivalue && _to_scalar_values!(indices,ncomps,d)
  IndexMap(indices)
end

function split_components(i::MultiValueSparseIndexMap{D}) where D
  indices = collect(eachslice(i;dims=D))
  IndexMaps.(indices)
end
