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

abstract type AbstractIndexMap{D,Ti} <: AbstractArray{Ti,D} end

Base.view(i::AbstractIndexMap,locations) = IndexMapView(i,locations)

function free_dofs_map(i::AbstractIndexMap)
  free_dofs_locations = findall(i.>zero(eltype(i)))
  view(i,free_dofs_locations)
end

function dirichlet_dofs_map(i::AbstractIndexMap)
  dir_dofs_locations = findall(i.<zero(eltype(i)))
  i.indices[dir_dofs_locations]
end

function inv_index_map(i::AbstractIndexMap)
  invi = reshape(invperm(vec(i)),size(i))
  IndexMap(invi)
end

function change_index_map(f,i::AbstractIndexMap)
  i′ = f(i)
  IndexMap(i′)
end

function compose_indices(index::AbstractArray{Ti,D},ncomps::Integer) where {Ti,D}
  indices = zeros(Ti,size(index)...,ncomps)
  @inbounds for comp = 1:ncomps
    selectdim(indices,D+1,comp) .= (index.-1).*ncomps .+ comp
  end
  return MultiValueIndexMap(indices)
end

function fix_dof_index_map(i::AbstractIndexMap,dof_to_fix::Int)
  FixedDofIndexMap(i,dof_to_fix)
end

struct TrivialIndexMap{Ti,D} <: AbstractIndexMap{D,Ti}
  size::NTuple{Ti,D}
end

function TrivialIndexMap(i::AbstractIndexMap)
  TrivialIndexMap(size(i))
end

Base.size(i::TrivialIndexMap) = i.size
Base.getindex(i::TrivialIndexMap,j::Integer) = LinearIndices(i.size)[j]

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

abstract type AbstractMultiValueIndexMap{D,Ti} <: AbstractIndexMap{D,Ti} end

Base.view(i::AbstractMultiValueIndexMap,locations) = MultiValueIndexMapView(i,locations)

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
  IndexMap.(indices)
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
  MultiValuePatternCSC(pa.matrix,ncomps)
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
Base.getindex(i::MultiValueIndexMap,j...) = getindex(i.indices,j...)
TensorValues.num_components(i::MultiValueIndexMap{D}) where D = size(i.indices,D)

struct MultiValueIndexMapView{D,Ti,I,L} <: AbstractMultiValueIndexMap{D,Ti}
  indices::I
  locations::L
  function MultiValueIndexMapView(indices::I,locations::L) where {D,Ti,I<:AbstractMultiValueIndexMap{D,Ti},L}
    new{D,Ti,I,L}(indices,locations)
  end
end

Base.size(i::MultiValueIndexMapView) = _shape_per_dir(i.locations)
Base.getindex(i::MultiValueIndexMapView,j::Integer) = i.indices[i.locations[j]]
TensorValues.num_components(i::MultiValueIndexMapView{D}) where D = size(i,D)

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
