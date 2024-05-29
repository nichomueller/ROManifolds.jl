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

Base.IndexStyle(::Type{<:AbstractIndexMap}) = IndexLinear()

function free_dofs_map(i::AbstractIndexMap)
  free_dofs_locations = findall(i.>zero(eltype(i)))
  IndexMap(i[free_dofs_locations])
end

function dirichlet_dofs_map(i::AbstractIndexMap)
  dir_dofs_locations = findall(i.<zero(eltype(i)))
  i.indices[dir_dofs_locations]
end

function inv_index_map(i::AbstractIndexMap)
  invi = reshape(invperm(vec(i)),size(i))
  IndexMap(invi)
end

function vectorize_index_map(i::AbstractIndexMap)
  vi = vec(collect(LinearIndices(size(i))))
  IndexMap(vi)
end

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

struct MultiValueIndexMap{D,Ti,I} <: AbstractIndexMap{D,Ti}
  indices::I
  function MultiValueIndexMap(indices::I) where {D,Ti,I<:AbstractArray{Ti,D}}
    new{D,Ti,I}(indices)
  end
end

Base.size(i::MultiValueIndexMap) = size(i.indices)
Base.getindex(i::MultiValueIndexMap,j...) = getindex(i.indices,j...)

function get_component(i::MultiValueIndexMap{D},d) where D
  indices = collect(selectdim(i.indices,D,d))
  IndexMap(indices)
end

function split_components(i::MultiValueIndexMap{D}) where D
  indices = collect(eachslice(i.indices;dims=D))
  IndexMap.(indices)
end

function get_component(i::IndexMapView{D,Ti,<:MultiValueIndexMap{D,Ti}},d) where {D,Ti}
  id = get_component(i.indices,d)
  free_dofs_map(id)
end

function split_components(i::IndexMapView{D,Ti,<:MultiValueIndexMap{D,Ti}}) where {D,Ti}
  id = split_components(i.indices)
  free_dofs_map.(id)
end

function permute_sparsity(a::SparsityPatternCSC,i::MultiValueIndexMap{D},j::MultiValueIndexMap{D}) where D
  ncomps = D
  i1 = get_component(i,1)
  j1 = get_component(j,1)
  pa = permute_sparsity(a,j1,i1)
  MultiValuePatternCSC(pa.matrix,ncomps)
end

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

# some index utils

function recast_indices(indices::AbstractVector,A::AbstractArray)
  nonzero_indices = get_nonzero_indices(A)
  entire_indices = nonzero_indices[indices]
  return entire_indices
end

function sparsify_indices(indices::AbstractVector,A::AbstractArray)
  nonzero_indices = get_nonzero_indices(A)
  sparse_indices = map(y->findfirst(x->x==y,nonzero_indices),indices)
  return sparse_indices
end

function get_nonzero_indices(A::AbstractVector)
  @notimplemented
end

function get_nonzero_indices(A::AbstractMatrix)
  return axes(A,1)
end

function get_nonzero_indices(A::AbstractSparseMatrix)
  i,j, = findnz(A)
  return i .+ (j .- 1)*A.m
end

function get_nonzero_indices(A::AbstractArray{T,3} where T)
  return axes(A,2)
end

function tensorize_indices(i::Integer,dofs::AbstractVector{<:Integer})
  D = length(dofs)
  cdofs = cumprod(dofs)
  ic = ()
  @inbounds for d = 1:D-1
    ic = (ic...,fast_index(i,cdofs[d]))
  end
  ic = (ic...,slow_index(i,cdofs[D-1]))
  return CartesianIndex(ic)
end

function tensorize_indices(indices::AbstractVector,dofs::AbstractVector{<:Integer})
  D = length(dofs)
  tindices = Vector{CartesianIndex{D}}(undef,length(indices))
  @inbounds for (ii,i) in enumerate(indices)
    tindices[ii] = tensorize_indices(i,dofs)
  end
  return tindices
end

function split_row_col_indices(i::CartesianIndex{D},dofs::AbstractMatrix{<:Integer}) where D
  @check size(dofs) == (2,D)
  nrows = view(dofs,:,1)
  irc = ()
  @inbounds for d = 1:D
    irc = (irc...,fast_index(i.I[d],nrows[d]),slow_index(i.I[d],nrows[d]))
  end
  return CartesianIndex(irc)
end

function split_row_col_indices(indices::AbstractVector{CartesianIndex{D}},dofs::AbstractMatrix{<:Integer}) where D
  rcindices = Vector{CartesianIndex{2*D}}(undef,length(indices))
  @inbounds for (ii,i) in enumerate(indices)
    rcindices[ii] = split_row_col_indices(i,dofs)
  end
  return rcindices
end
