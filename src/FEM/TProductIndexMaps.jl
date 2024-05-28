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
  function _admissible_shape(d::Int)
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

abstract type AbstractIndexMap{D} <: AbstractArray{Int,D} end

Base.IndexStyle(::Type{<:AbstractIndexMap}) = IndexLinear()

struct IndexMap{D} <: AbstractIndexMap{D}
  indices::Array{Int,D}
end

Base.size(i::IndexMap) = size(i.indices)
Base.getindex(i::IndexMap,j...) = getindex(i.indices,j...)

function free_dofs_map(i::IndexMap)
  free_dofs_locations = findall(i.indices.>0)
  IndexMapView(i.indices,free_dofs_locations)
end

function dirichlet_dofs_map(i::IndexMap)
  dir_dofs_locations = findall(i.indices.<0)
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

struct IndexMapView{D,L} <: AbstractIndexMap{D}
  indices::Array{Int,D}
  locations::L
end

Base.size(i::IndexMapView) = _shape_per_dir(i.locations)
Base.getindex(i::IndexMapView,j::Int) = i.indices[i.locations[j]]

struct TProductIndexMap{D} <: AbstractIndexMap{D}
  indices::Array{Int,D}
  indices_1d::Vector{Vector{Int}}
end

Base.size(i::TProductIndexMap) = size(i.indices)
Base.getindex(i::TProductIndexMap,j::Int) = getindex(i.indices,j...)
get_tp_indices(i::TProductIndexMap) = i.indices
get_univariate_indices(i::TProductIndexMap) = i.indices_1d

struct SparseIndexMap{D,A,B} <: AbstractIndexMap{D}
  global_2_local::A
  sparsity::B
  function SparseIndexMap(
    global_2_local::A,
    sparsity::B
    ) where {D,A<:AbstractIndexMap{D},B<:TProductSparsityPattern}
    new{D,A,B}(global_2_local,sparsity)
  end
end

Base.size(i::SparseIndexMap) = size(i.global_2_local)
Base.getindex(i::SparseIndexMap,j...) = getindex(i.global_2_local,j...)
get_global_2_local_map(i::SparseIndexMap) = i.global_2_local
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

function tensorize_indices(i::Int,dofs::Vector{Int})
  D = length(dofs)
  cdofs = cumprod(dofs)
  ic = ()
  @inbounds for d = 1:D-1
    ic = (ic...,fast_index(i,cdofs[d]))
  end
  ic = (ic...,slow_index(i,cdofs[D-1]))
  return CartesianIndex(ic)
end

function tensorize_indices(indices::AbstractVector,dofs::AbstractVector{Int})
  D = length(dofs)
  tindices = Vector{CartesianIndex{D}}(undef,length(indices))
  @inbounds for (ii,i) in enumerate(indices)
    tindices[ii] = tensorize_indices(i,dofs)
  end
  return tindices
end

function split_row_col_indices(i::CartesianIndex{D},dofs::AbstractMatrix{Int}) where D
  @check size(dofs) == (2,D)
  nrows = view(dofs,:,1)
  irc = ()
  @inbounds for d = 1:D
    irc = (irc...,fast_index(i.I[d],nrows[d]),slow_index(i.I[d],nrows[d]))
  end
  return CartesianIndex(irc)
end

function split_row_col_indices(indices::AbstractVector{CartesianIndex{D}},dofs::AbstractMatrix{Int}) where D
  rcindices = Vector{CartesianIndex{2*D}}(undef,length(indices))
  @inbounds for (ii,i) in enumerate(indices)
    rcindices[ii] = split_row_col_indices(i,dofs)
  end
  return rcindices
end
