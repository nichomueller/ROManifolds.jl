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

function vectorize_index_map(i::AbstractIndexMap)
  vi = vec(collect(LinearIndices(size(i))))
  IndexMap(vi)
end

function fix_dof_index_map(i::AbstractIndexMap,dof_to_fix::Int)
  FixedDofIndexMap(i,dof_to_fix)
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

abstract type AbstractMultiValueIndexMap{D,Ti} <: AbstractIndexMap{D,Ti} end

Base.view(i::AbstractMultiValueIndexMap,locations) = MultiValueIndexMapView(i,locations)

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

function _compose_indices(index::AbstractArray{Ti,D},ncomp::Integer) where {Ti,D}
  indices = zeros(Ti,size(index)...,ncomp)
  @inbounds for comp = 1:ncomp
    selectdim(indices,D+1,comp) .= (index.-1).*ncomp .+ comp
  end
  return indices
end

function _stack_indices(vec_indices::AbstractVector{<:AbstractArray})
  sizes = map(size,vec_indices)
  @check all(sizes .== [first(sizes)])
  stack(vec_indices)
end

function _to_scalar_values!(indices::AbstractArray,D::Integer,d::Integer)
  indices .= (indices .- d) ./ D .+ 1
end

struct MultiValueIndexMap{D,Ti,I} <: AbstractMultiValueIndexMap{D,Ti}
  indices::I
  function MultiValueIndexMap(indices::I) where {D,Ti,I<:AbstractArray{Ti,D}}
    new{D,Ti,I}(indices)
  end
end

function MultiValueIndexMap(indices::AbstractArray{<:Integer},ncomp::Integer)
  MultiValueIndexMap(_compose_indices(indices,ncomp))
end

function MultiValueIndexMap(indices::AbstractVector{<:AbstractArray})
  MultiValueIndexMap(_stack_indices(indices))
end

Base.size(i::MultiValueIndexMap) = size(i.indices)
Base.getindex(i::MultiValueIndexMap,j...) = getindex(i.indices,j...)
TensorValues.num_components(i::MultiValueIndexMap{D}) where D = size(i,D)

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
