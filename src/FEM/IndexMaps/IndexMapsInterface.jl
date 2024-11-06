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
- [`TProductIndexMap`](@ref)
- [`SparseIndexMap`](@ref)

"""
abstract type AbstractIndexMap{D,Ti} <: AbstractArray{Ti,D} end

Base.view(i::AbstractIndexMap,locations) = IndexMapView(i,locations)

"""
    remove_constrained_dofs(i::AbstractIndexMap) -> AbstractIndexMap

Removes the negative indices from the index map, which correspond to Dirichlet
boundary conditions.

"""
function remove_constrained_dofs(i::AbstractIndexMap)
  free_dofs_locations = findall(i.>zero(eltype(i)))
  s = _shape_per_dir(free_dofs_locations)
  free_dofs_locations′ = reshape(free_dofs_locations,s)
  view(i,free_dofs_locations′)
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

vectorize_map(i::AbstractIndexMap) = vec(i)

function permute_sparsity(a::SparsityPatternCSC,i::AbstractIndexMap,j::AbstractIndexMap)
  permute_sparsity(a,vectorize_map(i),vectorize_map(j))
end

sum_maps(i::AbstractIndexMap...) = first(i)

function sum_maps(i::Array{I,N}...) where {I<:AbstractIndexMap,N}
  i1 = first(i)
  s = size(i1)
  @check all(size(ii)==s for ii in i)
  i′ = Array{I,N}(undef,s)
  for j in eachindex(i1)
    ij = map(ii -> getindex(ii,j),i)
    i′[j] = sum_maps(ij...)
  end
  i′
end

function sum_maps(i::Tuple{Vararg{I}}...) where {I<:Union{AbstractIndexMap,AbstractArray{<:AbstractIndexMap}}}
  i′ = ()
  for ii in i
    i′ = (i′...,ii...)
  end
  sum_maps(i′...)
end

abstract type AbstractTrivialIndexMap <: AbstractIndexMap{1,Int} end

Base.getindex(i::AbstractTrivialIndexMap,j::Integer) = j
Base.setindex!(i::AbstractTrivialIndexMap,v::Integer,j::Integer) = nothing
Base.copy(i::AbstractTrivialIndexMap) = i
Base.collect(i::AbstractTrivialIndexMap) = i

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
Base.size(i::TrivialIndexMap) = (i.length,)

struct TrivialSparseIndexMap{A<:SparsityPattern} <: AbstractTrivialIndexMap
  sparsity::A
end

TrivialIndexMap(sparsity::SparsityPattern) = TrivialSparseIndexMap(sparsity)
TrivialIndexMap(i::TrivialSparseIndexMap) = i
Base.size(i::TrivialSparseIndexMap) = (nnz(i.sparsity),)

recast(a::AbstractArray,i::TrivialSparseIndexMap) = recast(a,i.sparsity)

get_sparsity(i::TrivialSparseIndexMap) = i.sparsity

function sum_maps(i::TrivialSparseIndexMap...)
  sparsity = sum_sparsities(map(get_sparsity,i)...)
  TrivialSparseIndexMap(sparsity)
end

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

abstract type ShowSlaveDofsStyle end
struct ShowSlaveDofs <: ShowSlaveDofsStyle end
struct DoNotShowSlaveDofs <: ShowSlaveDofsStyle end

struct ConstrainedDofsIndexMap{D,Ti,I<:AbstractIndexMap{D,Ti},A<:ShowSlaveDofsStyle} <: AbstractIndexMap{D,Ti}
  indices::I
  mloc_to_loc::Vector{CartesianIndex{D}}
  sloc_to_loc::Vector{CartesianIndex{D}}
  show_slaves::A
end

function ConstrainedDofsIndexMap(
  indices::AbstractIndexMap,
  mloc_to_loc::Vector{<:CartesianIndex},
  sloc_to_loc::Vector{<:CartesianIndex})

  ConstrainedDofsIndexMap(indices,mloc_to_loc,sloc_to_loc,ShowSlaveDofs())
end

function ConstrainedDofsIndexMap(
  indices::AbstractIndexMap{D},
  mdof_to_bdof::AbstractVector{<:Integer},
  sdof_to_bdof::AbstractVector{<:Integer},
  args...
  ) where D

  dof_layout = CartesianIndices(indices)
  mloc_to_loc = Vector{CartesianIndex{D}}(undef,length(mdof_to_bdof))
  sloc_to_loc = Vector{CartesianIndex{D}}(undef,length(sdof_to_bdof))
  mcount = 1
  scount = 1
  for (i,ii) in enumerate(eachindex(indices))
    ind = indices[i]
    if ind ∈ mdof_to_bdof
      mloc_to_loc[mcount] = ii
      mcount += 1
    elseif ind ∈ sdof_to_bdof
      sloc_to_loc[scount] = ii
      scount += 1
    end
  end
  ConstrainedDofsIndexMap(indices,mloc_to_loc,sloc_to_loc,args...)
end

Base.size(i::ConstrainedDofsIndexMap) = size(i.indices)
Base.getindex(i::ConstrainedDofsIndexMap{D},j::Vararg{Integer,D}) where D = getindex(i.indices,j...)
Base.setindex!(i::ConstrainedDofsIndexMap{D},v,j::Vararg{Integer,D}) where D = setindex!(i.indices,v,j...)
Base.copy(i::ConstrainedDofsIndexMap) = ConstrainedDofsIndexMap(copy(i.indices),i.mloc_to_loc,i.sloc_to_loc,i.show_slaves)
vectorize_map(i::ConstrainedDofsIndexMap) = collect(Base.OneTo(maximum(i)))

function get_masters(i::ConstrainedDofsIndexMap{D,Ti}) where {D,Ti}
  n = length(i.mloc_to_loc)
  m = Vector{Ti}(undef,n)
  for (ij,j) in enumerate(i.mloc_to_loc)
    m[ij] = i.indices[j]
  end
  return m
end

function get_slaves(i::ConstrainedDofsIndexMap{D,Ti,ShowSlaveDofs}) where {D,Ti}
  n = length(i.sloc_to_loc)
  s = Vector{Ti}(undef,n)
  for (ij,j) in enumerate(i.sloc_to_loc)
    s[ij] = i.indices[j]
  end
  return s
end

function get_background(i::ConstrainedDofsIndexMap{D,Ti,ShowSlaveDofs}) where {D,Ti}
  n = length(i) - length(i.mloc_to_loc) - length(i.sloc_to_loc)
  fill(zero(Ti),n)
end

function get_background(i::ConstrainedDofsIndexMap{D,Ti,DoNotShowSlaveDofs}) where {D,Ti}
  n = length(i) - length(i.mloc_to_loc)
  fill(zero(Ti),n)
end

function remove_constrained_dofs(i::ConstrainedDofsIndexMap)
  indices = copy(i.indices)
  z = zero(eltype(indices))
  for j in i.sloc_to_loc
    indices[j] = z
  end
  nslaves = 0
  for j in eachindex(indices)
    if j in i.sloc_to_loc
      nslaves += 1
    elseif j in i.mloc_to_loc
      mj = indices[j]
      indices[j] = mj - nslaves
    end
  end
  ConstrainedDofsIndexMap(indices,i.mloc_to_loc,i.sloc_to_loc,DoNotShowSlaveDofs())
end

function inv_index_map(i::ConstrainedDofsIndexMap)
  @notimplemented
end

function inv_index_map(i::ConstrainedDofsIndexMap{D,Ti,DoNotShowSlaveDofs}) where {D,Ti}
  i′ = copy(i)
  indices = i′.indices
  m = get_masters(i)
  invm = invperm(m)
  for (ij,j) in enumerate(i.mloc_to_loc)
    indices[j] = invm[ij]
  end
  return i′
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

function merge_components(i::AbstractArray{<:AbstractArray{Ti,D}}) where {Ti,D}
  @check all(size(j) == size(first(i)) for j in i)
  indices = stack(i;dims=D+1)
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

function MultiValueIndexMap(indices::AbstractArray{<:AbstractArray})
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

function sum_maps(i::SparseIndexMap...)
  function _sum_maps(inds...)
    ind′ = first(inds)
    for ind in inds
      @check size(ind) == size(ind′)
      for (i,(indi′,indi)) in enumerate(zip(ind′,ind))
        if !iszero(indi)
          if iszero(indi′)
            ind′[i] = indi
          else
            @check indi′ == indi
          end
        end
      end
    end
    return inds′
  end
  indices = _sum_maps(map(get_index_map,i)...)
  indices_sparse = _sum_maps(map(get_indices_sparse,i)...)
  sparsity = sum_sparsities(map(get_sparsity,i)...)
  SparseIndexMap(indices,indices_sparse,sparsity)
end

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
