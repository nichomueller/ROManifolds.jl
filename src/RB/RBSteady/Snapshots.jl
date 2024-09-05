"""
    abstract type AbstractSnapshots{T,N,L,D,I<:AbstractIndexMap{D},R<:AbstractParamRealization}
      <: AbstractParamContainer{T,N,L} end

Type representing a collection of parametric abstract arrays of eltype T and
parametric length L, that are associated with a realization of type R. The (spatial)
entries of any instance of AbstractSnapshots are indexed according to an index
map of type I<:AbstractIndexMap{D}, where D encodes the spatial dimension.

Subtypes:
- [`AbstractSteadySnapshots`](@ref)
- [`AbstractTransientSnapshots`](@ref)

"""
abstract type AbstractSnapshots{T,N,L,D,I<:AbstractIndexMap{D},R<:AbstractParamRealization} <: AbstractParamContainer{T,N,L} end

ParamDataStructures.get_values(s::AbstractSnapshots) = @abstractmethod
get_indexed_values(s::AbstractSnapshots) = @abstractmethod
IndexMaps.get_index_map(s::AbstractSnapshots) = @abstractmethod
get_realization(s::AbstractSnapshots) = @abstractmethod

"""
    num_space_dofs(s::AbstractSnapshots{T,N,L,D}) where {T,N,L,D} -> NTuple{D,Integer}

Returns the spatial size of the snapshots

"""
num_space_dofs(s::AbstractSnapshots) = size(get_index_map(s))
ParamDataStructures.num_params(s::AbstractSnapshots) = num_params(get_realization(s))

"""
    Snapshots(s::AbstractArray,i::AbstractIndexMap,r::AbstractParamRealization
      ) -> AbstractSnapshots

Constructor of an instance of AbstractSnapshots

"""
function Snapshots(s::AbstractArray,i::AbstractIndexMap,r::AbstractParamRealization)
  @abstractmethod
end

for I in (:AbstractIndexMap,:(AbstractArray{<:AbstractIndexMap}))
  @eval begin
    function Snapshots(a::ArrayContribution,i::$I,r::AbstractParamRealization)
      contribution(a.trians) do trian
        Snapshots(a[trian],i,r)
      end
    end

    function Snapshots(a::TupOfArrayContribution,i::$I,r::AbstractParamRealization)
      map(a->Snapshots(a,i,r),a)
    end
  end
end

function IndexMaps.change_index_map(f,s::AbstractSnapshots)
  index_map′ = change_index_map(f,get_index_map(s))
  Snapshots(param_data(s),index_map′,get_realization(s))
end

function IndexMaps.recast(s::AbstractSnapshots,a::AbstractArray)
  i = get_index_map(s)
  a′ = recast(i,a)
  return a′
end

"""
    flatten_snapshots(s::AbstractSnapshots) -> AbstractSnapshots

The output snapshots are indexed according to a [`TrivialIndexMap`](@ref)

"""
function flatten_snapshots(s::AbstractSnapshots)
  change_index_map(TrivialIndexMap,s)
end

"""
    abstract type AbstractSteadySnapshots{T,N,L,D,I,R<:ParamRealization}
      <: AbstractSnapshots{T,N,L,D,I,R} end

Spatial specialization of an [`AbstractSnapshots`](@ref). The dimension `N` of a
AbstractSteadySnapshots is equal to `D` + 1, where `D` represents the number of
spatial axes, to which a parametric dimension is added.

Subtypes:
- [`GenericSnapshots`](@ref).
- [`SnapshotsAtIndices`](@ref).

# Examples

```jldoctest
julia> ns1,ns2,np = 2,2,2
(2, 2, 2)
julia> data = [rand(ns1*ns2) for ip = 1:np]
2-element Vector{Vector{Float64}}:
 [0.4684452123483283, 0.1195886171030737, 0.1151790990455997, 0.0375575515915656]
 [0.9095165124078269, 0.7346081836882059, 0.8939511550403715, 0.2288086807377305]
julia> i = IndexMap(collect(LinearIndices((ns1,ns2))))
2×2 IndexMap{2, Int64}:
 1  3
 2  4
julia> pspace = ParamSpace(fill([0,1],3))
Set of parameters in [[0, 1], [0, 1], [0, 1]], sampled with UniformSampling()
julia> r = realization(pspace,nparams=np)
ParamRealization{Vector{Vector{Float64}}}([
  [0.4021870679335007, 0.6585653527784044, 0.5110768420820191],
  [0.0950901750101361, 0.7049711670440882, 0.3490097863258958]])
julia> s = Snapshots(ParamArray(data),i,r)
2×2×2 GenericSnapshots{Float64, 3, 2, 2, IndexMap{2, Int64},
  ParamRealization{Vector{Vector{Float64}}}, VectorOfVectors{Float64, 2}}:
  [:, :, 1] =
  0.468445  0.115179
  0.119589  0.0375576

  [:, :, 2] =
  0.909517  0.893951
  0.734608  0.228809
```

"""
abstract type AbstractSteadySnapshots{T,N,L,D,I,R<:ParamRealization} <: AbstractSnapshots{T,N,L,D,I,R} end

Base.size(s::AbstractSteadySnapshots) = (num_space_dofs(s)...,num_params(s))

"""
    struct GenericSnapshots{T,N,L,D,I,R,A} <: AbstractSteadySnapshots{T,N,L,D,I,R} end

Most standard implementation of a AbstractSteadySnapshots

"""
struct GenericSnapshots{T,N,L,D,I,R,A} <: AbstractSteadySnapshots{T,N,L,D,I,R}
  data::A
  index_map::I
  realization::R

  function GenericSnapshots(
    data::A,
    index_map::I,
    realization::R
    ) where {T,N,L,D,R,A<:AbstractParamArray{T,N,L},I<:AbstractIndexMap{D}}

    new{T,D+1,L,D,I,R,A}(data,index_map,realization)
  end
end

function Snapshots(s::AbstractParamArray,i::AbstractIndexMap,r::ParamRealization)
  GenericSnapshots(s,i,r)
end

ParamDataStructures.param_data(s::GenericSnapshots) = s.data
ParamDataStructures.get_values(s::GenericSnapshots) = s.data
IndexMaps.get_index_map(s::GenericSnapshots) = s.index_map
get_realization(s::GenericSnapshots) = s.realization

function get_indexed_values(s::GenericSnapshots)
  vi = vec(get_index_map(s))
  v = consecutive_getindex(s.data,vi,:)
  ConsecutiveArrayOfArrays(v)
end

Base.@propagate_inbounds function Base.getindex(
  s::GenericSnapshots{T,N},
  i::Vararg{Integer,N}
  ) where {T,N}

  @boundscheck checkbounds(s,i...)
  ispace...,iparam = i
  ispace′ = s.index_map[ispace...]
  ispace′ == 0 ? zero(eltype(s)) : consecutive_getindex(s.data,ispace′,iparam)
end

Base.@propagate_inbounds function Base.setindex!(
  s::GenericSnapshots{T,N},
  v,
  i::Vararg{Integer,N}
  ) where {T,N}

  @boundscheck checkbounds(s,i...)
  ispace...,iparam = i
  ispace′ = s.index_map[ispace...]
  ispace′ != 0 && consecutive_setindex!(s.data,v,ispace′,iparam)
end

"""
    struct SnapshotsAtIndices{T,N,L,D,I,R,A<:AbstractSteadySnapshots{T,N,L,D,I,R},B<:AbstractUnitRange{Int}}
      <: AbstractSteadySnapshots{T,N,L,D,I,R} end

Represents a AbstractSteadySnapshots `snaps` whose parametric range is restricted
to the indices in `prange`. This type essentially acts as a view for suptypes of
AbstractSteadySnapshots, at every space location, on a selected number of
parameter indices. An instance of SnapshotsAtIndices is created by calling the
function [`select_snapshots`](@ref)

"""
struct SnapshotsAtIndices{T,N,L,D,I,R,A<:AbstractSteadySnapshots{T,N,L,D,I,R},B<:AbstractUnitRange{Int}} <: AbstractSteadySnapshots{T,N,L,D,I,R}
  snaps::A
  prange::B
end

function SnapshotsAtIndices(s::SnapshotsAtIndices,prange)
  old_prange = s.prange
  @check intersect(old_prange,prange) == prange
  SnapshotsAtIndices(s.snaps,prange)
end

param_indices(s::SnapshotsAtIndices) = s.prange
ParamDataStructures.num_params(s::SnapshotsAtIndices) = length(param_indices(s))
ParamDataStructures.param_data(s::SnapshotsAtIndices) = param_data(s.snaps)

function ParamDataStructures.get_values(s::SnapshotsAtIndices)
  v = consecutive_getindex(s.snaps.data,:,param_indices(s))
  ConsecutiveArrayOfArrays(v)
end

function get_indexed_values(s::SnapshotsAtIndices)
  vi = vec(get_index_map(s))
  v = consecutive_getindex(s.snaps.data,vi,param_indices(s))
  ConsecutiveArrayOfArrays(v)
end

get_realization(s::SnapshotsAtIndices) = get_realization(s.snaps)[s.prange]

Base.@propagate_inbounds function Base.getindex(
  s::SnapshotsAtIndices{T,N},
  i::Vararg{Integer,N}
  ) where {T,N}

  @boundscheck checkbounds(s,i...)
  ispace...,iparam = i
  iparam′ = getindex(param_indices(s),iparam)
  getindex(s.snaps,ispace...,iparam′)
end

Base.@propagate_inbounds function Base.setindex!(
  s::SnapshotsAtIndices{T,N},
  v,
  i::Vararg{Integer,N}
  ) where {T,N}

  @boundscheck checkbounds(s,i...)
  ispace...,iparam
  iparam′ = getindex(param_indices(s),iparam)
  setindex!(s.snaps,v,ispace...,iparam′)
end

format_range(a::AbstractUnitRange,l::Int) = a
format_range(a::Base.OneTo{Int},l::Int) = 1:a.stop
format_range(a::Number,l::Int) = a:a
format_range(a::Colon,l::Int) = 1:l

"""
    select_snapshots(s::AbstractSteadySnapshots,prange) -> SnapshotsAtIndices
    select_snapshots(s::AbstractTransientSnapshots,trange,prange) -> TransientSnapshotsAtIndices

Restricts the parametric range of `s` to the indices `prange` steady cases, to
the indices `trange` and `prange` in transient cases, while leaving the spatial
entries intact. The restriction operation is lazy.

"""
function select_snapshots(s::AbstractSteadySnapshots,prange)
  prange = format_range(prange,num_params(s))
  SnapshotsAtIndices(s,prange)
end

struct ReshapedSnapshots{T,N,N′,L,D,I,R,A<:AbstractSteadySnapshots{T,N′,L,D,I,R},B} <: AbstractSteadySnapshots{T,N′,L,D,I,R}
  snaps::A
  size::NTuple{N,Int}
  mi::B
end

Base.size(s::ReshapedSnapshots) = s.size

function Base.reshape(s::AbstractSnapshots,dims::Dims)
  n = length(s)
  prod(dims) == n || DimensionMismatch()

  strds = Base.front(Base.size_to_strides(map(length,axes(s))..., 1))
  strds1 = map(s->max(1,Int(s)),strds)
  mi = map(Base.SignedMultiplicativeInverse,strds1)
  ReshapedSnapshots(parent,dims,reverse(mi))
end

Base.@propagate_inbounds function Base.getindex(
  s::ReshapedSnapshots{T,N},
  i::Vararg{Integer,N}
  ) where {T,N}

  @boundscheck checkbounds(s,i...)
  ax = axes(s.snaps)
  i′ = Base.offset_if_vec(Base._sub2ind(size(s),i...),ax)
  i′′ = Base.ind2sub_rs(ax,s.mi,i′)
  Base._unsafe_getindex_rs(s.snaps,i′′)
end

function Base.setindex!(
  s::ReshapedSnapshots{T,N},
  v,i::Vararg{Integer,N}
  ) where {T,N}

  @boundscheck checkbounds(s,i...)
  ax = axes(s.snaps)
  i′ = Base.offset_if_vec(Base._sub2ind(size(s),i...),ax)
  s.snaps[Base.ind2sub_rs(ax,s.mi,i′)] = v
  v
end

get_realization(s::ReshapedSnapshots) = get_realization(s.snaps)
IndexMaps.get_index_map(s::ReshapedSnapshots) = get_index_map(s.snaps)

function ParamDataStructures.get_values(s::ReshapedSnapshots)
  v = get_values(s.snaps)
  reshape(v.data,s.size)
end

function get_indexed_values(s::ReshapedSnapshots)
  v = get_indexed_values(s.snaps)
  vr = reshape(v.data,s.size)
  ConsecutiveArrayOfArrays(vr)
end

const StandardSparseSnapshots{T,N,L,D,I,R,A<:MatrixOfSparseMatricesCSC} = GenericSnapshots{T,N,L,D,I,R,A}
const SparseSnapshots{T,N,L,D,I,R,A<:MatrixOfSparseMatricesCSC} = Union{
  StandardSparseSnapshots{T,N,L,D,I,R,A},
  SnapshotsAtIndices{T,N,L,D,I,R,StandardSparseSnapshots{T,N,L,D,I,R,A}}
}

"""
    select_snapshots_entries(s::AbstractSteadySnapshots,srange) -> ArrayOfArrays
    select_snapshots_entries(s::AbstractTransientSnapshots,srange,trange) -> ArrayOfArrays

Selects the snapshots' entries corresponding to the spatial range `srange` in
steady cases, to the spatial-temporal ranges `srange` and `trange` in transient
cases, for every parameter.

"""
function select_snapshots_entries(s::AbstractSteadySnapshots,srange)
  _getindex(s::AbstractSteadySnapshots,is,it,ip) = consecutive_getindex(s.data,is,ip)
  _getindex(s::SparseSnapshots,is,it,ip) = param_getindex(s.data,ip)[is]

  T = eltype(s)
  nval = length(srange)
  np = num_params(s)
  entries = array_of_consecutive_arrays(zeros(T,nval),np)

  for ip = 1:np
    for (i,is) in enumerate(srange)
      v = _getindex(s.data,is,ip)
      consecutive_setindex!(entries,v,i,ip)
    end
  end

  return entries
end

const UnfoldingSteadySnapshots{T,L,I<:TrivialIndexMap,R} = AbstractSteadySnapshots{T,2,L,1,I,R}

function IndexMaps.recast(s::UnfoldingSteadySnapshots,a::AbstractMatrix)
  return recast(s.data,a)
end

function Base.:*(A::AbstractSnapshots{T,2},B::AbstractSnapshots{S,2}) where {T,S}
  consecutive_mul(get_indexed_values(A),get_indexed_values(B))
end

function Base.:*(A::AbstractSnapshots{T,2},B::Adjoint{S,<:AbstractSnapshots}) where {T,S}
  consecutive_mul(get_indexed_values(A),adjoint(get_indexed_values(B.parent)))
end

function Base.:*(A::AbstractSnapshots{T,2},B::AbstractMatrix{S}) where {T,S}
  consecutive_mul(get_indexed_values(A),B)
end

function Base.:*(A::AbstractSnapshots{T,2},B::Adjoint{T,<:AbstractMatrix{S}}) where {T,S}
  consecutive_mul(get_indexed_values(A),B)
end

function Base.:*(A::Adjoint{T,<:AbstractSnapshots{T,2}},B::AbstractSnapshots{S,2}) where {T,S}
  consecutive_mul(adjoint(get_indexed_values(A.parent)),get_indexed_values(B))
end

function Base.:*(A::AbstractMatrix{T},B::AbstractSnapshots{S,2}) where {T,S}
  consecutive_mul(A,get_indexed_values(B))
end

function Base.:*(A::Adjoint{T,<:AbstractMatrix},B::AbstractSnapshots{S,2}) where {T,S}
  consecutive_mul(A,get_indexed_values(B))
end

"""
    struct BlockSnapshots{S,N,L} <: AbstractParamContainer{S,N,L}

Block container for AbstractSnapshots of type `S` in a MultiField setting. This
type is conceived similarly to [`ArrayBlock`](@ref) in [`Gridap`](@ref)

"""
struct BlockSnapshots{S,N,L} <: AbstractParamContainer{S,N,L}
  array::Array{S,N}
  touched::Array{Bool,N}

  function BlockSnapshots(array::Array{S,N},touched::Array{Bool,N}) where {S,N}
    @check size(array) == size(touched)
    L = param_length(first(array))
    new{S,N,L}(array,touched)
  end
end

function BlockSnapshots(k::BlockMap{N},a::AbstractArray{S}) where {S,N}
  array = Array{S,N}(undef,k.size)
  touched = fill(false,k.size)
  for (t,i) in enumerate(k.indices)
    array[i] = a[t]
    touched[i] = true
  end
  BlockSnapshots(array,touched)
end

function Fields.BlockMap(s::NTuple,inds::AbstractVector{<:Integer})
  cis = [CartesianIndex((i,)) for i in inds]
  BlockMap(s,cis)
end

function Snapshots(data::BlockArrayOfArrays,i::AbstractArray{<:AbstractIndexMap},r::AbstractParamRealization)
  block_values = blocks(data)
  nblocks = blocksize(data)
  active_block_ids = findall(!iszero,block_values)
  block_map = BlockMap(nblocks,active_block_ids)
  active_block_snaps = [Snapshots(block_values[n],i[n],r) for n in active_block_ids]
  BlockSnapshots(block_map,active_block_snaps)
end

Base.size(s::BlockSnapshots) = size(s.array)

function Base.getindex(s::BlockSnapshots,i...)
  if !s.touched[i...]
    return nothing
  end
  s.array[i...]
end

function Base.setindex!(s::BlockSnapshots,v,i...)
  @check s.touched[i...] "Only touched entries can be set"
  s.array[i...] = v
end

function Arrays.testitem(s::BlockSnapshots)
  i = findall(s.touched)
  if length(i) != 0
    s.array[i[1]]
  else
    error("This block snapshots structure is empty")
  end
end

"""
    get_touched_blocks(a::AbstractArray{T,N}) where {T,N} -> Array{Bool,N}

Returns the indices corresponding to the touched entries of a block object

"""
get_touched_blocks(s) = @abstractmethod
get_touched_blocks(s::ArrayBlock) = findall(s.touched)
get_touched_blocks(s::BlockSnapshots) = findall(s.touched)

IndexMaps.get_index_map(s::BlockSnapshots) = get_index_map(testitem(s))
get_realization(s::BlockSnapshots) = get_realization(testitem(s))

function ParamDataStructures.get_values(s::BlockSnapshots)
  map(get_values,s.array) |> mortar
end

function get_indexed_values(s::BlockSnapshots)
  map(get_indexed_values,s.array) |> mortar
end

function IndexMaps.change_index_map(f,s::BlockSnapshots{S,N}) where {S,N}
  active_block_ids = get_touched_blocks(s)
  block_map = BlockMap(size(s),active_block_ids)
  active_block_snaps = [change_index_map(f,s[n]) for n in active_block_ids]
  BlockSnapshots(block_map,active_block_snaps)
end

function flatten_snapshots(s::BlockSnapshots{S,N}) where {S,N}
  active_block_ids = get_touched_blocks(s)
  block_map = BlockMap(size(s),active_block_ids)
  active_block_snaps = [flatten_snapshots(s[n]) for n in active_block_ids]
  BlockSnapshots(block_map,active_block_snaps)
end

function select_snapshots(s::BlockSnapshots{S,N},args...;kwargs...) where {S,N}
  active_block_ids = get_touched_blocks(s)
  block_map = BlockMap(size(s),active_block_ids)
  active_block_snaps = [select_snapshots(s[n],args...;kwargs...) for n in active_block_ids]
  BlockSnapshots(block_map,active_block_snaps)
end

function select_snapshots_entries(s::BlockSnapshots{S,N},srange::ArrayBlock{<:Any,N}) where {S,N}
  active_block_ids = get_touched_blocks(s)
  block_map = BlockMap(size(s),active_block_ids)
  active_block_snaps = [select_snapshots_entries(s[n],srange[n]) for n in active_block_ids]
  return_cache(block_map,active_block_snaps...)
end

# utils

function select_snapshots(a::ArrayContribution,args...;kwargs...)
  contribution(a.trians) do trian
    select_snapshots(a[trian],args...;kwargs...)
  end
end

function select_snapshots(a::TupOfArrayContribution,args...;kwargs...)
  map(a->select_snapshots(a,args...;kwargs...),a)
end
