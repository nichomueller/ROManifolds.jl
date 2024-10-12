"""
    abstract type AbstractTransientSnapshots{T,N,L,D,I,R<:TransientRealization,A}
      <: AbstractSnapshots{T,N,L,D,I,R,A} end

Transient specialization of an [`AbstractSnapshots`](@ref). The dimension `N` of a
AbstractSteadySnapshots is equal to `D` + 2, where `D` represents the number of
spatial axes, to which a temporal and a parametric dimension are added.

Subtypes:
- [`TransientGenericSnapshots`](@ref)
- [`GenericSnapshots`](@ref)
- [`TransientSnapshotsAtIndices`](@ref)
- [`ModeTransientSnapshots`](@ref)

# Examples

```jldoctest
julia> ns1,ns2,nt,np = 2,2,1,2
(2, 2, 1, 2)
julia> data = [rand(ns1*ns2) for ip = 1:np*nt]
2-element Vector{Vector{Float64}}:
 [0.4684452123483283, 0.1195886171030737, 0.1151790990455997, 0.0375575515915656]
 [0.9095165124078269, 0.7346081836882059, 0.8939511550403715, 0.2288086807377305]
julia> i = IndexMap(collect(LinearIndices((ns1,ns2))))
2×2 IndexMap{2, Int64}:
 1  3
 2  4
julia> ptspace = TransientParamSpace(fill([0,1],3))
Set of tuples (p,t) in [[0, 1], [0, 1], [0, 1]] × 0:1
julia> r = realization(ptspace,nparams=np)
GenericTransientRealization{Realization{Vector{Vector{Float64}}},
  Int64, Vector{Int64}}([
  [0.4021870679335007, 0.6585653527784044, 0.5110768420820191],
  [0.0950901750101361, 0.7049711670440882, 0.3490097863258958]],
  [1],
  0)
julia> s = Snapshots(ParamArray(data),i,r)
2×2×1×2 TransientGenericSnapshots{Float64, 4, 2, 2, IndexMap{2, Int64},
  GenericTransientRealization{Realization{Vector{Vector{Float64}}}, Int64, Vector{Int64}},
  VectorOfVectors{Float64, 2}}:
  [:, :, 1, 1] =
  0.468445  0.115179
  0.119589  0.0375576

  [:, :, 1, 2] =
  0.909517  0.893951
  0.734608  0.228809
```

"""
abstract type AbstractTransientSnapshots{T,N,L,D,I,R<:TransientRealization,A} <: AbstractSnapshots{T,N,L,D,I,R,A} end

ParamDataStructures.num_times(s::AbstractTransientSnapshots) = num_times(get_realization(s))

Base.size(s::AbstractTransientSnapshots) = (num_space_dofs(s)...,num_times(s),num_params(s))

"""
    struct TransientGenericSnapshots{T,N,L,D,I,R,A} <: AbstractTransientSnapshots{T,N,L,D,I,R,A} end

Most standard implementation of a AbstractTransientSnapshots

"""
struct TransientGenericSnapshots{T,N,L,D,I,R,A} <: AbstractTransientSnapshots{T,N,L,D,I,R,A}
  data::A
  index_map::I
  realization::R
  function TransientGenericSnapshots(
    data::A,
    index_map::I,
    realization::R
    ) where {T,N,L,D,R,A<:AbstractParamArray{T,N,L},I<:AbstractIndexMap{D}}
    new{T,D+2,L,D,I,R,A}(data,index_map,realization)
  end
end

function RBSteady.Snapshots(s::AbstractParamArray,i::AbstractIndexMap,r::TransientRealization)
  TransientGenericSnapshots(s,i,r)
end

ParamDataStructures.param_data(s::TransientGenericSnapshots) = s.data
ParamDataStructures.get_values(s::TransientGenericSnapshots) = s.data
IndexMaps.get_index_map(s::TransientGenericSnapshots) = s.index_map
RBSteady.get_realization(s::TransientGenericSnapshots) = s.realization

function RBSteady.get_indexed_values(s::TransientGenericSnapshots)
  vi = vec(get_index_map(s))
  v = consecutive_getindex(s.data,vi,:)
  ConsecutiveArrayOfArrays(v)
end

Base.@propagate_inbounds function Base.getindex(
  s::TransientGenericSnapshots{T,N},
  i::Vararg{Integer,N}
  ) where {T,N}

  @boundscheck checkbounds(s,i...)
  ispace...,itime,iparam = i
  ispace′ = s.index_map[ispace...]
  ispace′ == 0 ? zero(eltype(s)) : consecutive_getindex(s.data,ispace′,iparam+(itime-1)*num_params(s))
end

Base.@propagate_inbounds function Base.setindex!(
  s::TransientGenericSnapshots{T,N},
  v,
  i::Vararg{Integer,N}
  ) where {T,N}

  @boundscheck checkbounds(s,i...)
  ispace...,itime,iparam = i
  ispace′ = s.index_map[ispace...]
  ispace′ != 0 && consecutive_setindex!(s.data,v,ispace′,iparam+(itime-1)*num_params(s))
end

function RBSteady.Snapshots(s::Vector{<:AbstractParamArray},i::AbstractIndexMap,r::TransientRealization)
  item = consecutive_getindex(first(s),:,1)
  sflat = array_of_consecutive_arrays(item,num_params(r)*num_times(r))
  @inbounds for it in 1:num_times(r)
    dit = s[it]
    @inbounds for ip in 1:num_params(r)
      itp = (it-1)*num_params(r)+ip
      v = consecutive_getindex(dit,:,ip)
      consecutive_setindex!(sflat,v,:,itp)
    end
  end
  Snapshots(sflat,i,r)
end

"""
    struct TransientSnapshotsAtIndices{T,N,L,D,I,R,A<:AbstractTransientSnapshots{T,N,L,D,I,R},B,C
      } <: AbstractTransientSnapshots{T,N,L,D,I,R,A}

Represents a AbstractTransientSnapshots `snaps` whose parametric and temporal ranges
are restricted to the indices in `prange` and `trange`. This type essentially acts
as a view for suptypes of AbstractTransientSnapshots, at every space location, on
a selected number of parameter/time indices. An instance of TransientSnapshotsAtIndices
is created by calling the function [`select_snapshots`](@ref)

"""
struct TransientSnapshotsAtIndices{T,N,L,D,I,R,A<:AbstractTransientSnapshots{T,N,L,D,I,R},B,C} <: AbstractTransientSnapshots{T,N,L,D,I,R,A}
  snaps::A
  trange::B
  prange::C
end

function TransientSnapshotsAtIndices(s::TransientSnapshotsAtIndices,trange,prange)
  old_trange,old_prange = s.trange,s.prange
  @check intersect(old_trange,trange) == trange
  @check intersect(old_prange,prange) == prange
  TransientSnapshotsAtIndices(s.snaps,trange,prange)
end

function RBSteady.Snapshots(s::TransientSnapshotsAtIndices,i::AbstractIndexMap,r::TransientRealization)
  snaps = Snapshots(s.snaps,i,r)
  TransientSnapshotsAtIndices(snaps,s.trange,s.prange)
end

time_indices(s::TransientSnapshotsAtIndices) = s.trange
ParamDataStructures.num_times(s::TransientSnapshotsAtIndices) = length(time_indices(s))
RBSteady.param_indices(s::TransientSnapshotsAtIndices) = s.prange
ParamDataStructures.num_params(s::TransientSnapshotsAtIndices) = length(RBSteady.param_indices(s))

_num_all_params(s::AbstractSnapshots) = num_params(s)
_num_all_params(s::TransientSnapshotsAtIndices) = _num_all_params(s.snaps)

ParamDataStructures.param_data(s::TransientSnapshotsAtIndices) = param_data(s.snaps)
IndexMaps.get_index_map(s::TransientSnapshotsAtIndices) = get_index_map(s.snaps)
RBSteady.get_realization(s::TransientSnapshotsAtIndices) = get_realization(s.snaps)[s.prange,s.trange]

function ParamDataStructures.get_values(s::TransientSnapshotsAtIndices)
  prange = RBSteady.param_indices(s)
  trange = time_indices(s)
  np = _num_all_params(s)
  ptrange = range_1d(prange,trange,np)
  v = consecutive_getindex(param_data(s),:,ptrange)
  isa(s,SparseSnapshots) ? recast(v,s) : ConsecutiveArrayOfArrays(v)
end

function RBSteady.get_indexed_values(s::TransientSnapshotsAtIndices)
  vi = vec(get_index_map(s))
  prange = RBSteady.param_indices(s)
  trange = time_indices(s)
  np = _num_all_params(s)
  ptrange = range_1d(prange,trange,np)
  v = consecutive_getindex(param_data(s),vi,ptrange)
  ConsecutiveArrayOfArrays(v)
end

Base.@propagate_inbounds function Base.getindex(
  s::TransientSnapshotsAtIndices{T,N},
  i::Vararg{Integer,N}
  ) where {T,N}

  @boundscheck checkbounds(s,i...)
  ispace...,itime,iparam = i
  itime′ = getindex(time_indices(s),itime)
  iparam′ = getindex(RBSteady.param_indices(s),iparam)
  getindex(s.snaps,ispace...,itime′,iparam′)
end

Base.@propagate_inbounds function Base.setindex!(
  s::TransientSnapshotsAtIndices{T,N},
  v,
  i::Vararg{Integer,N}
  ) where {T,N}

  @boundscheck checkbounds(s,i...)
  ispace...,itime,iparam = i
  itime′ = getindex(time_indices(s),itime)
  iparam′ = getindex(RBSteady.param_indices(s),iparam)
  setindex!(s.snaps,v,ispace,itime′,iparam′)
end

function RBSteady.flatten_snapshots(s::TransientSnapshotsAtIndices)
  data = get_values(s)
  sbasic = Snapshots(data,get_index_map(s),get_realization(s))
  flatten_snapshots(sbasic)
end

function RBSteady.select_snapshots(s::AbstractTransientSnapshots,trange,prange)
  trange = RBSteady.format_range(trange,num_times(s))
  prange = RBSteady.format_range(prange,num_params(s))
  TransientSnapshotsAtIndices(s,trange,prange)
end

function RBSteady.select_snapshots(s::AbstractTransientSnapshots,prange;trange=1:num_times(s))
  select_snapshots(s,trange,prange)
end

struct TransientReshapedSnapshots{T,N,N′,L,D,I,R,A<:AbstractTransientSnapshots{T,N′,L,D,I,R},B} <: AbstractTransientSnapshots{T,N,L,D,I,R,A}
  snaps::A
  size::NTuple{N,Int}
  mi::B
end

Base.size(s::TransientReshapedSnapshots) = s.size

function Base.reshape(s::AbstractTransientSnapshots,dims::Dims)
  n = length(s)
  prod(dims) == n || DimensionMismatch()

  strds = Base.front(Base.size_to_strides(map(length,axes(s))..., 1))
  strds1 = map(s->max(1,Int(s)),strds)
  mi = map(Base.SignedMultiplicativeInverse,strds1)
  TransientReshapedSnapshots(s,dims,reverse(mi))
end

function Base.getindex(
  s::TransientReshapedSnapshots{T,N},
  i::Vararg{Integer,N}
  ) where {T,N}

  @boundscheck checkbounds(s,i...)
  ax = axes(s.snaps)
  i′ = Base.offset_if_vec(Base._sub2ind(size(s),i...),ax)
  i′′ = Base.ind2sub_rs(ax,s.mi,i′)
  Base._unsafe_getindex_rs(s.snaps,i′′)
end

function Base.setindex!(
  s::TransientReshapedSnapshots{T,N},
  v,i::Vararg{Integer,N}
  ) where {T,N}

  @boundscheck checkbounds(s,i...)
  ax = axes(s.snaps)
  i′ = Base.offset_if_vec(Base._sub2ind(size(s),i...),ax)
  s.snaps[Base.ind2sub_rs(ax,s.mi,i′)] = v
  v
end

RBSteady.get_realization(s::TransientReshapedSnapshots) = get_realization(s.snaps)
IndexMaps.get_index_map(s::TransientReshapedSnapshots) = get_index_map(s.snaps)

function ParamDataStructures.get_values(s::TransientReshapedSnapshots)
  v = get_values(s.snaps)
  reshape(v.data,s.size)
end

function RBSteady.get_indexed_values(s::TransientReshapedSnapshots)
  v = get_indexed_values(s.snaps)
  vr = reshape(v.data,s.size)
  ConsecutiveArrayOfArrays(vr)
end

const TransientSparseSnapshots{T,N,L,D,I,R} = Union{
  TransientGenericSnapshots{T,N,L,D,I,R,<:ParamSparseMatrix},
  TransientSnapshotsAtIndices{T,N,L,D,I,R,<:ParamSparseMatrix}
}

"""
    const UnfoldingTransientSnapshots{T,L,R,A}
      = AbstractTransientSnapshots{T,3,L,1,<:AbstractTrivialIndexMap,R,A}

"""
const UnfoldingTransientSnapshots{T,L,R,A} = AbstractTransientSnapshots{T,3,L,1,<:AbstractTrivialIndexMap,R,A}

abstract type ModeAxes end
struct Mode1Axes <: ModeAxes end
struct Mode2Axes <: ModeAxes end

change_mode(::Mode1Axes) = Mode2Axes()
change_mode(::Mode2Axes) = Mode1Axes()

"""
    struct ModeTransientSnapshots{M<:ModeAxes,T,L,I,R,A<:UnfoldingTransientSnapshots{T,L,I,R}}
      <: AbstractTransientSnapshots{T,2,L,1,I,R,A}

Represents a AbstractTransientSnapshots with a TrivialIndexMap indexing strategy
as an AbstractMatrix with a system of mode-unfolding representations. Only two
mode-unfolding representations are considered:

Mode1Axes:

[u(x1,t1,μ1) ⋯ u(x1,t1,μP) u(x1,t2,μ1) ⋯ u(x1,t2,μP) u(x1,t3,μ1) ⋯ ⋯ u(x1,tT,μ1) ⋯ u(x1,tT,μP)]
      ⋮             ⋮          ⋮            ⋮           ⋮              ⋮             ⋮
 u(xN,t1,μ1) ⋯ u(xN,t1,μP) u(xN,t2,μ1) ⋯ u(xN,t2,μP) u(xN,t3,μ1) ⋯ ⋯ u(xN,tT,μ1) ⋯ u(xN,tT,μP)]

Mode2Axes:

[u(x1,t1,μ1) ⋯ u(x1,t1,μP) u(x2,t1,μ1) ⋯ u(x2,t1,μP) u(x3,t1,μ1) ⋯ ⋯ u(xN,t1,μ1) ⋯ u(xN,t1,μP)]
      ⋮             ⋮          ⋮            ⋮           ⋮              ⋮             ⋮
 u(x1,tT,μ1) ⋯ u(x1,tT,μP) u(x2,tT,μ1) ⋯ u(x2,tT,μP) u(x3,tT,μ1) ⋯ ⋯ u(xN,tT,μ1) ⋯ u(xN,tT,μP)]

"""
struct ModeTransientSnapshots{M<:ModeAxes,T,L,R,A<:UnfoldingTransientSnapshots{T,L,R}} <: AbstractTransientSnapshots{T,2,L,1,AbstractTrivialIndexMap,R,A}
  snaps::A
  mode::M
end

function ModeTransientSnapshots(s::AbstractTransientSnapshots)
  ModeTransientSnapshots(s,get_mode(s))
end

function RBSteady.flatten_snapshots(s::AbstractTransientSnapshots)
  s′ = change_index_map(TrivialIndexMap,s)
  ModeTransientSnapshots(s′)
end

ParamDataStructures.param_data(s::ModeTransientSnapshots) = param_data(s.snaps)
RBSteady.num_space_dofs(s::ModeTransientSnapshots) = prod(num_space_dofs(s.snaps))
ParamDataStructures.get_values(s::ModeTransientSnapshots) = get_values(s.snaps)
RBSteady.get_realization(s::ModeTransientSnapshots) = get_realization(s.snaps)
IndexMaps.get_index_map(s::ModeTransientSnapshots) = get_index_map(s.snaps)

change_mode(s::UnfoldingTransientSnapshots) = ModeTransientSnapshots(s,change_mode(get_mode(s)))
change_mode(s::ModeTransientSnapshots) = ModeTransientSnapshots(s.snaps,change_mode(get_mode(s)))

get_mode(s::UnfoldingTransientSnapshots) = Mode1Axes()
get_mode(s::ModeTransientSnapshots) = s.mode

const Mode1TransientSnapshots{T,L,R,A} = ModeTransientSnapshots{Mode1Axes,T,L,R,A}
const Mode2TransientSnapshots{T,L,R,A} = ModeTransientSnapshots{Mode2Axes,T,L,R,A}

Base.size(s::Mode1TransientSnapshots) = (num_space_dofs(s),num_times(s)*num_params(s))
Base.size(s::Mode2TransientSnapshots) = (num_times(s),num_space_dofs(s)*num_params(s))

Base.@propagate_inbounds function Base.getindex(s::Mode1TransientSnapshots,ispace::Integer,icol::Integer)
  @boundscheck checkbounds(s,ispace,icol)
  itime = slow_index(icol,num_params(s))
  iparam = fast_index(icol,num_params(s))
  getindex(s.snaps,ispace,itime,iparam)
end

Base.@propagate_inbounds function Base.setindex!(s::Mode1TransientSnapshots,v,ispace::Integer,icol::Integer)
  @boundscheck checkbounds(s,ispace,icol)
  itime = slow_index(icol,num_params(s))
  iparam = fast_index(icol,num_params(s))
  setindex!(s.snaps,v,ispace,itime,iparam)
end

Base.@propagate_inbounds function Base.getindex(s::Mode2TransientSnapshots,itime::Integer,icol::Integer)
  @boundscheck checkbounds(s,itime,icol)
  ispace = slow_index(icol,num_params(s))
  iparam = fast_index(icol,num_params(s))
  getindex(s.snaps,ispace,itime,iparam)
end

Base.@propagate_inbounds function Base.setindex!(s::Mode2TransientSnapshots,v,itime::Integer,icol::Integer)
  @boundscheck checkbounds(s,itime,icol)
  ispace = slow_index(icol,num_params(s))
  iparam = fast_index(icol,num_params(s))
  setindex!(s.snaps,v,ispace,itime,iparam)
end

function change_mode(a::AbstractMatrix,np::Integer)
  n1 = size(a,1)
  n2 = Int(size(a,2)/np)
  a′ = zeros(eltype(a),n2,n1*np)
  @inbounds for i = 1:np
    @views a′[:,(i-1)*n1+1:i*n1] = a[:,i:np:np*n2]'
  end
  return a′
end

RBSteady.get_indexed_values(s::Mode1TransientSnapshots) = get_indexed_values(s.snaps)
RBSteady.get_indexed_values(s::Mode2TransientSnapshots) = collect(s)

# block snapshots

function RBSteady.Snapshots(
  data::AbstractVector{<:BlockArrayOfArrays},
  i::AbstractArray{<:AbstractIndexMap},
  r::AbstractRealization)

  block_values = blocks.(data)
  nblocks = blocksize(first(data))
  active_block_ids = findall(!iszero,blocks(first(data)))
  block_map = BlockMap(nblocks,active_block_ids)
  active_block_snaps = [Snapshots(map(v->getindex(v,n),block_values),i[n],r) for n in active_block_ids]
  BlockSnapshots(block_map,active_block_snaps)
end

# utils

struct Range2D{I<:AbstractVector,J<:AbstractVector} <: AbstractMatrix{Int32}
  axis1::I
  axis2::J
  scale::Int
end

range_2d(i::AbstractVector,j::AbstractVector,nJ=length(j)) = Range2D(i,j,nJ)
range_1d(i::AbstractVector,j::AbstractVector,args...) = vec(range_2d(i,j,args...))

Base.size(r::Range2D) = (length(r.axis1),length(r.axis2))
Base.getindex(r::Range2D,i::Integer,j::Integer) = r.axis1[i] + (r.axis2[j]-1)*r.scale

# utils

for I in (:AbstractIndexMap,:(AbstractArray{<:AbstractIndexMap}))
  @eval begin
    function RBSteady.Snapshots(a::TupOfArrayContribution,i::$I,r::AbstractRealization)
      map(a->Snapshots(a,i,r),a)
    end
  end
end

function RBSteady.select_snapshots(a::TupOfArrayContribution,args...;kwargs...)
  map(a->select_snapshots(a,args...;kwargs...),a)
end
