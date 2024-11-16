"""
    abstract type AbstractTransientSnapshots{T,N,D,I,R<:TransientRealization,A}
      <: AbstractSnapshots{T,N,D,I,R,A} end

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
julia> i = DofMap(collect(LinearIndices((ns1,ns2))))
2×2 DofMap{2, Int64}:
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
2×2×1×2 TransientGenericSnapshots{Float64, 4, 2, 2, DofMap{2, Int64},
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
abstract type AbstractTransientSnapshots{T,N,D,I,R<:TransientRealization,A} <: AbstractSnapshots{T,N,D,I,R,A} end

ParamDataStructures.num_times(s::AbstractTransientSnapshots) = num_times(get_realization(s))

Base.size(s::AbstractTransientSnapshots) = (num_space_dofs(s)...,num_times(s),num_params(s))

"""
    struct TransientGenericSnapshots{T,N,D,I,R,A} <: AbstractTransientSnapshots{T,N,D,I,R,A} end

Most standard implementation of a AbstractTransientSnapshots

"""
struct TransientGenericSnapshots{T,N,D,I,R,A} <: AbstractTransientSnapshots{T,N,D,I,R,A}
  data::A
  dof_map::I
  realization::R
  function TransientGenericSnapshots(
    data::A,
    dof_map::I,
    realization::R
    ) where {T,N,D,R,A<:AbstractParamArray{T,N},I<:AbstractDofMap{D}}
    new{T,D+2,D,I,R,A}(data,dof_map,realization)
  end
end

function RBSteady.Snapshots(s::AbstractParamArray,i::AbstractDofMap,r::TransientRealization)
  TransientGenericSnapshots(s,i,r)
end

ParamDataStructures.get_all_data(s::TransientGenericSnapshots) = s.data
Utils.get_values(s::TransientGenericSnapshots) = s.data
DofMaps.get_dof_map(s::TransientGenericSnapshots) = s.dof_map
RBSteady.get_realization(s::TransientGenericSnapshots) = s.realization

function RBSteady.get_indexed_values(s::TransientGenericSnapshots{T}) where T
  # vi = vectorize(get_dof_map(s))
  # idata = view(data,vi,:)
  data = get_all_data(s.data)
  i = get_dof_map(s)
  idata = zeros(T,size(data))
  for (j,ij) in enumerate(i)
    for k in 1:num_params(s)*num_times(s)
      if ij > 0
        @inbounds idata[ij,k] = data[j,k]
      end
    end
  end
  ConsecutiveParamArray(idata)
end

Base.@propagate_inbounds function Base.getindex(
  s::TransientGenericSnapshots{T,N},
  i::Vararg{Integer,N}
  ) where {T,N}

  @boundscheck checkbounds(s,i...)
  ispace...,itime,iparam = i
  ispace′ = s.dof_map[ispace...]
  data = get_all_data(s.data)
  ispace′ == 0 ? zero(eltype(s)) : data[ispace′,iparam+(itime-1)*num_params(s)]
end

Base.@propagate_inbounds function Base.setindex!(
  s::TransientGenericSnapshots{T,N},
  v,
  i::Vararg{Integer,N}
  ) where {T,N}

  @boundscheck checkbounds(s,i...)
  ispace...,itime,iparam = i
  ispace′ = s.dof_map[ispace...]
  data = get_all_data(s.data)
  ispace′ != 0 && (data[ispace′,iparam+(itime-1)*num_params(s)] = v)
end

function RBSteady.Snapshots(s::Vector{<:AbstractParamArray},i::AbstractDofMap,r::TransientRealization)
  item = testitem(first(s))
  sflat = zeros(length(item),num_params(r)*num_times(r))
  @inbounds for it in 1:num_times(r)
    dit = get_all_data(s[it])
    for ip in 1:num_params(r)
      itp = (it-1)*num_params(r)+ip
      for is in eachindex(item)
        v = dit[is,ip]
        sflat[is,itp] = v
      end
    end
  end
  Snapshots(ConsecutiveParamArray(sflat),i,r)
end

"""
    struct TransientSnapshotsAtIndices{T,N,D,I,R,A<:AbstractTransientSnapshots{T,N,D,I,R},B,C
      } <: AbstractTransientSnapshots{T,N,D,I,R,A}

Represents a AbstractTransientSnapshots `snaps` whose parametric and temporal ranges
are restricted to the indices in `prange` and `trange`. This type essentially acts
as a view for suptypes of AbstractTransientSnapshots, at every space location, on
a selected number of parameter/time indices. An instance of TransientSnapshotsAtIndices
is created by calling the function [`select_snapshots`](@ref)

"""
struct TransientSnapshotsAtIndices{T,N,D,I,R,A<:AbstractTransientSnapshots{T,N,D,I,R},B,C} <: AbstractTransientSnapshots{T,N,D,I,R,A}
  snaps::A
  trange::B
  prange::C
  function TransientSnapshotsAtIndices(snaps::A,trange::B,prange::C) where {T,N,D,I,R,A<:AbstractTransientSnapshots{T,N,D,I,R},B,C}
    @assert 1 <= minimum(trange) <= maximum(trange) <= _num_all_times(snaps)
    @assert 1 <= minimum(prange) <= maximum(prange) <= _num_all_params(snaps)
    new{T,N,D,I,R,A,B,C}(snaps,trange,prange)
  end
end

function TransientSnapshotsAtIndices(s::TransientSnapshotsAtIndices,trange,prange)
  old_trange,old_prange = s.trange,s.prange
  @check intersect(old_trange,trange) == trange
  @check intersect(old_prange,prange) == prange
  TransientSnapshotsAtIndices(s.snaps,trange,prange)
end

function RBSteady.Snapshots(s::TransientSnapshotsAtIndices,i::AbstractDofMap,r::TransientRealization)
  snaps = Snapshots(s.snaps,i,r)
  TransientSnapshotsAtIndices(snaps,s.trange,s.prange)
end

time_indices(s::TransientSnapshotsAtIndices) = s.trange
ParamDataStructures.num_times(s::TransientSnapshotsAtIndices) = length(time_indices(s))
RBSteady.param_indices(s::TransientSnapshotsAtIndices) = s.prange
ParamDataStructures.num_params(s::TransientSnapshotsAtIndices) = length(RBSteady.param_indices(s))

_num_all_params(s::AbstractSnapshots) = num_params(s)
_num_all_params(s::TransientSnapshotsAtIndices) = _num_all_params(s.snaps)
_num_all_times(s::AbstractSnapshots) = num_times(s)
_num_all_times(s::TransientSnapshotsAtIndices) = _num_all_times(s.snaps)

ParamDataStructures.get_all_data(s::TransientSnapshotsAtIndices) = get_all_data(s.snaps)
DofMaps.get_dof_map(s::TransientSnapshotsAtIndices) = get_dof_map(s.snaps)
RBSteady.get_realization(s::TransientSnapshotsAtIndices) = get_realization(s.snaps)[s.prange,s.trange]

function Utils.get_values(s::TransientSnapshotsAtIndices)
  prange = RBSteady.param_indices(s)
  trange = time_indices(s)
  np = _num_all_params(s)
  ptrange = range_1d(prange,trange,np)
  data = get_all_data(get_all_data(s))
  v = view(data,:,ptrange)
  isa(s,SparseSnapshots) ? recast(v,s) : ConsecutiveParamArray(v)
end

function RBSteady.get_indexed_values(s::TransientSnapshotsAtIndices{T}) where T
  # vi = vectorize(get_dof_map(s))
  # data = get_all_data(get_all_data(s))
  # v = view(data,vi,ptrange)
  prange = RBSteady.param_indices(s)
  trange = time_indices(s)
  np = _num_all_params(s)
  ptrange = range_1d(prange,trange,np)
  data = get_all_data(get_all_data(s))
  i = get_dof_map(s)
  idata = zeros(T,size(data))
  for (j,ij) in enumerate(i)
    for k in ptrange
      if ij > 0
        @inbounds idata[ij,k] = data[j,k]
      end
    end
  end
  ConsecutiveParamArray(idata)
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
  sbasic = Snapshots(data,get_dof_map(s),get_realization(s))
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

struct TransientReshapedSnapshots{T,N,N′,D,I,R,A<:AbstractTransientSnapshots{T,N′,D,I,R},B} <: AbstractTransientSnapshots{T,N,D,I,R,A}
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
DofMaps.get_dof_map(s::TransientReshapedSnapshots) = get_dof_map(s.snaps)

function Utils.get_values(s::TransientReshapedSnapshots)
  v = get_values(s.snaps)
  reshape(v.data,s.size)
end

function RBSteady.get_indexed_values(s::TransientReshapedSnapshots)
  v = get_indexed_values(s.snaps)
  vr = reshape(v.data,s.size)
  ConsecutiveParamArray(vr)
end

const TransientSparseSnapshots{T,N,D,I,R} = Union{
  TransientGenericSnapshots{T,N,D,I,R,<:ParamSparseMatrix},
  TransientSnapshotsAtIndices{T,N,D,I,R,<:ParamSparseMatrix}
}

"""
    const UnfoldingTransientSnapshots{T,R,A}
      = AbstractTransientSnapshots{T,3,1,<:AbstractTrivialDofMap,R,A}

"""
const UnfoldingTransientSnapshots{T,I<:AbstractTrivialDofMap,R,A} = AbstractTransientSnapshots{T,3,1,I,R,A}

abstract type ModeAxes end
struct Mode1Axes <: ModeAxes end
struct Mode2Axes <: ModeAxes end

change_mode(::Mode1Axes) = Mode2Axes()
change_mode(::Mode2Axes) = Mode1Axes()

"""
    struct ModeTransientSnapshots{M<:ModeAxes,T,I,R,A<:UnfoldingTransientSnapshots{T,I,R}}
      <: AbstractTransientSnapshots{T,2,1,I,R,A}

Represents a AbstractTransientSnapshots with a TrivialDofMap indexing strategy
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
struct ModeTransientSnapshots{M<:ModeAxes,T,I,R,A<:UnfoldingTransientSnapshots{T,I,R}} <: AbstractTransientSnapshots{T,2,1,I,R,A}
  snaps::A
  mode::M
end

function ModeTransientSnapshots(s::AbstractTransientSnapshots)
  ModeTransientSnapshots(s,get_mode(s))
end

function RBSteady.flatten_snapshots(s::AbstractTransientSnapshots)
  i′ = TrivialDofMap(get_dof_map(s))
  s′ = Snapshots(get_all_data(s),i′,get_realization(s))
  ModeTransientSnapshots(s′)
end

ParamDataStructures.get_all_data(s::ModeTransientSnapshots) = get_all_data(s.snaps)
RBSteady.num_space_dofs(s::ModeTransientSnapshots) = prod(num_space_dofs(s.snaps))
Utils.get_values(s::ModeTransientSnapshots) = get_values(s.snaps)
RBSteady.get_realization(s::ModeTransientSnapshots) = get_realization(s.snaps)
DofMaps.get_dof_map(s::ModeTransientSnapshots) = get_dof_map(s.snaps)

change_mode(s::UnfoldingTransientSnapshots) = ModeTransientSnapshots(s,change_mode(get_mode(s)))
change_mode(s::ModeTransientSnapshots) = ModeTransientSnapshots(s.snaps,change_mode(get_mode(s)))

get_mode(s::UnfoldingTransientSnapshots) = Mode1Axes()
get_mode(s::ModeTransientSnapshots) = s.mode

const Mode1TransientSnapshots{T,I,R,A} = ModeTransientSnapshots{Mode1Axes,T,I,R,A}
const Mode2TransientSnapshots{T,I,R,A} = ModeTransientSnapshots{Mode2Axes,T,I,R,A}

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
  data::AbstractVector{<:BlockParamArray},
  i::AbstractArray{<:AbstractDofMap},
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

function RBSteady.Snapshots(
  a::TupOfArrayContribution,
  i::TupOfArrayContribution,
  r::AbstractRealization)

  map((a,i)->Snapshots(a,i,r),a,i)
end

function RBSteady.select_snapshots(a::TupOfArrayContribution,args...;kwargs...)
  map(a->select_snapshots(a,args...;kwargs...),a)
end
