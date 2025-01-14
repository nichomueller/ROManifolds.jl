"""
    abstract type TransientSnapshots{T,N,D,I,R<:TransientRealization,A} <: Snapshots{T,N,D,I,R,A} end

Transient specialization of a `Snapshots`. The dimension `N` of a
SteadySnapshots is equal to `D` + 2, where `D` represents the number of
spatial axes, to which a temporal and a parametric dimension are added.

Subtypes:
- `TransientGenericSnapshots`
- `GenericSnapshots`
- `TransientSnapshotsAtIndices`
- `TransientReshapedSnapshots`
- `TransientSnapshotsWithIC`
- `ModeTransientSnapshots`
"""
abstract type TransientSnapshots{T,N,D,I,R<:TransientRealization,A} <: Snapshots{T,N,D,I,R,A} end

ParamDataStructures.num_times(s::TransientSnapshots) = num_times(get_realization(s))

Base.size(s::TransientSnapshots) = (num_space_dofs(s)...,num_times(s),num_params(s))

get_initial_values(s::TransientSnapshots) = @abstractmethod

"""
    struct TransientGenericSnapshots{T,N,D,I,R,A} <: TransientSnapshots{T,N,D,I,R,A}
      data::A
      dof_map::I
      realization::R
    end

Most standard implementation of a TransientSnapshots
"""
struct TransientGenericSnapshots{T,N,D,I,R,A} <: TransientSnapshots{T,N,D,I,R,A}
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

ParamDataStructures.get_all_data(s::TransientGenericSnapshots) = get_all_data(s.data)
Utils.get_values(s::TransientGenericSnapshots) = s.data
DofMaps.get_dof_map(s::TransientGenericSnapshots) = s.dof_map
RBSteady.get_realization(s::TransientGenericSnapshots) = s.realization

function RBSteady.get_indexed_data(s::TransientGenericSnapshots{T}) where T
  vi = vectorize(get_dof_map(s))
  data = get_all_data(s)
  if isnothing(findfirst(iszero,vi))
    return view(data,vi,:)
  end
  i = get_dof_map(s)
  idata = zeros(T,size(data))
  for (j,ij) in enumerate(i)
    for k in 1:num_params(s)*num_times(s)
      if ij > 0
        @inbounds idata[ij,k] = data[j,k]
      end
    end
  end
  return idata
end

Base.@propagate_inbounds function Base.getindex(
  s::TransientGenericSnapshots{T,N},
  i::Vararg{Integer,N}
  ) where {T,N}

  @boundscheck checkbounds(s,i...)
  ispace...,itime,iparam = i
  ispace′ = s.dof_map[ispace...]
  data = get_all_data(s)
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
  data = get_all_data(s)
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
    struct TransientSnapshotsAtIndices{T,N,D,I,R,A<:TransientSnapshots{T,N,D,I,R},B,C} <: TransientSnapshots{T,N,D,I,R,A}
      snaps::A
      trange::B
      prange::C
    end

Represents a TransientSnapshots `snaps` whose parametric and temporal ranges
are restricted to the indices in `prange` and `trange`. This type essentially acts
as a view for suptypes of TransientSnapshots, at every space location, on
a selected number of parameter/time indices. An instance of TransientSnapshotsAtIndices
is created by calling the function `select_snapshots`
"""
struct TransientSnapshotsAtIndices{T,N,D,I,R,A<:TransientSnapshots{T,N,D,I,R},B,C} <: TransientSnapshots{T,N,D,I,R,A}
  snaps::A
  trange::B
  prange::C
  function TransientSnapshotsAtIndices(snaps::A,trange::B,prange::C) where {T,N,D,I,R,A<:TransientSnapshots{T,N,D,I,R},B,C}
    @assert 1 <= minimum(trange) <= maximum(trange) <= _num_all_times(snaps)
    @assert 1 <= minimum(prange) <= maximum(prange) <= _num_all_params(snaps)
    new{T,N,D,I,R,A,B,C}(snaps,trange,prange)
  end
end

function TransientSnapshotsAtIndices(s::TransientSnapshotsAtIndices,trange,prange)
  trange′ = s.trange[trange]
  prange′ = s.prange[prange]
  TransientSnapshotsAtIndices(s.snaps,trange′,prange′)
end

function RBSteady.Snapshots(s::TransientSnapshotsAtIndices,i::AbstractDofMap,r::TransientRealization)
  snaps = Snapshots(s.snaps,i,r)
  TransientSnapshotsAtIndices(snaps,s.trange,s.prange)
end

time_indices(s::TransientSnapshotsAtIndices) = s.trange
ParamDataStructures.num_times(s::TransientSnapshotsAtIndices) = length(time_indices(s))
RBSteady.param_indices(s::TransientSnapshotsAtIndices) = s.prange
ParamDataStructures.num_params(s::TransientSnapshotsAtIndices) = length(RBSteady.param_indices(s))

_num_all_params(s::Snapshots) = num_params(s)
_num_all_params(s::TransientSnapshotsAtIndices) = _num_all_params(s.snaps)
_num_all_times(s::Snapshots) = num_times(s)
_num_all_times(s::TransientSnapshotsAtIndices) = _num_all_times(s.snaps)

ParamDataStructures.get_all_data(s::TransientSnapshotsAtIndices) = get_all_data(s.snaps)
DofMaps.get_dof_map(s::TransientSnapshotsAtIndices) = get_dof_map(s.snaps)
RBSteady.get_realization(s::TransientSnapshotsAtIndices) = get_realization(s.snaps)[s.prange,s.trange]

function Utils.get_values(s::TransientSnapshotsAtIndices)
  prange = RBSteady.param_indices(s)
  trange = time_indices(s)
  np = _num_all_params(s)
  ptrange = range_1d(prange,trange,np)
  data = get_all_data(s)
  v = view(data,:,ptrange)
  isa(s,SparseSnapshots) ? recast(v,s) : ConsecutiveParamArray(v)
end

function RBSteady.get_indexed_data(s::TransientSnapshotsAtIndices{T}) where T
  prange = RBSteady.param_indices(s)
  trange = time_indices(s)
  np = _num_all_params(s)
  ptrange = range_1d(prange,trange,np)
  idata = get_indexed_data(s.snaps)
  view(idata,:,ptrange)
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

function RBSteady.select_snapshots(s::TransientSnapshots,trange,prange)
  trange = RBSteady.format_range(trange,num_times(s))
  prange = RBSteady.format_range(prange,num_params(s))
  TransientSnapshotsAtIndices(s,trange,prange)
end

function RBSteady.select_snapshots(s::TransientSnapshots,prange;trange=1:num_times(s))
  select_snapshots(s,trange,prange)
end

"""
    struct TransientReshapedSnapshots{T,N,N′,D,I,R,A<:TransientSnapshots{T,N′,D,I,R},B} <: TransientSnapshots{T,N,D,I,R,A}
      snaps::A
      size::NTuple{N,Int}
      mi::B
    end

Represents a TransientSnapshots `snaps` whose size is resized to `size`. This struct
is equivalent to `ReshapedArray`, and is only used to make sure the result
of this operation is still a subtype of TransientSnapshots
"""
struct TransientReshapedSnapshots{T,N,N′,D,I,R,A<:TransientSnapshots{T,N′,D,I,R},B} <: TransientSnapshots{T,N,D,I,R,A}
  snaps::A
  size::NTuple{N,Int}
  mi::B
end

Base.size(s::TransientReshapedSnapshots) = s.size

function Base.reshape(s::TransientSnapshots,dims::Dims)
  n = length(s)
  prod(dims) == n || DimensionMismatch()

  strds = Base.front(Base.size_to_strides(map(length,axes(s))...,1))
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

function RBSteady.get_indexed_data(s::TransientReshapedSnapshots)
  v = get_indexed_data(s.snaps)
  vr = reshape(v.data,s.size)
  ConsecutiveParamArray(vr)
end

"""
    struct TransientSnapshotsWithIC{T,N,D,I,R,A,B<:TransientSnapshots{T,N,D,I,R,A}} <: TransientSnapshots{T,N,D,I,R,A}
      initial_data::A
      snaps::B
    end

Stores a TransientSnapshots `snaps` alongside a parametric initial condition `initial_data`
"""
struct TransientSnapshotsWithIC{T,N,D,I,R,A,B<:TransientSnapshots{T,N,D,I,R}} <: TransientSnapshots{T,N,D,I,R,A}
  initial_data::A
  snaps::B
end

function RBSteady.Snapshots(s,s0::AbstractParamArray,i::AbstractDofMap,r::TransientRealization)
  snaps = Snapshots(s,i,r)
  TransientSnapshotsWithIC(s0,snaps)
end

ParamDataStructures.get_all_data(s::TransientSnapshotsWithIC) = get_all_data(s.snaps)
get_initial_values(s::TransientSnapshotsWithIC) = s.initial_data
Utils.get_values(s::TransientSnapshotsWithIC) = get_values(s.snaps)
DofMaps.get_dof_map(s::TransientSnapshotsWithIC) = get_dof_map(s.snaps)
RBSteady.get_realization(s::TransientSnapshotsWithIC) = get_realization(s.snaps)
RBSteady.get_indexed_data(s::TransientSnapshotsWithIC) = get_indexed_data(s.snaps)

Base.@propagate_inbounds function Base.getindex(
  s::TransientSnapshotsWithIC{T,N},
  i::Vararg{Integer,N}
  ) where {T,N}

  getindex(s.snaps,i...)
end

Base.@propagate_inbounds function Base.setindex!(
  s::TransientSnapshotsWithIC{T,N},
  v,
  i::Vararg{Integer,N}
  ) where {T,N}

  setindex!(s.snaps,v,i...)
end

function TransientSnapshotsAtIndices(s::TransientSnapshotsWithIC,trange,prange)
  snaps′ = TransientSnapshotsAtIndices(s.snaps,trange,prange)
  initial_data′ = ConsecutiveParamArray(view(get_all_data(s.initial_data),:,prange))
  TransientSnapshotsWithIC(initial_data′,snaps′)
end

const TransientSparseSnapshots{T,N,D,I,R} = Union{
  TransientGenericSnapshots{T,N,D,I,R,<:ParamSparseMatrix},
  TransientSnapshotsAtIndices{T,N,D,I,R,<:ParamSparseMatrix}
}

const UnfoldingTransientSnapshots{T,I<:AbstractTrivialDofMap,R,A} = TransientSnapshots{T,3,1,I,R,A}

abstract type ModeAxes end
struct Mode1Axes <: ModeAxes end
struct Mode2Axes <: ModeAxes end

change_mode(::Mode1Axes) = Mode2Axes()
change_mode(::Mode2Axes) = Mode1Axes()

"""
    struct ModeTransientSnapshots{M<:ModeAxes,T,I,R,A<:UnfoldingTransientSnapshots{T,I,R}} <: TransientSnapshots{T,2,1,I,R,A}
      snaps::A
      mode::M
    end

Represents a TransientSnapshots with a TrivialDofMap indexing strategy
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
struct ModeTransientSnapshots{M<:ModeAxes,T,I,R,A<:UnfoldingTransientSnapshots{T,I,R}} <: TransientSnapshots{T,2,1,I,R,A}
  snaps::A
  mode::M
end

function ModeTransientSnapshots(s::TransientSnapshots)
  ModeTransientSnapshots(s,get_mode(s))
end

function RBSteady.flatten_snapshots(s::TransientSnapshots)
  i′ = TrivialDofMap(get_dof_map(s))
  s′ = Snapshots(get_values(s),i′,get_realization(s))
  ModeTransientSnapshots(s′)
end

ParamDataStructures.get_all_data(s::ModeTransientSnapshots) = get_all_data(s.snaps)
RBSteady.num_space_dofs(s::ModeTransientSnapshots) = prod(num_space_dofs(s.snaps))
Utils.get_values(s::ModeTransientSnapshots) = get_values(s.snaps)
RBSteady.get_realization(s::ModeTransientSnapshots) = get_realization(s.snaps)
DofMaps.get_dof_map(s::ModeTransientSnapshots) = get_dof_map(s.snaps)

"""
    change_mode(s::ModeTransientSnapshots) -> ModeTransientSnapshots

Returns the snapshots obtained by opposing the mode of `s`. The result is a
subtype of AbstractMatrix with entries equal to those of `s`, but with swapped
spatial and temporal axes
"""
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

RBSteady.get_indexed_data(s::Mode1TransientSnapshots) = get_indexed_data(s.snaps)
RBSteady.get_indexed_data(s::Mode2TransientSnapshots) = collect(s)

# block snapshots

function RBSteady.Snapshots(
  data::AbstractVector{<:BlockParamArray{T,N}},
  i::AbstractArray{<:AbstractDofMap},
  r::AbstractRealization) where {T,N}

  block_values = blocks.(data)
  block_value1 = first(block_values)
  s = size(block_value1)
  @check s == size(i)

  array = Array{Snapshots,N}(undef,s)
  touched = Array{Bool,N}(undef,s)
  for (j,data1j) in enumerate(block_value1)
    if !iszero(data1j)
      dataj = map(v->getindex(v,j),block_values)
      array[j] = Snapshots(dataj,i[j],r)
      touched[j] = true
    else
      touched[j] = false
    end
  end

  BlockSnapshots(array,touched)
end

function RBSteady.Snapshots(
  data::AbstractVector{<:BlockParamArray{T,N}},
  data0::BlockParamArray{T,N},
  i::AbstractArray{<:AbstractDofMap},
  r::AbstractRealization) where {T,N}

  block_values = blocks.(data)
  block_value0 = blocks(data0)
  block_value1 = first(block_values)
  s = size(block_value0)
  @check s == size(i)

  array = Array{Snapshots,N}(undef,s)
  touched = Array{Bool,N}(undef,s)
  for (j,data1j) in enumerate(block_value1)
    if !iszero(data1j)
      dataj = map(v->getindex(v,j),block_values)
      array[j] = Snapshots(dataj,block_value0[j],i[j],r)
      touched[j] = true
    else
      touched[j] = false
    end
  end

  BlockSnapshots(array,touched)
end

function Arrays.return_cache(::typeof(get_initial_values),s::BlockSnapshots{S,N}) where {S,N}
  cache = get_initial_values(testitem(s))
  block_cache = Array{typeof(cache),N}(undef,size(s))
  return block_cache
end

function get_initial_values(s::BlockSnapshots)
  values = return_cache(get_initial_values,s)
  for i in eachindex(s.touched)
    if s.touched[i]
      values[i] = get_initial_values(s[i])
    end
  end
  return mortar(values)
end

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
