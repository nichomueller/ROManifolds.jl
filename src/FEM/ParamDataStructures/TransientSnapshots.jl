"""
    abstract type TransientSnapshots{T,N,D,I,R<:TransientRealization,A} <: Snapshots{T,N,D,I,R,A} end

Transient specialization of a `Snapshots`. The dimension `N` of a
SteadySnapshots is equal to `D` + 2, where `D` represents the number of
spatial axes, to which a temporal and a parametric dimension are added.

Subtypes:
- [`TransientGenericSnapshots`](@ref)
- [`GenericSnapshots`](@ref)
- [`TransientSnapshotsAtIndices`](@ref)
- [`TransientSnapshotsWithIC`](@ref)
- [`ModeTransientSnapshots`](@ref)
"""
abstract type TransientSnapshots{T,N,D,I,R<:TransientRealization,A} <: Snapshots{T,N,D,I,R,A} end

num_times(s::TransientSnapshots) = num_times(get_realization(s))

Base.size(s::TransientSnapshots) = (space_dofs(s)...,num_times(s),num_params(s))

get_initial_data(s::TransientSnapshots) = @abstractmethod

"""
    struct TransientGenericSnapshots{T,N,D,I,R,A} <: TransientSnapshots{T,N,D,I,R,A}
      data::A
      dof_map::I
      realization::R
    end

Most standard implementation of a [`TransientSnapshots`](@ref)
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

function Snapshots(s::AbstractParamArray,i::AbstractDofMap,r::TransientRealization)
  TransientGenericSnapshots(s,i,r)
end

get_all_data(s::TransientGenericSnapshots) = get_all_data(s.data)
get_param_data(s::TransientGenericSnapshots) = s.data
DofMaps.get_dof_map(s::TransientGenericSnapshots) = s.dof_map
get_realization(s::TransientGenericSnapshots) = s.realization

function get_indexed_data(s::TransientGenericSnapshots{T}) where T
  i = get_dof_map(s)
  data = get_all_data(s)
  idata = zeros(T,size(data))
  for ipt in 1:num_params(s)*num_times(s)
    for (ij,j) in enumerate(i)
      if j > 0
        @inbounds idata[ij,ipt] = data[j,ipt]
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

"""
    struct TransientSnapshotsAtIndices{T,N,D,I,R,A<:TransientSnapshots{T,N,D,I,R},B,C} <: TransientSnapshots{T,N,D,I,R,A}
      snaps::A
      trange::B
      prange::C
    end

Represents a [`TransientSnapshots`](@ref) `snaps` whose parametric and temporal ranges
are restricted to the indices in `prange` and `trange`. This type essentially acts
as a view for suptypes of `TransientSnapshots`, at every space location, on
a selected number of parameter/time indices. An instance of `TransientSnapshotsAtIndices`
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

function Snapshots(s::TransientSnapshotsAtIndices,i::AbstractDofMap,r::TransientRealization)
  snaps = Snapshots(s.snaps,i,r)
  TransientSnapshotsAtIndices(snaps,s.trange,s.prange)
end

time_indices(s::TransientSnapshotsAtIndices) = s.trange
num_times(s::TransientSnapshotsAtIndices) = length(time_indices(s))
param_indices(s::TransientSnapshotsAtIndices) = s.prange
num_params(s::TransientSnapshotsAtIndices) = length(param_indices(s))

_num_all_times(s::TransientSnapshots) = num_times(s)
_num_all_params(s::TransientSnapshotsAtIndices) = _num_all_params(s.snaps)
_num_all_times(s::TransientSnapshotsAtIndices) = _num_all_times(s.snaps)

get_all_data(s::TransientSnapshotsAtIndices) = get_all_data(s.snaps)
DofMaps.get_dof_map(s::TransientSnapshotsAtIndices) = get_dof_map(s.snaps)
get_realization(s::TransientSnapshotsAtIndices) = get_realization(s.snaps)[s.prange,s.trange]

function get_param_data(s::TransientSnapshotsAtIndices)
  prange = param_indices(s)
  trange = time_indices(s)
  np = _num_all_params(s)
  ptrange = range_1d(prange,trange,np)
  data = get_all_data(s)
  v = view(data,:,ptrange)
  isa(s,SparseSnapshots) ? recast(v,s) : ConsecutiveParamArray(v)
end

function get_indexed_data(s::TransientSnapshotsAtIndices{T}) where T
  i = get_dof_map(s)
  data = get_all_data(s)
  idata = zeros(T,num_space_dofs(s),num_times(s)*num_params(s))
  for (it,t) in enumerate(time_indices(s))
    for (ip,p) in enumerate(param_indices(s))
      ipt = (it-1)*num_params(s)+ip
      pt = (t-1)*_num_all_params(s)+p
      for (ij,j) in enumerate(i)
        if j > 0
          @inbounds idata[ij,ipt] = data[j,pt]
        end
      end
    end
  end
  return idata
end

Base.@propagate_inbounds function Base.getindex(
  s::TransientSnapshotsAtIndices{T,N},
  i::Vararg{Integer,N}
  ) where {T,N}

  @boundscheck checkbounds(s,i...)
  ispace...,itime,iparam = i
  itime′ = getindex(time_indices(s),itime)
  iparam′ = getindex(param_indices(s),iparam)
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
  iparam′ = getindex(param_indices(s),iparam)
  setindex!(s.snaps,v,ispace,itime′,iparam′)
end

function DofMaps.flatten(s::TransientSnapshotsAtIndices)
  data = get_param_data(s)
  sbasic = Snapshots(data,get_dof_map(s),get_realization(s))
  flatten(sbasic)
end

function select_snapshots(s::TransientSnapshots,trange,prange)
  trange = format_range(trange,num_times(s))
  prange = format_range(prange,num_params(s))
  TransientSnapshotsAtIndices(s,trange,prange)
end

function select_snapshots(s::TransientSnapshots,prange;trange=1:num_times(s))
  select_snapshots(s,trange,prange)
end

# trivial dimensions

function select_snapshots(s::TransientSnapshots{T,N},trange::Integer,prange) where {T,N}
  srange = select_snapshots(s,format_range(trange,num_times(s)),prange)
  dropdims(srange;dims=N-1)
end

function select_snapshots(s::TransientSnapshots{T,N},trange,prange::Integer) where {T,N}
  srange = select_snapshots(s,trange,format_range(prange,num_params(s)))
  dropdims(srange;dims=N)
end

function select_snapshots(s::TransientSnapshots{T,N},trange::Integer,prange::Integer) where {T,N}
  srange = select_snapshots(s,format_range(trange,num_times(s)),format_range(prange,num_params(s)))
  dropdims(srange;dims=(N-1,N))
end

"""
    struct TransientSnapshotsWithIC{T,N,D,I,R,A,B<:TransientSnapshots{T,N,D,I,R,A}} <: TransientSnapshots{T,N,D,I,R,A}
      initial_data::A
      snaps::B
    end

Stores a [`TransientSnapshots`](@ref) `snaps` alongside a parametric initial condition `initial_data`
"""
struct TransientSnapshotsWithIC{T,N,D,I,R,A,B<:TransientSnapshots{T,N,D,I,R}} <: TransientSnapshots{T,N,D,I,R,A}
  initial_data::A
  snaps::B
end

function Snapshots(s,s0::AbstractParamArray,i::AbstractDofMap,r::TransientRealization)
  snaps = Snapshots(s,i,r)
  TransientSnapshotsWithIC(s0,snaps)
end

get_all_data(s::TransientSnapshotsWithIC) = get_all_data(s.snaps)
get_initial_data(s::TransientSnapshotsWithIC) = s.initial_data
get_param_data(s::TransientSnapshotsWithIC) = get_param_data(s.snaps)
DofMaps.get_dof_map(s::TransientSnapshotsWithIC) = get_dof_map(s.snaps)
get_realization(s::TransientSnapshotsWithIC) = get_realization(s.snaps)
get_indexed_data(s::TransientSnapshotsWithIC) = get_indexed_data(s.snaps)

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

"""
"""
const TransientSparseSnapshots{T,N,D,I,R} = Union{
  TransientGenericSnapshots{T,N,D,I,R,<:ParamSparseMatrix},
  TransientSnapshotsAtIndices{T,N,D,I,R,<:ParamSparseMatrix}
}

const UnfoldingTransientSnapshots{T,I<:TrivialDofMap,R,A} = TransientSnapshots{T,3,1,I,R,A}

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

Represents a [`TransientSnapshots`](@ref) with a [`TrivialDofMap`](@ref) indexing strategy
as an `AbstractMatrix` with a system of mode-unfolding representations. Only two
mode-unfolding representations are considered:

Mode1Axes:

  ```[u(x1,t1,μ1) ⋯ u(x1,t1,μP) u(x1,t2,μ1) ⋯ u(x1,t2,μP) u(x1,t3,μ1) ⋯ ⋯ u(x1,tT,μ1) ⋯ u(x1,tT,μP)]
        ⋮             ⋮          ⋮            ⋮           ⋮              ⋮             ⋮
  u(xN,t1,μ1) ⋯ u(xN,t1,μP) u(xN,t2,μ1) ⋯ u(xN,t2,μP) u(xN,t3,μ1) ⋯ ⋯ u(xN,tT,μ1) ⋯ u(xN,tT,μP)]```

Mode2Axes:

  ```[u(x1,t1,μ1) ⋯ u(x1,t1,μP) u(x2,t1,μ1) ⋯ u(x2,t1,μP) u(x3,t1,μ1) ⋯ ⋯ u(xN,t1,μ1) ⋯ u(xN,t1,μP)]
        ⋮             ⋮          ⋮            ⋮           ⋮              ⋮             ⋮
  u(x1,tT,μ1) ⋯ u(x1,tT,μP) u(x2,tT,μ1) ⋯ u(x2,tT,μP) u(x3,tT,μ1) ⋯ ⋯ u(xN,tT,μ1) ⋯ u(xN,tT,μP)]```
"""
struct ModeTransientSnapshots{M<:ModeAxes,T,I,R,A<:UnfoldingTransientSnapshots{T,I,R}} <: TransientSnapshots{T,2,1,I,R,A}
  snaps::A
  mode::M
end

function ModeTransientSnapshots(s::TransientSnapshots)
  ModeTransientSnapshots(s,get_mode(s))
end

function DofMaps.flatten(s::TransientSnapshots)
  i′ = flatten(get_dof_map(s))
  s′ = Snapshots(get_param_data(s),i′,get_realization(s))
  ModeTransientSnapshots(s′)
end

get_all_data(s::ModeTransientSnapshots) = get_all_data(s.snaps)
get_param_data(s::ModeTransientSnapshots) = get_param_data(s.snaps)
get_realization(s::ModeTransientSnapshots) = get_realization(s.snaps)
DofMaps.get_dof_map(s::ModeTransientSnapshots) = get_dof_map(s.snaps)

"""
    change_mode(s::ModeTransientSnapshots) -> ModeTransientSnapshots

Returns the snapshots obtained by opposing the mode of `s`. The result is a
subtype of `AbstractMatrix` with entries equal to those of `s`, but with swapped
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

get_indexed_data(s::Mode1TransientSnapshots) = get_indexed_data(s.snaps)
get_indexed_data(s::Mode2TransientSnapshots) = collect(s)

# block snapshots

function Snapshots(
  data::BlockParamArray{T,N},
  data0::BlockParamArray,
  i::AbstractArray{<:AbstractDofMap},
  r::AbstractRealization
  ) where {T,N}

  block_values = blocks(data)
  block_value0 = blocks(data0)
  s = size(block_values)
  @check s == size(i)

  array = Array{Snapshots,N}(undef,s)
  touched = Array{Bool,N}(undef,s)
  for j in 1:length(block_values)
    dataj = block_values[j]
    data0j = block_value0[j]
    if !iszero(dataj)
      array[j] = Snapshots(dataj,data0j,i[j],r)
      touched[j] = true
    else
      touched[j] = false
    end
  end

  BlockSnapshots(array,touched)
end

function Arrays.return_cache(::typeof(get_initial_data),s::BlockSnapshots{S,N}) where {S,N}
  cache = get_initial_data(testitem(s))
  block_cache = Array{typeof(cache),N}(undef,size(s))
  return block_cache
end

function get_initial_data(s::BlockSnapshots)
  values = return_cache(get_initial_data,s)
  for i in eachindex(s.touched)
    if s.touched[i]
      values[i] = get_initial_data(s[i])
    end
  end
  return mortar(values)
end

# utils

function Snapshots(
  a::TupOfArrayContribution,
  i::TupOfArrayContribution,
  r::AbstractRealization)

  map((a,i)->Snapshots(a,i,r),a,i)
end

function select_snapshots(a::TupOfArrayContribution,args...;kwargs...)
  map(a->select_snapshots(a,args...;kwargs...),a)
end
