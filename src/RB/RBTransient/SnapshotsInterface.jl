abstract type AbstractTransientSnapshots{T,N,L,D,I,R<:TransientParamRealization} <: AbstractSnapshots{T,N,L,D,I,R} end

ParamDataStructures.num_times(s::AbstractTransientSnapshots) = num_times(get_realization(s))

Base.size(s::AbstractTransientSnapshots) = (num_space_dofs(s)...,num_times(s),num_params(s))

struct BasicTransientSnapshots{T,N,L,D,I,R,A} <: AbstractTransientSnapshots{T,N,L,D,I,R}
  data::A
  index_map::I
  realization::R
  function BasicTransientSnapshots(
    data::A,
    index_map::I,
    realization::R
    ) where {T,N,L,A<:AbstractParamArray{T,N,L}}
    new{T,N+2,L,I,R,A}(data,index_map,realization)
  end
end

function Snapshots(s::AbstractParamArray,i::AbstractIndexMap,r::TransientParamRealization)
  BasicTransientSnapshots(s,i,r)
end

ParamDataStructures.get_values(s::BasicTransientSnapshots) = s.data
IndexMaps.get_index_map(s::BasicTransientSnapshots) = s.index_map
RBSteady.get_realization(s::BasicTransientSnapshots) = s.realization

# time is the outer variable, param the inner
Base.@propagate_inbounds function Base.getindex(
  s::BasicTransientSnapshots{T,N},
  i::Vararg{Integer,N}
  ) where {T,N}

  @boundscheck checkbounds(s,i...)
  ispace...,itime,iparam = i
  ispace′ = s.index_map[ispace...]
  sparam = param_getindex(s.data,iparam+(itime-1)*num_params(s))
  getindex(sparam,ispace′)
end

Base.@propagate_inbounds function Base.setindex!(
  s::BasicTransientSnapshots{T,N},
  v,
  i::Vararg{Integer,N}
  ) where {T,N}

  @boundscheck checkbounds(s,i...)
  ispace...,iparam = i
  ispace′ = s.index_map[ispace...]
  sparam = param_view(s.data,iparam+(itime-1)*num_params(s))
  setindex!(sparam,v,ispace′)
end

struct TransientSnapshots{T,N,L,D,I,R,A} <: AbstractTransientSnapshots{T,N,L,D,I,R}
  data::Vector{A}
  index_map::I
  realization::R
  function TransientSnapshots(
    data::A,
    index_map::I,
    realization::R
    ) where {T,N,L,A<:Vector{<:AbstractParamArray{T,N,L}}}
    new{T,N+2,L,I,R,A}(data,index_map,realization)
  end
end

function RBSteady.Snapshots(s::Vector{<:AbstractParamArray},i::AbstractIndexMap,r::TransientParamRealization)
  TransientSnapshots(s,i,r)
end

function ParamDataStructures.get_values(s::TransientSnapshots)
  vdata = s.data
  item = all_data(first(vdata))
  T = eltype(item)
  N = ndims(item)
  s...,send = size(item)
  tL = length(vdata)
  s′ = (s...,send*tL)
  data = Array{T,N}(undef,s′)
  @inbounds for i = eachindex(vdata)
    @views data[s...,(i-1)*tL:i*tL] = all_data(vdata[i])
  end
  return ArrayOfArrays(data)
end

IndexMaps.get_index_map(s::TransientSnapshots) = s.index_map
RBSteady.get_realization(s::TransientSnapshots) = s.realization

Base.@propagate_inbounds function Base.getindex(
  s::TransientSnapshots{T,N},
  i::Vararg{Integer,N}
  ) where {T,N}

  @boundscheck checkbounds(s,i...)
  ispace...,iparam = i
  ispace′ = s.index_map[ispace...]
  sparam = param_getindex(s.data[itime],iparam)
  getindex(sparam,ispace′)
end

Base.@propagate_inbounds function Base.setindex!(
  s::BasicTransientSnapshots{T,N},
  v,
  i::Vararg{Integer,N}
  ) where {T,N}

  @boundscheck checkbounds(s,i...)
  ispace...,iparam = i
  ispace′ = s.index_map[ispace...]
  sparam = param_view(s.data[itime],iparam)
  setindex!(sparam,v,ispace′)
end

struct TransientSnapshotsAtIndices{T,N,L,D,I,R,A<:AbstractTransientSnapshots{T,N,L,D,I,R},B<:NTuple{N,AbstractUnitRange{Int}}} <: AbstractTransientSnapshots{T,N,L,D,I,R}
  snaps::A
  indices::B
end

function RBSteady.SnapshotsAtIndices(s::TransientSnapshotsAtIndices,indices)
  new_srange,new_trange,new_prange = indices
  old_srange,old_trange,old_prange = s.indices
  @check intersect(old_srange,new_srange) == new_srange
  @check intersect(old_trange,new_trange) == new_trange
  @check intersect(old_prange,new_prange) == new_prange
  TransientSnapshotsAtIndices(s.snaps,indices)
end

RBSteady.space_indices(s::TransientSnapshotsAtIndices{T,N}) where {T,N} = s.selected_indices[1:N-1]
RBSteady.param_indices(s::TransientSnapshotsAtIndices) = s.selected_indices[N]
RBSteady.num_space_dofs(s::TransientSnapshotsAtIndices) = length.(space_indices(s))

ParamDataStructures.get_values(s::TransientSnapshotsAtIndices) = get_values(s.snaps)
IndexMaps.get_index_map(s::TransientSnapshotsAtIndices) = get_index_map(s.snaps)
RBSteady.get_realization(s::TransientSnapshotsAtIndices) = get_realization(s.snaps)

Base.@propagate_inbounds function Base.getindex(
  s::TransientSnapshotsAtIndices{T,N},
  i::Vararg{Integer,N}
  ) where {T,N}

  @boundscheck checkbounds(s,i...)
  ispace...,itime,iparam = i
  ispace′ = getindex.(space_indices(s),ispace)
  itime′ = getindex(time_indices(s),itime)
  iparam′ = getindex(param_indices(s),iparam)
  getindex(s.snaps,ispace′...,itime′,iparam′)
end

Base.@propagate_inbounds function Base.setindex!(
  s::TransientSnapshotsAtIndices{T,N},
  v,
  i::Vararg{Integer,N}
  ) where {T,N}

  @boundscheck checkbounds(s,i...)
  ispace...,itime,iparam = i
  ispace′ = getindex.(space_indices(s),ispace)
  itime′ = getindex(time_indices(s),itime)
  iparam′ = getindex(param_indices(s),iparam)
  setindex!(s.snaps,v,ispace′,itime′,iparam′)
end

function RBSteady.select_snapshots(
  s::AbstractTransientSnapshots,
  spacerange::Tuple{Vararg{AbstractUnitRange}},
  timerange::AbstractUnitRange,
  paramrange::AbstractUnitRange)

  indices = (spacerange,timerange,paramrange)
  TransientSnapshotsAtIndices(s,indices)
end

function RBSteady.select_snapshots(s::AbstractTransientSnapshots,spacerange,timerange,paramrange)
  srange = Tuple.(format_range(spacerange,num_space_dofs(s)))
  trange = format_range(timerange,num_params(s))
  prange = format_range(paramrange,num_params(s))
  select_snapshots(s,srange,trange,prange)
end

function RBSteady.select_snapshots(
  s::AbstractTransientSnapshots,
  timerange,
  paramrange;
  spacerange=Base.OneTo.(num_space_dofs(s)))

  select_snapshots(s,spacerange,timerange,paramrange)
end

function RBSteady.select_snapshots(
  s::AbstractTransientSnapshots,
  paramrange;
  spacerange=Base.OneTo.(num_space_dofs(s)),
  timerange=Base.OneTo(num_times(s)))

  select_snapshots(s,spacerange,timerange,paramrange)
end

function RBSteady.select_snapshots_entries(s::AbstractTransientSnapshots,spacerange,timerange)
  ss = select_snapshots(s,spacerange,timerange,Base.OneTo(num_params(s)))
  data = collect(ss)
  return ArrayOfArrays(data)
end

const TransientSparseSnapshots{T,N,L,D,I,R,A<:MatrixOfSparseMatricesCSC} = Union{
  TransientBasicSnapshots{T,N,L,D,I,R,A},
  TransientSnapshots{T,N,L,D,I,R,A},
}

function ParamDataStructures.recast(s::TransientSparseSnapshots,a::AbstractMatrix)
  A = param_getindex(s.data,1)
  return recast(A,a)
end

const StandardTransientSnapshots{T,3,L,I,R} = AbstractTransientSnapshots{T,N,L,1,I,R}
const StandardTransientSparseSnapshots{T,L,I,R,A<:MatrixOfSparseMatricesCSC} = TransientSparseSnapshots{T,3,L,1,I,R,A}

abstract type ModeAxes end
struct Mode1Axes <: ModeAxes end
struct Mode2Axes <: ModeAxes end

change_mode(::Mode1Axes) = Mode2Axes()
change_mode(::Mode2Axes) = Mode1Axes()

struct ModeTransientSnapshots{M<:ModeAxes,T,L,I,R,A<:StandardTransientSnapshots{T,L,I,R}} <: StandardTransientSnapshots{T,L,I,R}
  snaps::A
  mode::M
end

function ModeTransientSnapshots(s::AbstractTransientSnapshots)
  ModeTransientSnapshots(s,Mode1Axes())
end

function RBSteady.flatten_snapshots(s::AbstractTransientSnapshots)
  s′ = change_index_map(TrivialIndexMap,s)
  ModeTransientSnapshots(s′)
end

change_mode(s::StandardTransientSnapshots) = ModeTransientSnapshots(s,Mode2Axes())
change_mode(s::StandardTransientSnapshots) = ModeTransientSnapshots(s,change_mode(get_mode(s)))

get_mode(s::ModeTransientSnapshots) = s.mode

Base.@propagate_inbounds function Base.getindex(s::ModeTransientSnapshots,irow,icol)
  @boundscheck checkbounds(s,irow,icol)
  islow = slow_index(icol,num_params(s))
  ifast = fast_index(icol,num_params(s))
  getindex(s.snaps,irow,islow,ifast)
end

Base.@propagate_inbounds function Base.setindex!(s::ModeTransientSnapshots,v,irow,icol)
  @boundscheck checkbounds(s,irow,icol)
  islow = slow_index(icol,num_params(s))
  ifast = fast_index(icol,num_params(s))
  setindex!(s.snaps,v,irow,islow,ifast)
end

const Mode1TransientSnapshots{T,L,I,R,A} = ModeTransientSnapshots{Mode1Axes,T,L,I,R,A}
const Mode2TransientSnapshots{T,L,I,R,A} = ModeTransientSnapshots{Mode2Axes,T,L,I,R,A}

Base.size(s::Mode1TransientSnapshots) = (num_space_dofs(s),num_times(s)*num_params(s))
Base.size(s::Mode2TransientSnapshots) = (num_times(s),num_space_dofs(s)*num_params(s))

# compression operation

_compress(s,a,X::AbstractMatrix) = a'*X*s
_compress(s,a,args...) = a'*s

function compress(s::Mode1TransientSnapshots,a::AbstractMatrix,args...;change_mode=true)
  s′ = _compress(s,a,args...)
  if change_mode
    s′ = change_mode(s′,num_params(s))
  end
  return s′
end

function change_mode(a::AbstractMatrix,np::Integer)
  n1 = size(a,1)
  n2 = Int(size(a,2)/np)
  a′ = zeros(eltype(a),n2,n1*np)
  @inbounds for i = 1:np
    @views a′[:,(i-1)*n1+1:i*n1] = a[:,(i-1)*n2+1:i*n2]'
  end
  return a′
end
