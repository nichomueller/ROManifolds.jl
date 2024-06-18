abstract type AbstractTransientSnapshots{T,N,L,D,I,R<:TransientParamRealization} <: AbstractSnapshots{T,N,L,D,I,R} end

ParamDataStructures.num_times(s::AbstractTransientSnapshots) = num_times(get_realization(s))

Base.size(s::AbstractTransientSnapshots) = (num_space_dofs(s)...,num_times(s),num_params(s))

struct TransientBasicSnapshots{T,N,L,D,I,R,A} <: AbstractTransientSnapshots{T,N,L,D,I,R}
  data::A
  index_map::I
  realization::R
  function TransientBasicSnapshots(
    data::A,
    index_map::I,
    realization::R
    ) where {T,N,L,D,R,A<:AbstractParamArray{T,N,L},I<:AbstractIndexMap{D}}
    new{T,D+2,L,D,I,R,A}(data,index_map,realization)
  end
end

function RBSteady.Snapshots(s::AbstractParamArray,i::AbstractIndexMap,r::TransientParamRealization)
  TransientBasicSnapshots(s,i,r)
end

function RBSteady.Snapshots(s::TransientBasicSnapshots,i::AbstractIndexMap,r::TransientParamRealization)
  TransientBasicSnapshots(s.data,i,r)
end

ParamDataStructures.get_values(s::TransientBasicSnapshots) = s.data
IndexMaps.get_index_map(s::TransientBasicSnapshots) = s.index_map
RBSteady.get_realization(s::TransientBasicSnapshots) = s.realization

# time is the outer variable, param the inner
Base.@propagate_inbounds function Base.getindex(
  s::TransientBasicSnapshots{T,N},
  i::Vararg{Integer,N}
  ) where {T,N}

  @boundscheck checkbounds(s,i...)
  ispace...,itime,iparam = i
  ispace′ = s.index_map[ispace...]
  sparam = param_getindex(s.data,iparam+(itime-1)*num_params(s))
  getindex(sparam,ispace′)
end

Base.@propagate_inbounds function Base.setindex!(
  s::TransientBasicSnapshots{T,N},
  v,
  i::Vararg{Integer,N}
  ) where {T,N}

  @boundscheck checkbounds(s,i...)
  ispace...,itime,iparam = i
  ispace′ = s.index_map[ispace...]
  sparam = param_getindex(s.data,iparam+(itime-1)*num_params(s))
  setindex!(sparam,v,ispace′)
end

struct TransientSnapshots{T,N,L,D,I,R,A} <: AbstractTransientSnapshots{T,N,L,D,I,R}
  data::Vector{A}
  index_map::I
  realization::R
  function TransientSnapshots(
    data::Vector{A},
    index_map::I,
    realization::R
    ) where {T,N,L,D,R,A<:AbstractParamArray{T,N,L},I<:AbstractIndexMap{D}}
    new{T,D+2,L,D,I,R,A}(data,index_map,realization)
  end
end

function RBSteady.Snapshots(s::Vector{<:AbstractParamArray},i::AbstractIndexMap,r::TransientParamRealization)
  TransientSnapshots(s,i,r)
end

function RBSteady.Snapshots(s::TransientSnapshots,i::AbstractIndexMap,r::TransientParamRealization)
  TransientSnapshots(s.data,i,r)
end

function ParamDataStructures.get_values(s::TransientSnapshots)
  item = testitem(first(s.data).data)
  data = array_of_similar_arrays(item,num_params(s)*num_times(s))
  @inbounds for (ipt,index) in enumerate(CartesianIndices((num_params(s),num_times(s))))
    ip,it = index.I
    @views data[ipt] = s.data[it][ip]
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
  ispace...,itime,iparam = i
  ispace′ = s.index_map[ispace...]
  sparam = param_getindex(s.data[itime],iparam)
  getindex(sparam,ispace′)
end

Base.@propagate_inbounds function Base.setindex!(
  s::TransientSnapshots{T,N},
  v,
  i::Vararg{Integer,N}
  ) where {T,N}

  @boundscheck checkbounds(s,i...)
  ispace...,itime,iparam = i
  ispace′ = s.index_map[ispace...]
  sparam = param_getindex(s.data[itime],iparam)
  setindex!(sparam,v,ispace′)
end

struct TransientSnapshotsAtIndices{T,N,L,D,I,R,A<:AbstractTransientSnapshots{T,N,L,D,I,R},B,C} <: AbstractTransientSnapshots{T,N,L,D,I,R}
  snaps::A
  trange::B
  prange::C
end

function RBSteady.Snapshots(s::TransientSnapshotsAtIndices,i::AbstractIndexMap,r::TransientParamRealization)
  snaps = Snapshots(s.snaps,i,r)
  TransientSnapshotsAtIndices(snaps,s.trange,s.prange)
end

time_indices(s::TransientSnapshotsAtIndices) = s.trange
ParamDataStructures.num_times(s::TransientSnapshotsAtIndices) = length(time_indices(s))
RBSteady.param_indices(s::TransientSnapshotsAtIndices) = s.prange
ParamDataStructures.num_params(s::TransientSnapshotsAtIndices) = length(RBSteady.param_indices(s))

IndexMaps.get_index_map(s::TransientSnapshotsAtIndices) = get_index_map(s.snaps)
RBSteady.get_realization(s::TransientSnapshotsAtIndices) = get_realization(s.snaps)[s.prange,s.trange]

function ParamDataStructures.get_values(
  s::TransientSnapshotsAtIndices{T,N,L,D,I,R,<:TransientBasicSnapshots}
  ) where {T,N,L,D,I,R}

  snaps = s.snaps
  item = testitem(snaps.data)
  data = array_of_similar_arrays(item,num_params(s)*num_times(s))
  indices = CartesianIndices((RBSteady.param_indices(s),time_indices(s)))
  @inbounds for (ipt,index) in enumerate(indices)
    ip,it = index.I
    @views data[ipt] = snaps.data[ip+(it-1)*num_params(s)]
  end
  return ArrayOfArrays(data)
end

function ParamDataStructures.get_values(
  s::TransientSnapshotsAtIndices{T,N,L,D,I,R,<:TransientSnapshots}
  ) where {T,N,L,D,I,R}

  snaps = s.snaps
  item = testitem(first(snaps.data).data)
  data = array_of_similar_arrays(item,num_params(s)*num_times(s))
  indices = CartesianIndices((RBSteady.param_indices(s),time_indices(s)))
  @inbounds for (ipt,index) in enumerate(indices)
    ip,it = index.I
    data[ipt] = snaps.data[it][ip]
  end
  return ArrayOfArrays(data)
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

function RBSteady.select_snapshots(s::TransientSnapshotsAtIndices,trange,prange)
  old_trange,old_prange = s.indices
  @check intersect(old_trange,trange) == trange
  @check intersect(old_prange,prange) == prange
  TransientSnapshotsAtIndices(s.snaps,trange,prange)
end

function RBSteady.select_snapshots(s::AbstractTransientSnapshots,trange,prange)
  trange = RBSteady.format_range(trange,num_times(s))
  prange = RBSteady.format_range(prange,num_params(s))
  TransientSnapshotsAtIndices(s,trange,prange)
end

function RBSteady.select_snapshots(s::AbstractTransientSnapshots,prange;trange=Base.OneTo(num_times(s)))
  select_snapshots(s,trange,prange)
end

function RBSteady.select_snapshots_entries(s::AbstractTransientSnapshots,srange,trange)
  _getindex(s::TransientBasicSnapshots,is,it,ip) = s.data.data[ip+(it-1)*num_params(s)][is]
  _getindex(s::TransientSnapshots,is,it,ip) = s.data[it].data[ip][is]

  @assert length(srange) == length(trange)

  T = eltype(s)
  nval = length(srange)
  np = num_params(s)
  entries = array_of_similar_arrays(zeros(T,nval),np)

  @inbounds for ip = 1:np
    vip = entries.data[ip]
    for (i,(is,it)) in enumerate(zip(srange,trange))
      vip[i] = _getindex(s,is,it,ip)
    end
  end

  return entries
end

const TransientSparseSnapshots{T,N,L,D,I,R,A<:MatrixOfSparseMatricesCSC} = Union{
  TransientBasicSnapshots{T,N,L,D,I,R,A},
  TransientSnapshots{T,N,L,D,I,R,A},
}

function ParamDataStructures.recast(s::TransientSparseSnapshots,a::AbstractVector{<:AbstractArray{T,3}}) where T
  index_map = get_index_map(s)
  ls = IndexMaps.get_univariate_sparsity(index_map)
  asparse = map(SparseCore,a,ls)
  return asparse
end

const UnfoldingTransientSnapshots{T,L,I<:TrivialIndexMap,R} = AbstractTransientSnapshots{T,3,L,1,I,R}
const UnfoldingTransientSparseSnapshots{T,L,I<:TrivialIndexMap,R,A<:MatrixOfSparseMatricesCSC} = TransientSparseSnapshots{T,3,L,1,I,R,A}

function ParamDataStructures.recast(s::UnfoldingTransientSparseSnapshots,a::AbstractMatrix)
  return recast(s.data,a)
end

abstract type ModeAxes end
struct Mode1Axes <: ModeAxes end
struct Mode2Axes <: ModeAxes end

change_mode(::Mode1Axes) = Mode2Axes()
change_mode(::Mode2Axes) = Mode1Axes()

struct ModeTransientSnapshots{M<:ModeAxes,T,L,I,R,A<:UnfoldingTransientSnapshots{T,L,I,R}} <: AbstractTransientSnapshots{T,2,L,1,I,R}
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

RBSteady.num_space_dofs(s::ModeTransientSnapshots) = prod(num_space_dofs(s.snaps))
ParamDataStructures.get_values(s::ModeTransientSnapshots) = get_values(s.snaps)
RBSteady.get_realization(s::ModeTransientSnapshots) = get_realization(s.snaps)

change_mode(s::UnfoldingTransientSnapshots) = ModeTransientSnapshots(s,change_mode(get_mode(s)))
change_mode(s::ModeTransientSnapshots) = ModeTransientSnapshots(s.snaps,change_mode(get_mode(s)))

get_mode(s::UnfoldingTransientSnapshots) = Mode1Axes()
get_mode(s::ModeTransientSnapshots) = s.mode

function RBSteady.select_snapshots_entries(s::UnfoldingTransientSnapshots,srange,trange)
  _getindex(s::TransientBasicSnapshots,is,it,ip) = param_getindex(s.data,ip+(it-1)*num_params(s))[is]
  _getindex(s::TransientSnapshots,is,it,ip) = param_getindex(s.data[it],ip)[is]

  T = eltype(s)
  nval = length(srange),length(trange)
  np = num_params(s)
  entries = array_of_similar_arrays(zeros(T,nval),np)

  @inbounds for ip = 1:np, (i,it) = enumerate(trange)
    entries.data[ip][:,i] = _getindex(s,srange,it,ip)
  end

  return entries
end

const Mode1TransientSnapshots{T,L,I,R,A} = ModeTransientSnapshots{Mode1Axes,T,L,I,R,A}
const Mode2TransientSnapshots{T,L,I,R,A} = ModeTransientSnapshots{Mode2Axes,T,L,I,R,A}

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

# compression operation

_compress(s,a,X::AbstractMatrix) = a'*X*s
_compress(s,a,args...) = a'*s

function compress(s::Mode1TransientSnapshots,a::AbstractMatrix,args...;swap_mode=true)
  s′ = _compress(collect(s),a,args...)
  if swap_mode
    s′ = change_mode(s′,num_params(s))
  end
  return s′
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
