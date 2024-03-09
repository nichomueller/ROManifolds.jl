abstract type TTSnapshots{T,N} <: AbstractSnapshots{T,N} end

Base.size(s::TTSnapshots) = (num_space_dofs(s),num_times(s),num_params(s))
Base.length(s::TTSnapshots) = prod(size(s))
Base.axes(s::TTSnapshots) = Base.OneTo.(size(s))

#= representation of a standard tensor-train snapshot
   [ [u(x1,t1,μ1) ⋯ u(x1,t1,μP)] [u(x1,t2,μ1) ⋯ u(x1,t2,μP)] [u(x1,t3,μ1) ⋯] [⋯] [u(x1,tT,μ1) ⋯ u(x1,tT,μP)] ]
         ⋮             ⋮          ⋮            ⋮           ⋮              ⋮             ⋮
   [ [u(xN,t1,μ1) ⋯ u(xN,t1,μP)] [u(xN,t2,μ1) ⋯ u(xN,t2,μP)] [u(xN,t3,μ1) ⋯] [⋯] [u(xN,tT,μ1) ⋯ u(xN,tT,μP)] ]
=#

struct BasicTTSnapshots{T,N,P,R} <: TTSnapshots{T,N}
  values::P
  realization::R
  function BasicTTSnapshots(values::P,realization::R) where {P<:ParamTTArray,R}
    T = eltype(P)
    N = FEM.get_dim(P)+2
    new{T,N,P,R}(values,realization)
  end
end

function BasicSnapshots(values::ParamTTArray,realization::TransientParamRealization,args...)
  BasicTTSnapshots(values,realization)
end

function BasicSnapshots(s::BasicTTSnapshots)
  s
end

num_space_dofs(s::BasicTTSnapshots) = length(first(s.values))

function Base.getindex(s::BasicTTSnapshots,ispace::Integer,itime::Integer,iparam::Integer)
  s.values[iparam+(itime-1)*num_params(s)][ispace]
end

function Base.setindex!(s::BasicTTSnapshots,v,ispace::Integer,itime::Integer,iparam::Integer)
  s.values[iparam+(itime-1)*num_params(s)][ispace] = v
end

struct TransientTTSnapshots{T,N,P,R,V} <: TTSnapshots{T,N}
  values::V
  realization::R
  function TransientTTSnapshots(
    values::AbstractVector{P},
    realization::R
    ) where {P<:ParamTTArray,R<:TransientParamRealization}

    V = typeof(values)
    T = eltype(P)
    N = FEM.get_dim(P)+2
    new{T,N,P,R,V}(values,realization)
  end
end

function TransientSnapshots(
  values::AbstractVector{P},realization::TransientParamRealization,args...) where P<:ParamTTArray
  TransientTTSnapshots(values,realization,args...)
end

num_space_dofs(s::TransientTTSnapshots) = length(first(first(s.values)))

function Base.getindex(s::TransientTTSnapshots,ispace::Integer,itime::Integer,iparam::Integer)
  s.values[itime][iparam][ispace]
end

function Base.setindex!(s::TransientTTSnapshots,v,ispace::Integer,itime::Integer,iparam::Integer)
  s.values[itime][iparam][ispace] = v
end

function BasicSnapshots(
  s::TransientTTSnapshots{T,<:ParamTTArray{T,N,A}}
  ) where {T,N,A}

  nt = num_times(s)
  np = num_params(s)
  array = Vector{eltype(A)}(undef,nt*np)
  @inbounds for i = 1:nt*np
    it = slow_index(i,np)
    ip = fast_index(i,np)
    array[i] = s.values[it][ip]
  end
  basic_values = ParamArray(array)
  BasicSnapshots(basic_values,s.realization,s.mode)
end

function FEM.get_values(s::TransientTTSnapshots)
  get_values(BasicSnapshots(s))
end

struct SelectedTTSnapshotsAtIndices{T,N,S,I} <: TTSnapshots{T,N}
  snaps::S
  selected_indices::I
  function SelectedTTSnapshotsAtIndices(
    snaps::TTSnapshots{T,N},
    selected_indices::I
    ) where {T,N,I}

    S = typeof(snaps)
    new{T,N,S,I}(snaps,selected_indices)
  end
end

function SelectedSnapshotsAtIndices(snaps::TTSnapshots,args...)
  SelectedTTSnapshotsAtIndices(snaps,args...)
end

function SelectedSnapshotsAtIndices(s::SelectedTTSnapshotsAtIndices,selected_indices)
  new_srange,new_trange,new_prange = selected_indices
  old_srange,old_trange,old_prange = s.selected_indices
  @check intersect(old_srange,new_srange) == new_srange
  @check intersect(old_trange,new_trange) == new_trange
  @check intersect(old_prange,new_prange) == new_prange
  SelectedTTSnapshotsAtIndices(s.snaps,selected_indices)
end

space_indices(s::SelectedTTSnapshotsAtIndices) = s.selected_indices[1]
time_indices(s::SelectedTTSnapshotsAtIndices) = s.selected_indices[2]
param_indices(s::SelectedTTSnapshotsAtIndices) = s.selected_indices[3]
num_space_dofs(s::SelectedTTSnapshotsAtIndices) = length(space_indices(s))
FEM.num_times(s::SelectedTTSnapshotsAtIndices) = length(time_indices(s))
FEM.num_params(s::SelectedTTSnapshotsAtIndices) = length(param_indices(s))

function Base.getindex(
  s::SelectedTTSnapshotsAtIndices,
  ispace::Integer,itime::Integer,iparam::Integer)
  is = space_indices(s)[ispace]
  it = time_indices(s)[itime]
  ip = param_indices(s)[iparam]
  getindex(s.snaps,is,it,ip)
end

function Base.setindex!(
  s::SelectedTTSnapshotsAtIndices,
  v,ispace::Integer,itime::Integer,iparam::Integer)
  is = space_indices(s)[ispace]
  it = time_indices(s)[itime]
  ip = param_indices(s)[iparam]
  setindex!(s.snaps,v,is,it,ip)
end

function FEM.get_values(s::SelectedTTSnapshotsAtIndices{T,N,<:BasicTTSnapshots}) where {T,N}
  @check space_indices(s) == Base.OneTo(num_space_dofs(s))
  v = get_values(s.snaps)
  array = Vector{typeof(first(v))}(undef,num_cols(s))
  @inbounds for (i,it) in enumerate(time_indices(s))
    for (j,jp) in enumerate(param_indices(s))
      array[(i-1)*num_params(s)+j] = v[(it-1)*num_params(s)+jp]
    end
  end
  ParamArray(array)
end

function FEM.get_values(s::SelectedTTSnapshotsAtIndices{T,N,<:TransientTTSnapshots}) where {T,N}
  get_values(BasicSnapshots(s))
end

function get_realization(s::SelectedTTSnapshotsAtIndices)
  r = get_realization(s.snaps)
  r[param_indices(s),time_indices(s)]
end

function BasicSnapshots(s::SelectedTTSnapshotsAtIndices{T,N,<:BasicTTSnapshots}) where {T,N}
  values = get_values(s)
  r = get_realization(s)
  BasicTTSnapshots(values,r)
end

function BasicSnapshots(s::SelectedTTSnapshotsAtIndices{T,N,<:TransientTTSnapshots}) where {T,N}
  @check space_indices(s) == Base.OneTo(num_space_dofs(s))
  v = s.snaps.values
  basic_values = Vector{typeof(first(first(v)))}(undef,num_times(s)*num_params(s))
  @inbounds for (i,it) in enumerate(time_indices(s))
    for (j,jp) in enumerate(param_indices(s))
      basic_values[(i-1)*num_params(s)+j] = v[it][jp]
    end
  end
  r = get_realization(s)
  BasicTTSnapshots(ParamArray(basic_values),r)
end

const BasicNnzTTSnapshots = BasicTTSnapshots{T,N,P,R} where {T,N,P<:ParamTTSparseMatrix,R}
const TransientNnzTTSnapshots = TransientTTSnapshots{T,N,P,R} where {T,N,P<:ParamTTSparseMatrix,R}
const SelectedNnzTTSnapshotsAtIndices = SelectedTTSnapshotsAtIndices{T,N,S,I} where {T,N,S<:Union{BasicNnzTTSnapshots,TransientNnzTTSnapshots},I}
const NnzTTSnapshots = Union{
  BasicNnzTTSnapshots{T,N},
  TransientNnzTTSnapshots{T,N},
  SelectedNnzTTSnapshotsAtIndices{T,N}} where {T,N}

num_space_dofs(s::BasicNnzTTSnapshots) = nnz(first(s.values))

function tensor_getindex(s::BasicNnzTTSnapshots,ispace::Integer,itime::Integer,iparam::Integer)
  nonzeros(s.values[iparam+(itime-1)*num_params(s)])[ispace]
end

function tensor_setindex!(s::BasicNnzTTSnapshots,v,ispace::Integer,itime::Integer,iparam::Integer)
  nonzeros(s.values[iparam+(itime-1)*num_params(s)])[ispace] = v
end

num_space_dofs(s::TransientNnzTTSnapshots) = nnz(first(first(s.values)))

function tensor_getindex(
  s::TransientNnzTTSnapshots,
  ispace::Integer,itime::Integer,iparam::Integer)
  nonzeros(s.values[itime][iparam])[ispace]
end

function tensor_setindex!(
  s::TransientNnzTTSnapshots,
  v,ispace::Integer,itime::Integer,iparam::Integer)
  nonzeros(s.values[itime][iparam])[ispace] = v
end

function get_nonzero_indices(s::NnzTTSnapshots)
  v = isa(s,BasicTTSnapshots) ? first(s.values) : first(first(s.values))
  i,j, = findnz(v)
  return i .+ (j .- 1)*v.m
end

function recast(s::NnzTTSnapshots{Mode1Axis},a::AbstractMatrix)
  s1 = first(s.values)
  r = get_realization(s)
  r1 = r[axes(a,2),Base.OneTo(1)]
  i,j, = findnz(s1)
  m,n = size(s1)
  asparse = map(eachcol(a)) do v
    sparse(i,j,v,m,n)
  end
  return Snapshots(ParamArray(asparse),r1,s.mode)
end
