abstract type TTSnapshots{T,N} <: AbstractSnapshots{T,N} end

Base.size(s::TTSnapshots) = (num_space_dofs(s)...,num_times(s),num_params(s))
Base.length(s::TTSnapshots) = prod(size(s))
Base.axes(s::TTSnapshots) = Base.OneTo.(size(s))

num_space_dofs(s::TTSnapshots) = size(get_index_map(s))

function Base.getindex(s::TTSnapshots{T,3},ix,itime,iparam) where T
  view(s,ix,itime,iparam)
end
function Base.getindex(s::TTSnapshots{T,4},ix,iy,itime,iparam) where T
  view(s,ix,iy,itime,iparam)
end
function Base.getindex(s::TTSnapshots{T,5},ix,iy,iz,itime,iparam) where T
  view(s,ix,iy,iz,itime,iparam)
end

function Base.getindex(s::TTSnapshots{T,3},ix::Integer,itime::Integer,iparam::Integer) where T
  tensor_getindex(s,CartesianIndex(ix),itime,iparam)
end
function Base.getindex(s::TTSnapshots{T,4},ix::Integer,iy::Integer,itime::Integer,iparam::Integer) where T
  tensor_getindex(s,CartesianIndex(ix,iy),itime,iparam)
end
function Base.getindex(s::TTSnapshots{T,5},ix::Integer,iy::Integer,iz::Integer,itime::Integer,iparam::Integer) where T
  tensor_getindex(s,CartesianIndex(ix,iy,iz),itime,iparam)
end

function Base.setindex!(s::TTSnapshots{T,3},v,ix::Integer,itime::Integer,iparam::Integer) where T
  tensor_setindex!(s,v,CartesianIndex(ix),itime,iparam)
end
function Base.setindex!(s::TTSnapshots{T,4},v,ix::Integer,iy::Integer,itime::Integer,iparam::Integer) where T
  tensor_setindex!(s,v,CartesianIndex(ix,iy),itime,iparam)
end
function Base.setindex!(s::TTSnapshots{T,5},v,ix::Integer,iy::Integer,iz::Integer,itime::Integer,iparam::Integer) where T
  tensor_setindex!(s,v,CartesianIndex(ix,iy,iz),itime,iparam)
end

reverse_snapshots(s::TTSnapshots) = s

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
    D = ndims(get_index_map(values))
    N = D+2
    new{T,N,P,R}(values,realization)
  end
end

function BasicSnapshots(values::ParamTTArray,realization::TransientParamRealization)
  BasicTTSnapshots(values,realization)
end

function BasicSnapshots(s::BasicTTSnapshots)
  s
end

FEM.get_index_map(s::BasicTTSnapshots) = get_index_map(s.values)

function tensor_getindex(s::BasicTTSnapshots,ispace,itime,iparam)
  perm_ispace = get_index_map(s)[ispace]
  s.values[iparam+(itime-1)*num_params(s)][perm_ispace]
end

function tensor_setindex!(s::BasicTTSnapshots,v,ispace,itime,iparam)
  perm_ispace = get_index_map(s)[ispace]
  s.values[iparam+(itime-1)*num_params(s)][perm_ispace] = v
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
    D = ndims(get_index_map(first(values)))
    N = D+2
    new{T,N,P,R,V}(values,realization)
  end
end

function TransientSnapshots(
  values::AbstractVector{P},
  realization::TransientParamRealization
  ) where P<:ParamTTArray

  TransientTTSnapshots(values,realization)
end

FEM.get_index_map(s::TransientTTSnapshots) = get_index_map(first(s.values))

function tensor_getindex(s::TransientTTSnapshots,ispace,itime,iparam)
  perm_ispace = get_index_map(s)[ispace]
  s.values[itime][iparam][perm_ispace]
end

function tensor_setindex!(s::TransientTTSnapshots,v,ispace,itime,iparam)
  perm_ispace = get_index_map(s)[ispace]
  s.values[itime][iparam][perm_ispace] = v
end

function BasicSnapshots(
  s::TransientTTSnapshots{T,N,<:ParamArray{T,M,L,A}}
  ) where {T,N,M,L,A}

  nt = num_times(s)
  np = num_params(s)
  array = Vector{eltype(A)}(undef,nt*np)
  @inbounds for i = 1:nt*np
    it = slow_index(i,np)
    ip = fast_index(i,np)
    array[i] = s.values[it][ip]
  end
  basic_values = ParamArray(array)
  BasicSnapshots(basic_values,s.realization)
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

function select_snapshots(s::TTSnapshots,spacerange,timerange,paramrange)
  srange = isa(spacerange,Colon) ? Base.OneTo.(num_space_dofs(s)) : spacerange
  srange = map(i->isa(i,Integer) ? [i] : i,srange)
  trange = isa(timerange,Colon) ? Base.OneTo(num_times(s)) : timerange
  trange = isa(trange,Integer) ? [trange] : trange
  prange = isa(paramrange,Colon) ? Base.OneTo(num_params(s)) : paramrange
  prange = isa(prange,Integer) ? [prange] : prange
  selected_indices = (srange,trange,prange)
  SelectedSnapshotsAtIndices(s,selected_indices)
end

space_indices(s::SelectedTTSnapshotsAtIndices) = s.selected_indices[1]
time_indices(s::SelectedTTSnapshotsAtIndices) = s.selected_indices[2]
param_indices(s::SelectedTTSnapshotsAtIndices) = s.selected_indices[3]
num_space_dofs(s::SelectedTTSnapshotsAtIndices) = length.(space_indices(s))
FEM.num_times(s::SelectedTTSnapshotsAtIndices) = length(time_indices(s))
FEM.num_params(s::SelectedTTSnapshotsAtIndices) = length(param_indices(s))

FEM.get_index_map(s::SelectedTTSnapshotsAtIndices) = get_index_map(s.snaps)

function tensor_getindex(s::SelectedTTSnapshotsAtIndices,ispace,itime,iparam)
  is = CartesianIndex(getindex.(space_indices(s),Tuple(ispace)))
  it = time_indices(s)[itime]
  ip = param_indices(s)[iparam]
  getindex(s.snaps,is,it,ip)
end

function tensor_getindex(s::SelectedTTSnapshotsAtIndices,v,ispace,itime,iparam)
  is = CartesianIndex(getindex.(space_indices(s),Tuple(ispace)))
  it = time_indices(s)[itime]
  ip = param_indices(s)[iparam]
  setindex!(s.snaps,v,is,it,ip)
end

function FEM.get_values(s::SelectedTTSnapshotsAtIndices{T,N,<:BasicTTSnapshots}) where {T,N}
  v = get_values(s.snaps)
  array = Vector{typeof(first(v))}(undef,num_cols(s))
  ispace = space_indices(s)
  index_map = get_index_map(s)
  perm_ispace = map(i->index_map[CartesianIndex(i)],ispace)
  @inbounds for (i,it) in enumerate(time_indices(s))
    for (j,jp) in enumerate(param_indices(s))
      array[(i-1)*num_params(s)+j] = v[(it-1)*num_params(s)+jp][perm_ispace]
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
  v = s.snaps.values
  basic_values = Vector{typeof(first(first(v)))}(undef,num_times(s)*num_params(s))
  ispace = space_indices(s)
  index_map = get_index_map(s)
  perm_ispace = map(i->index_map[CartesianIndex(i)],ispace)
  @inbounds for (i,it) in enumerate(time_indices(s))
    for (j,jp) in enumerate(param_indices(s))
      basic_values[(i-1)*num_params(s)+j] = v[it][jp][perm_ispace]
    end
  end
  r = get_realization(s)
  BasicTTSnapshots(ParamArray(basic_values),r)
end

function select_snapshots_entries(s::TTSnapshots,ispace,itime)
  _select_snapshots_entries(s,ispace,itime)
end

function _select_snapshots_entries(s::TTSnapshots{T},ispace,itime) where T
  @assert length(ispace) == length(itime)
  nval = length(ispace)
  np = num_params(s)
  values = allocate_param_array(zeros(T,nval),np)
  for ip = 1:np
    vip = values[ip]
    for (istp,(is,it)) in enumerate(zip(ispace,itime))
      vip[istp] = s[is,it,ip]
    end
  end
  return values
end

const BasicNnzTTSnapshots = BasicTTSnapshots{T,N,P,R} where {T,N,P<:ParamSparseMatrix,R}
const TransientNnzTTSnapshots = TransientTTSnapshots{T,N,P,R} where {T,N,P<:ParamSparseMatrix,R}
const SelectedNnzTTSnapshotsAtIndices = SelectedTTSnapshotsAtIndices{T,N,S,I} where {T,N,S<:Union{BasicNnzTTSnapshots,TransientNnzTTSnapshots},I}
const NnzTTSnapshots = Union{
  BasicNnzTTSnapshots{T,N},
  TransientNnzTTSnapshots{T,N},
  SelectedNnzTTSnapshotsAtIndices{T,N}} where {T,N}

function tensor_getindex(s::BasicNnzTTSnapshots,ispace,itime,iparam)
  perm_ispace = get_index_map(s)[ispace]
  nonzeros(s.values[iparam+(itime-1)*num_params(s)])[perm_ispace]
end

function tensor_setindex!(s::BasicNnzTTSnapshots,v,ispace,itime,iparam)
  perm_ispace = get_index_map(s)[ispace]
  nonzeros(s.values[iparam+(itime-1)*num_params(s)])[perm_ispace] = v
end

function tensor_getindex(s::TransientNnzTTSnapshots,ispace,itime,iparam)
  nonzeros(s.values[itime][iparam])[ispace]
end

function tensor_setindex!(s::TransientNnzTTSnapshots,v,ispace,itime,iparam)
  perm_ispace = get_index_map(s)[ispace]
  nonzeros(s.values[itime][iparam])[perm_ispace] = v
end

sparsify_indices(s::BasicNnzTTSnapshots,srange::AbstractVector) = sparsify_indices(first(s.values),srange)
sparsify_indices(s::TransientNnzTTSnapshots,srange::AbstractVector) = sparsify_indices(first(first(s.values)),srange)

function select_snapshots_entries(s::NnzTTSnapshots,ispace,itime)
  _select_snapshots_entries(s,sparsify_indices(s,ispace),itime)
end

function get_nonzero_indices(s::NnzTTSnapshots)
  v = isa(s,BasicTTSnapshots) ? first(s.values) : first(first(s.values))
  i,j, = findnz(v)
  return i .+ (j .- 1)*v.m
end

function recast(s::NnzTTSnapshots,a::AbstractArray{T,3}) where T
  @check size(a,1) == 1
  v = isa(s,BasicTTSnapshots) ? first(s.values) : first(first(s.values))
  i,j, = findnz(v)
  m,n = size(v)
  asparse = map(eachcol(dropdims(a;dims=1))) do v
    sparse(i,j,v,m,n)
  end
  return VecOfSparseMat2Arr3(asparse)
end

struct VecOfSparseMat2Arr3{Tv,Ti,V} <: AbstractArray{Tv,3}
  values::V
  function VecOfSparseMat2Arr3(values::V) where {Tv,Ti,V<:AbstractVector{<:AbstractSparseMatrix{Tv,Ti}}}
    new{Tv,Ti,V}(values)
  end
end

FEM.get_values(s::VecOfSparseMat2Arr3) = s.values
Base.size(s::VecOfSparseMat2Arr3) = (1,nnz(first(s.values)),length(s.values))

function Base.getindex(s::VecOfSparseMat2Arr3,i::Integer,j,k::Integer)
  @check i == 1
  nonzeros(s.values[k])[j]
end

function Base.getindex(s::VecOfSparseMat2Arr3,i::Integer,j,k)
  @check i == 1
  view(s,i,j,k)
end

function get_nonzero_indices(s::VecOfSparseMat2Arr3)
  get_nonzero_indices(first(s.values))
end
