abstract type TTSnapshots{T,N} <: AbstractSnapshots{T,N} end

Base.size(s::TTSnapshots) = (num_space_dofs(s)...,num_times(s),num_params(s))
Base.length(s::TTSnapshots) = prod(size(s))
Base.axes(s::TTSnapshots) = Base.OneTo.(size(s))

num_space_dofs(s::TTSnapshots) = size(get_index_map(s))

Base.IndexStyle(::Type{<:TTSnapshots}) = IndexCartesian()

function Base.getindex(s::TTSnapshots,i...)
  view(s,i...)
end

function Base.getindex(s::TTSnapshots{T,N},i::Integer...) where {T,N}
  getindex(s,CartesianIndex(i))
end

function Base.getindex(s::TTSnapshots{T,N},i::CartesianIndex{N}) where {T,N}
  ispace = CartesianIndex(i.I[1:N-2])
  itime = i.I[end-1]
  iparam = i.I[end]
  tensor_getindex(s,ispace,itime,iparam)
end

function Base.setindex!(s::TTSnapshots{T,N},v,i::Integer...) where {T,N}
  ispace = CartesianIndex(i[1:N-2])
  itime = i[end-1]
  iparam = i[end]
  tensor_setindex!(s,v,ispace,itime,iparam)
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
  @check isa(spacerange,Union{Colon,CartesianIndices})
  srange = isa(spacerange,Colon) ? CartesianIndices(num_space_dofs(s)) : spacerange
  trange = isa(timerange,Colon) ? LinearIndices((num_times(s),)) : timerange
  trange = isa(trange,Integer) ? [trange] : trange
  prange = isa(paramrange,Colon) ? LinearIndices((num_params(s),)) : paramrange
  prange = isa(prange,Integer) ? [prange] : prange
  selected_indices = (srange,trange,prange)
  SelectedSnapshotsAtIndices(s,selected_indices)
end

space_indices(s::SelectedTTSnapshotsAtIndices) = s.selected_indices[1]
time_indices(s::SelectedTTSnapshotsAtIndices) = s.selected_indices[2]
param_indices(s::SelectedTTSnapshotsAtIndices) = s.selected_indices[3]
num_space_dofs(s::SelectedTTSnapshotsAtIndices) = size(space_indices(s))
FEM.num_times(s::SelectedTTSnapshotsAtIndices) = length(time_indices(s))
FEM.num_params(s::SelectedTTSnapshotsAtIndices) = length(param_indices(s))

FEM.get_index_map(s::SelectedTTSnapshotsAtIndices) = get_index_map(s.snaps)

function tensor_getindex(s::SelectedTTSnapshotsAtIndices,ispace,itime,iparam)
  is = space_indices(s)[ispace]
  it = time_indices(s)[itime]
  ip = param_indices(s)[iparam]
  tensor_getindex(s.snaps,is,it,ip)
end

function tensor_setindex!(s::SelectedTTSnapshotsAtIndices,v,ispace,itime,iparam)
  is = space_indices(s)[ispace]
  it = time_indices(s)[itime]
  ip = param_indices(s)[iparam]
  tensor_setindex!(s.snaps,v,is,it,ip)
end

function FEM.get_values(s::SelectedTTSnapshotsAtIndices{T,N,<:BasicTTSnapshots}) where {T,N}
  v = get_values(s.snaps)
  array = Vector{typeof(first(v))}(undef,num_times(s)*num_params(s))
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

function _to_linear_indices(i::CartesianIndices,args...)
  vec(LinearIndices(i))
end
function _to_linear_indices(i::Tuple{Vararg{AbstractVector}},sizes)
  for k = 1:length(i)-1
    i[k+1] .+= sizes[k]
  end
  vcat(i...)
end

function BasicSnapshots(s::SelectedTTSnapshotsAtIndices{T,N,<:TransientTTSnapshots}) where {T,N}
  v = s.snaps.values
  basic_values = Vector{typeof(first(first(v)))}(undef,num_times(s)*num_params(s))
  ispace = space_indices(s)
  lispace = _to_linear_indices(ispace,num_space_dofs(s.snaps))
  @inbounds for (i,it) in enumerate(time_indices(s))
    for (j,jp) in enumerate(param_indices(s))
      basic_values[(i-1)*num_params(s)+j] = vec(v[it][jp][lispace])
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

###################################################################

abstract type MatrixTTSnapshots{T,N} <: TTSnapshots{T,N} end

function _mat_size(i::AbstractIndexMap{D}) where D
  _size_d(d) = size(i,cld(d,2))
  ntuple(_size_d,2*D)
end

num_space_dofs(s::MatrixTTSnapshots) = _mat_size(get_index_map(s)) #(size(get_index_map(s))...,size(get_index_map(s))...)

function tensor_getindex(s::MatrixTTSnapshots,ispace::CartesianIndex{D},itime,iparam) where D
  ispace_row = ispace.I[1:2:D]
  ispace_col = ispace.I[2:2:D]
  perm_ispace_row = get_index_map(s)[CartesianIndex(ispace_row)]
  perm_ispace_col = get_index_map(s)[CartesianIndex(ispace_col)]
  _tensor_getindex(s,perm_ispace_row,perm_ispace_col,itime,iparam)
end

function tensor_setindex!(s::MatrixTTSnapshots,v,ispace::CartesianIndex{D},itime,iparam) where D
  ispace_row = ispace.I[1:2:2*D]
  ispace_col = ispace.I[2:2:2*D]
  perm_ispace_row = get_index_map(s)[CartesianIndex(ispace_row)]
  perm_ispace_col = get_index_map(s)[CartesianIndex(ispace_col)]
  _tensor_setindex!(s,v,perm_ispace_row,perm_ispace_col,itime,iparam)
end

struct MatrixBasicTTSnapshots{T,N,P,R} <: MatrixTTSnapshots{T,N}
  values::P
  realization::R
  function MatrixBasicTTSnapshots(values::P,realization::R) where {P<:FEM.ParamTTSparseMatrix,R}
    T = eltype(P)
    D = ndims(get_index_map(values))
    N = 2*D+2
    new{T,N,P,R}(values,realization)
  end
end

function BasicSnapshots(values::FEM.ParamTTSparseMatrix,realization::TransientParamRealization)
  MatrixBasicTTSnapshots(values,realization)
end

FEM.get_index_map(s::MatrixBasicTTSnapshots) = get_index_map(s.values)

function _tensor_getindex(s::MatrixBasicTTSnapshots,ispace_row,ispace_col,itime,iparam)
  s.values[iparam+(itime-1)*num_params(s)][ispace_row,ispace_col]
end

function _tensor_setindex!(s::MatrixBasicTTSnapshots,v,ispace_row,ispace_col,itime,iparam)
  s.values[iparam+(itime-1)*num_params(s)][ispace_row,ispace_col] = v
end

struct MatrixTransientTTSnapshots{T,N,P,R} <: MatrixTTSnapshots{T,N}
  values::P
  realization::R
  function MatrixTransientTTSnapshots(
    values::AbstractVector{P},realization::R) where {P<:FEM.ParamTTSparseMatrix,R}
    T = eltype(P)
    D = ndims(get_index_map(values))
    N = 2*D+2
    new{T,N,P,R}(values,realization)
  end
end

function TransientSnapshots(
  values::AbstractVector{P},
  realization::TransientParamRealization
  ) where P<:FEM.ParamTTSparseMatrix

  MatrixTransientTTSnapshots(values,realization)
end

FEM.get_index_map(s::MatrixTransientTTSnapshots) = get_index_map(first(s.values))

function _tensor_getindex(s::MatrixTransientTTSnapshots,ispace_row,ispace_col,itime,iparam)
  s.values[itime][iparam][ispace_row,ispace_col]
end

function _tensor_setindex!(s::MatrixTransientTTSnapshots,v,ispace_row,ispace_col,itime,iparam)
  s.values[itime][iparam][ispace_row,ispace_col] = v
end
