#= mode-1 representation of a snapshot tensor (default)
   [ [u(x1,t1,μ1) ⋯ u(x1,t1,μP)] [u(x1,t2,μ1) ⋯ u(x1,t2,μP)] [u(x1,t3,μ1) ⋯] [⋯] [u(x1,tT,μ1) ⋯ u(x1,tT,μP)] ]
         ⋮             ⋮          ⋮            ⋮           ⋮              ⋮             ⋮
   [ [u(xN,t1,μ1) ⋯ u(xN,t1,μP)] [u(xN,t2,μ1) ⋯ u(xN,t2,μP)] [u(xN,t3,μ1) ⋯] [⋯] [u(xN,tT,μ1) ⋯ u(xN,tT,μP)] ]
=#
#= mode-2 representation of a snapshot tensor
   [ [u(x1,t1,μ1) ⋯ u(x1,t1,μP)] [u(x2,t1,μ1) ⋯ u(x2,t1,μP)] [u(x3,t1,μ1) ⋯] [⋯] [u(xN,t1,μ1) ⋯ u(xN,t1,μP)] ]
         ⋮             ⋮          ⋮            ⋮           ⋮              ⋮             ⋮
   [ [u(x1,tT,μ1) ⋯ u(x1,tT,μP)] [u(x2,tT,μ1) ⋯ u(x2,tT,μP)] [u(x3,tT,μ1) ⋯] [⋯] [u(xN,tT,μ1) ⋯ u(xN,tT,μP)] ]
=#

struct Mode1Axis end
struct Mode2Axis end

abstract type AbstractTransientSnapshots{M,T} <: AbstractParamContainer{T,2} end

Base.ndims(::AbstractTransientSnapshots) = 2
Base.ndims(::Type{<:AbstractTransientSnapshots}) = 2
Base.IndexStyle(::Type{<:AbstractTransientSnapshots}) = IndexLinear()

FEM.get_values(s::AbstractTransientSnapshots) = s.values
get_realization(s::AbstractTransientSnapshots) = s.realization
get_mode(s::AbstractTransientSnapshots) = s.mode

FEM.num_times(s::AbstractTransientSnapshots) = num_times(get_realization(s))
FEM.num_params(s::AbstractTransientSnapshots) = num_params(get_realization(s))
num_rows(s::AbstractTransientSnapshots{Mode1Axis}) = num_space_dofs(s)
num_rows(s::AbstractTransientSnapshots{Mode2Axis}) = num_times(s)
num_cols(s::AbstractTransientSnapshots{Mode1Axis}) = num_times(s)*num_params(s)
num_cols(s::AbstractTransientSnapshots{Mode2Axis}) = num_space_dofs(s)*num_params(s)

Base.length(s::AbstractTransientSnapshots) = num_rows(s)*num_cols(s)
Base.size(s::AbstractTransientSnapshots) = (num_rows(s),num_cols(s))
Base.axes(s::AbstractTransientSnapshots) = Base.OneTo.(size(s))
Base.eltype(::AbstractTransientSnapshots{M,T}) where {M,T} = T
Base.eltype(::Type{<:AbstractTransientSnapshots{M,T}}) where {M,T} = T

slow_index(i,N::Int) = Int.(floor.((i .- 1) ./ N) .+ 1)
slow_index(i::Colon,::Int) = i
fast_index(i,N::Int) = mod.(i .- 1,N) .+ 1
fast_index(i::Colon,::Int) = i

function col_index(s::AbstractTransientSnapshots,mode2_index,param_index)
  (mode2_index .- 1)*num_params(s) .+ param_index
end
function col_index(s::AbstractTransientSnapshots,mode2_index::AbstractVector,param_index::AbstractVector)
  vec(transpose((mode2_index.-1)*num_params(s) .+ collect(param_index)'))
end
function col_index(s::AbstractTransientSnapshots{Mode1Axis},::Colon,param_index)
  col_index(s,1:num_times(s),param_index)
end
function col_index(s::AbstractTransientSnapshots{Mode2Axis},::Colon,param_index)
  col_index(s,1:num_space_dofs(s),param_index)
end
function col_index(s::AbstractTransientSnapshots{Mode1Axis},::Colon,::Colon)
  col_index(s,1:num_times(s),1:num_params(s))
end
function col_index(s::AbstractTransientSnapshots{Mode2Axis},::Colon,::Colon)
  col_index(s,1:num_space_dofs(s),1:num_params(s))
end

function Base.getindex(s::AbstractTransientSnapshots,i)
  nrow = num_rows(s)
  irow = fast_index(i,nrow)
  icol = slow_index(i,nrow)
  getindex(s,irow,icol)
end

function Base.getindex(s::AbstractTransientSnapshots,k::CartesianIndex)
  ispace,j = k.I
  getindex(s,ispace,j)
end

function Base.getindex(s::AbstractTransientSnapshots{Mode1Axis},ispace,j)
  np = num_params(s)
  itime = slow_index(j,np)
  iparam = fast_index(j,np)
  tensor_getindex(s,ispace,itime,iparam)
end

function Base.getindex(s::AbstractTransientSnapshots{Mode2Axis},itime,j)
  np = num_params(s)
  ispace = slow_index(j,np)
  iparam = fast_index(j,np)
  tensor_getindex(s,ispace,itime,iparam)
end

function tensor_getindex(s::AbstractTransientSnapshots,ispace,itime,iparam)
  view(s,ispace,col_index(s,itime,iparam))
end

function Base.setindex!(s::AbstractTransientSnapshots,v,i)
  nrow = num_rows(s)
  irow = fast_index(i,nrow)
  icol = slow_index(i,nrow)
  setindex!(s,v,irow,icol)
end

function Base.setindex!(s::AbstractTransientSnapshots,v,k::CartesianIndex)
  ispace,j = k.I
  setindex!(s,v,ispace,j)
end

function Base.setindex!(s::AbstractTransientSnapshots{Mode1Axis},v,ispace,j)
  np = num_params(s)
  itime = slow_index(j,np)
  iparam = fast_index(j,np)
  tensor_setindex!(s,v,ispace,itime,iparam)
end

function Base.setindex!(s::AbstractTransientSnapshots{Mode2Axis},v,itime,j)
  np = num_params(s)
  ispace = slow_index(j,np)
  iparam = fast_index(j,np)
  tensor_setindex!(s,v,ispace,itime,iparam)
end

function tensor_setindex!(s::AbstractTransientSnapshots,v,ispace,itime,iparam)
  si = view(s,ispace,col_index(s,itime,iparam))
  si = v
end

function compress(a::AbstractMatrix,s::AbstractTransientSnapshots{Mode1Axis})
  @fastmath compressed_values = a'*s
  r = get_realization(s)
  Snapshots(compressed_values,r,Mode1Axis())
end

function compress(a::AbstractMatrix,s::AbstractTransientSnapshots{Mode2Axis})
  @fastmath compressed_values = a'*s
  r = get_realization(s)
  compressed_realization = r[:,axes(a,2)]
  Snapshots(compressed_values,compressed_realization,Mode2Axis())
end

function Snapshots(a::ArrayContribution,args...)
  b = array_contribution()
  for (trian,values) in a.dict
    b[trian] = Snapshots(values,args...)
  end
  b
end

struct BasicSnapshots{M,T,P,R} <: AbstractTransientSnapshots{M,T}
  mode::M
  values::P
  realization::R
  function BasicSnapshots(
    mode::M,
    values::P,
    realization::R
    ) where {M,P<:AbstractParamContainer,R}

    T = eltype(P)
    new{M,T,P,R}(mode,values,realization)
  end
end

function BasicSnapshots(s::BasicSnapshots)
  s
end

function Snapshots(
  values::P,
  realization::R,
  mode::M=Mode1Axis()
  ) where {M,P<:AbstractParamContainer,R}

  BasicSnapshots(mode,values,realization)
end

num_space_dofs(s::BasicSnapshots) = length(first(s.values))

function change_mode(s::BasicSnapshots{Mode1Axis})
  BasicSnapshots(Mode2Axis(),s.values,s.realization)
end

function change_mode(s::BasicSnapshots{Mode2Axis})
  BasicSnapshots(Mode1Axis(),s.values,s.realization)
end

function tensor_getindex(s::BasicSnapshots,ispace::Integer,itime::Integer,iparam::Integer)
  s.values[iparam+(itime-1)*num_params(s)][ispace]
end

function tensor_setindex!(s::BasicSnapshots,v,ispace::Integer,itime::Integer,iparam::Integer)
  s.values[iparam+(itime-1)*num_params(s)][ispace] = v
end

struct TransientSnapshots{M,T,P,R,V} <: AbstractTransientSnapshots{M,T}
  mode::M
  values::V
  realization::R
  function TransientSnapshots(
    mode::M,
    values::AbstractVector{P},
    realization::R
    ) where {M,P<:AbstractParamContainer,R<:TransientParamRealization}

    V = typeof(values)
    T = eltype(P)
    new{M,T,P,R,V}(mode,values,realization)
  end
end

function Snapshots(
  values::AbstractVector{P},
  realization::R,
  mode::M=Mode1Axis()
  ) where {M,P<:AbstractParamContainer,R}

  TransientSnapshots(mode,values,realization)
end

num_space_dofs(s::TransientSnapshots) = length(first(first(s.values)))

function change_mode(s::TransientSnapshots{Mode1Axis})
  TransientSnapshots(Mode2Axis(),s.values,s.realization)
end

function change_mode(s::TransientSnapshots{Mode2Axis})
  TransientSnapshots(Mode1Axis(),s.values,s.realization)
end

function tensor_getindex(s::TransientSnapshots,ispace::Integer,itime::Integer,iparam::Integer)
  s.values[itime][iparam][ispace]
end

function tensor_setindex!(s::TransientSnapshots,v,ispace::Integer,itime::Integer,iparam::Integer)
  s.values[itime][iparam][ispace] = v
end

function BasicSnapshots(
  s::TransientSnapshots{M,T,<:ParamArray{T,N,A}}
  ) where {M,T,N,A}

  nt = num_times(s)
  np = num_params(s)
  array = Vector{eltype(A)}(undef,nt*np)
  @inbounds for i = 1:nt*np
    it = slow_index(i,np)
    ip = fast_index(i,np)
    array[i] = s.values[it][ip]
  end
  basic_values = ParamArray(array)
  BasicSnapshots(s.mode,basic_values,s.realization)
end

function FEM.get_values(s::TransientSnapshots)
  get_values(BasicSnapshots(s))
end

struct CompressedTransientSnapshots{M,N,T,R,V} <: AbstractTransientSnapshots{M,T}
  current_mode::M
  initial_mode::N
  values::V
  realization::R
  function CompressedTransientSnapshots(
    current_mode::M,
    initial_mode::N,
    values::AbstractMatrix{T},
    realization::R) where {M,N,T,R}

    V = typeof(values)
    new{M,N,T,R,V}(current_mode,initial_mode,values,realization)
  end
end

function Snapshots(
  values::AbstractMatrix{T},
  realization::R,
  mode::M=Mode1Axis()
  ) where {M,T,R}

  CompressedTransientSnapshots(mode,mode,values,realization)
end

num_space_dofs(s::CompressedTransientSnapshots{M,Mode1Axis}) where M = size(s.values,1)
num_space_dofs(s::CompressedTransientSnapshots{M,Mode2Axis}) where M = Int(size(s.values,2) / num_params(s))

function change_mode(s::CompressedTransientSnapshots{Mode1Axis})
  CompressedTransientSnapshots(Mode2Axis(),Mode1Axis(),s.values,s.realization)
end

function change_mode(s::CompressedTransientSnapshots{Mode2Axis})
  CompressedTransientSnapshots(Mode1Axis(),Mode2Axis(),s.values,s.realization)
end

column_index(a,b,Na,Nb) = (a-1)*Nb+b
column_index(a::Colon,b,Na,Nb) = b:Nb:Na*Nb
column_index(a,b::Colon,Na,Nb) = a:Na:Na*Nb
column_index(a::Colon,b::Colon,Na,Nb) = a

function tensor_getindex(
  s::CompressedTransientSnapshots{M,Mode1Axis},
  ispace,itime,iparam
  ) where M

  nt = num_times(s)
  np = num_params(s)
  icolumn = column_index(itime,iparam,nt,np)
  s.values[ispace,icolumn]
end
function tensor_getindex(
  s::CompressedTransientSnapshots{M,Mode2Axis},
  ispace,itime,iparam
  ) where M

  ns = num_space_dofs(s)
  np = num_params(s)
  icolumn = column_index(ispace,iparam,ns,np)
  s.values[itime,icolumn]
end

function tensor_setindex!(
  s::CompressedTransientSnapshots{M,Mode1Axis},
  v,ispace,itime,iparam
  ) where M

  nt = num_times(s)
  np = num_params(s)
  icolumn = column_index(itime,iparam,nt,np)
  s.values[ispace,icolumn] = v
end
function tensor_setindex!(
  s::CompressedTransientSnapshots{M,Mode2Axis},
  v,ispace,itime,iparam
  ) where M

  ns = num_space_dofs(s)
  np = num_params(s)
  icolumn = column_index(ispace,iparam,ns,np)
  s.values[itime,icolumn] = v
end

# convert to vector of ParamArrays
function as_param_arrays(s::CompressedTransientSnapshots,values::AbstractMatrix{T}) where T
  np = num_params(s)
  nt = num_times(s)
  map(1:nt) do it
    param_idx = (it-1)*np+1:it*np
    array = Vector{Vector{T}}(undef,np)
    for (i,itp) = enumerate(param_idx)
      array[i] = values[:,itp]
    end
    ParamArray(array)
  end
end

function recast(a::AbstractMatrix,s::CompressedTransientSnapshots{Mode1Axis})
  @fastmath recast_values = a*s
  param_array_values = as_param_arrays(s,recast_values)
  Snapshots(param_array_values,s.realization,Mode1Axis())
end

function recast_compress(a::AbstractMatrix,s::AbstractTransientSnapshots{Mode1Axis})
  s_compress = compress(a,s)
  s_recast_compress = recast(a,s_compress)
  s_recast_compress
end

struct TransientSnapshotsWithDirichletValues{M,T,P,S} <: AbstractTransientSnapshots{M,T}
  snaps::S
  dirichlet_values::P
  function TransientSnapshotsWithDirichletValues(
    snaps::AbstractTransientSnapshots{M,T},
    dirichlet_values::P
    ) where {M,T,P<:AbstractParamContainer}

    S = typeof(snaps)
    new{M,T,P,S}(snaps,dirichlet_values)
  end
end

num_space_dofs(s::TransientSnapshotsWithDirichletValues) = num_space_free_dofs(s.snaps) + num_space_dirichlet_dofs(s)
num_space_free_dofs(s::TransientSnapshotsWithDirichletValues) = num_space_dofs(s.snaps)
num_space_dirichlet_dofs(s::TransientSnapshotsWithDirichletValues) = length(first(s.dirichlet_values))

function change_mode(s::TransientSnapshotsWithDirichletValues)
  TransientSnapshotsWithDirichletValues(change_mode(s.snaps),s.dirichlet_values)
end

function tensor_getindex(s::TransientSnapshotsWithDirichletValues,ispace::Integer,itime,iparam)
  if ispace > num_space_free_dofs(s)
    ispace_dir = ispace-num_space_free_dofs(s)
    s.dirichlet_values[itime][iparam][ispace_dir]
  else
    tensor_getindex(s.snaps,ispace,itime,iparam)
  end
end

function tensor_setindex!(s::TransientSnapshotsWithDirichletValues,v,ispace,itime,iparam)
  if ispace > num_space_free_dofs(s)
    s.dirichlet_values[itime][iparam][ispace-num_space_free_dofs(s)] = v
  else
    tensor_setindex!(s.snaps,v,ispace,itime,iparam)
  end
end

struct SelectedSnapshotsAtIndices{M,T,S,I} <: AbstractTransientSnapshots{M,T}
  snaps::S
  selected_indices::I
  function SelectedSnapshotsAtIndices(
    snaps::AbstractTransientSnapshots{M,T},
    selected_indices::I
    ) where {M,T,I}

    S = typeof(snaps)
    new{M,T,S,I}(snaps,selected_indices)
  end
end

function SelectedSnapshotsAtIndices(s::SelectedSnapshotsAtIndices,selected_indices)
  new_srange,new_trange,new_prange = selected_indices
  old_srange,old_trange,old_prange = s.selected_indices
  @check intersect(old_srange,new_srange) == new_srange
  @check intersect(old_trange,new_trange) == new_trange
  @check intersect(old_prange,new_prange) == new_prange
  SelectedSnapshotsAtIndices(s.snaps,selected_indices)
end

function select_snapshots(s::AbstractTransientSnapshots,spacerange,timerange,paramrange)
  srange = isa(spacerange,Colon) ? Base.OneTo(num_space_dofs(s)) : spacerange
  srange = isa(srange,Integer) ? [srange] : srange
  trange = isa(timerange,Colon) ? Base.OneTo(num_times(s)) : timerange
  trange = isa(trange,Integer) ? [trange] : trange
  prange = isa(paramrange,Colon) ? Base.OneTo(num_params(s)) : paramrange
  prange = isa(prange,Integer) ? [prange] : prange
  selected_indices = (srange,trange,prange)
  SelectedSnapshotsAtIndices(s,selected_indices)
end

function select_snapshots(s::AbstractTransientSnapshots,timerange,paramrange;spacerange=:)
  select_snapshots(s,spacerange,timerange,paramrange)
end

function select_snapshots(s::AbstractTransientSnapshots,paramrange;spacerange=:,timerange=:)
  select_snapshots(s,spacerange,timerange,paramrange)
end

function select_snapshots(s::AbstractTransientSnapshots;kwargs...)
  paramrange = isa(s,SelectedSnapshotsAtIndices) ? last(s.selected_indices) : Colon()
  select_snapshots(s,paramrange;kwargs...)
end

space_indices(s::SelectedSnapshotsAtIndices) = s.selected_indices[1]
time_indices(s::SelectedSnapshotsAtIndices) = s.selected_indices[2]
param_indices(s::SelectedSnapshotsAtIndices) = s.selected_indices[3]
num_space_dofs(s::SelectedSnapshotsAtIndices) = length(space_indices(s))
FEM.num_times(s::SelectedSnapshotsAtIndices) = length(time_indices(s))
FEM.num_params(s::SelectedSnapshotsAtIndices) = length(param_indices(s))

function change_mode(s::SelectedSnapshotsAtIndices)
  snaps = change_mode(s.snaps)
  SelectedSnapshotsAtIndices(snaps,s.selected_indices)
end

function tensor_getindex(
  s::SelectedSnapshotsAtIndices,
  ispace::Integer,itime::Integer,iparam::Integer)
  is = space_indices(s)[ispace]
  it = time_indices(s)[itime]
  ip = param_indices(s)[iparam]
  tensor_getindex(s.snaps,is,it,ip)
end

function tensor_setindex!(
  s::SelectedSnapshotsAtIndices,
  v,ispace::Integer,itime::Integer,iparam::Integer)
  is = space_indices(s)[ispace]
  it = time_indices(s)[itime]
  ip = param_indices(s)[iparam]
  tensor_setindex!(s.snaps,v,is,it,ip)
end

function FEM.get_values(s::SelectedSnapshotsAtIndices{Mode1Axis,T,<:BasicSnapshots}) where T
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

function FEM.get_values(s::SelectedSnapshotsAtIndices{M,T,<:TransientSnapshots}) where {M,T}
  get_values(BasicSnapshots(s))
end

function get_realization(s::SelectedSnapshotsAtIndices)
  r = get_realization(s.snaps)
  r[param_indices(s),time_indices(s)]
end

get_mode(s::SelectedSnapshotsAtIndices) = get_mode(s.snaps)

function BasicSnapshots(s::SelectedSnapshotsAtIndices{Mode1Axis,T,<:BasicSnapshots}) where T
  values = get_values(s)
  r = get_realization(s)
  mode = s.snaps.mode
  BasicSnapshots(mode,values,r)
end

function BasicSnapshots(s::SelectedSnapshotsAtIndices{Mode1Axis,T,<:TransientSnapshots}) where T
  @check space_indices(s) == Base.OneTo(num_space_dofs(s))
  v = s.snaps.values
  basic_values = Vector{typeof(first(first(v)))}(undef,num_cols(s))
  @inbounds for (i,it) in enumerate(time_indices(s))
    for (j,jp) in enumerate(param_indices(s))
      basic_values[(i-1)*num_params(s)+j] = v[it][jp]
    end
  end
  r = get_realization(s)
  mode = s.snaps.mode
  BasicSnapshots(mode,ParamArray(basic_values),r)
end

#= mode-1 representation of a snapshot of type TransientSnapshotsSwappedColumns
   [ [u(x1,t1,μ1) ⋯ u(x1,tT,μ1)] [u(x1,t1,μ2) ⋯ u(x1,tT,μ2)] [u(x1,t1,μ3) ⋯] [⋯] [u(x1,t1,μP) ⋯ u(x1,tT,μP)] ]
         ⋮             ⋮          ⋮            ⋮           ⋮              ⋮             ⋮
   [ [u(xN,t1,μ1) ⋯ u(xN,tT,μ1)] [u(xN,t1,μ2) ⋯ u(xN,tT,μ2)] [u(xN,t1,μ3) ⋯] [⋯] [u(xN,t1,μP) ⋯ u(xN,tT,μP)] ]
=#
#= mode-2 representation of a snapshot of type TransientSnapshotsSwappedColumns
   [ [u(x1,t1,μ1) ⋯ u(xN,t1,μ1)] [u(x1,t1,μ2) ⋯ u(xN,t1,μ2)] [u(x1,t1,μ3) ⋯] [⋯] [u(x1,t1,μP) ⋯ u(xN,t1,μP)] ]
         ⋮             ⋮          ⋮            ⋮           ⋮              ⋮             ⋮
   [ [u(x1,tT,μ1) ⋯ u(xN,tT,μ1)] [u(x1,tT,μ2) ⋯ u(xN,tT,μ2)] [u(x1,tT,μ3) ⋯] [⋯] [u(x1,tT,μP) ⋯ u(xN,tT,μP)] ]
=#

abstract type TransientSnapshotsSwappedColumns{T} <: AbstractTransientSnapshots{Mode1Axis,T} end

num_space_dofs(s::TransientSnapshotsSwappedColumns) = length(first(first(s.values)))

function col_index(s::TransientSnapshotsSwappedColumns,time_index::Integer,param_index::Integer)
  (param_index-1)*num_times(s)+time_index
end

function change_mode(s::TransientSnapshotsSwappedColumns)
  @notimplemented
end

function Base.getindex(s::TransientSnapshotsSwappedColumns,ispace,j)
  nt = num_times(s)
  iparam = slow_index(j,nt)
  itime = fast_index(j,nt)
  tensor_getindex(s,ispace,itime,iparam)
end

function Base.setindex!(s::TransientSnapshotsSwappedColumns,v,ispace,j)
  nt = num_times(s)
  iparam = slow_index(j,nt)
  itime = fast_index(j,nt)
  tensor_setindex!(s,v,ispace,itime,iparam)
end

struct InnerTimeOuterParamTransientSnapshots{T,P,R,V} <: TransientSnapshotsSwappedColumns{T}
  values::V
  realization::R
  function InnerTimeOuterParamTransientSnapshots(
    values::AbstractVector{P},
    realization::R
    ) where {P<:AbstractParamContainer,R<:TransientParamRealization}

    V = typeof(values)
    T = eltype(P)
    new{T,P,R,V}(values,realization)
  end
end

function reverse_snapshots(s::AbstractTransientSnapshots)
  @abstractmethod
end

function reverse_snapshots(s::InnerTimeOuterParamTransientSnapshots)
  s
end

function reverse_snapshots(s::SelectedSnapshotsAtIndices)
  SelectedInnerTimeOuterParamTransientSnapshots(s)
end

function reverse_snapshots(
  values::AbstractVector{P},
  r::TransientParamRealization
  ) where P<:AbstractParamContainer
  InnerTimeOuterParamTransientSnapshots(values,r)
end

function reverse_snapshots(a::ArrayContribution,args...)
  b = array_contribution()
  for (trian,values) in a.dict
    b[trian] = reverse_snapshots(values,args...)
  end
  b
end

function reverse_snapshots(
  s::BasicSnapshots{M,T,<:ParamArray{T,N,A}}) where {M,T,N,A}

  nt = num_times(s)
  np = num_params(s)
  P = ParamArray{T,N,A,nt}
  array = Vector{P}(undef,np)
  @inbounds for ip = 1:np
    it = ip:np:nt*np
    array[ip] = ParamArray(map(i->s.values[i],it))
  end
  InnerTimeOuterParamTransientSnapshots(array,s.realization)
end

function reverse_snapshots(
  s::TransientSnapshots{M,T,<:ParamArray{T,N,A}}) where {M,T,N,A}

  nt = num_times(s)
  np = num_params(s)
  P = ParamArray{T,N,A,nt}
  array = Vector{P}(undef,np)
  @inbounds for ip = 1:np
    array[ip] = ParamArray(map(i->s.values[i][ip],1:nt))
  end
  InnerTimeOuterParamTransientSnapshots(array,s.realization)
end

function reverse_snapshots_at_indices(
  s::BasicSnapshots{M,T,<:ParamArray{T}},
  indices_space::AbstractVector) where {M,T}

  nt = num_times(s)
  np = num_params(s)
  P = ParamVector{T,Vector{Vector{T}},nt}
  array = Vector{P}(undef,np)
  @inbounds for ip = 1:np
    it = ip:np:nt*np
    array[ip] = ParamArray(map(i->collect(s.values[i][indices_space]),it))
  end
  InnerTimeOuterParamTransientSnapshots(array,s.realization)
end

function reverse_snapshots_at_indices(
  s::TransientSnapshots{M,T,<:ParamArray{T}},
  indices_space::AbstractVector) where {M,T}

  nt = num_times(s)
  np = num_params(s)
  P = ParamVector{T,Vector{Vector{T}},nt}
  array = Vector{P}(undef,np)
  @inbounds for ip = 1:np
    array[ip] = ParamArray(map(i->collect(s.values[i][ip][indices_space]),1:nt))
  end
  InnerTimeOuterParamTransientSnapshots(array,s.realization)
end

function tensor_getindex(
  s::InnerTimeOuterParamTransientSnapshots,
  ispace::Integer,itime::Integer,iparam::Integer)
  s.values[iparam][itime][ispace]
end

function tensor_setindex!(
  s::InnerTimeOuterParamTransientSnapshots,
  v,ispace::Integer,itime::Integer,iparam::Integer)
  s.values[iparam][itime][ispace] = v
end

function SelectedSnapshotsAtIndices(
  s::InnerTimeOuterParamTransientSnapshots,selected_indices::Tuple)
  SelectedInnerTimeOuterParamTransientSnapshots(s,selected_indices)
end

struct SelectedInnerTimeOuterParamTransientSnapshots{T,S,I} <: TransientSnapshotsSwappedColumns{T}
  snaps::S
  selected_indices::I
  function SelectedInnerTimeOuterParamTransientSnapshots(
    snaps::TransientSnapshotsSwappedColumns{T},
    selected_indices::I
    ) where {T,I}

    S = typeof(snaps)
    new{T,S,I}(snaps,selected_indices)
  end
end

function SelectedInnerTimeOuterParamTransientSnapshots(s::SelectedSnapshotsAtIndices)
  sswap = reverse_snapshots(s.snaps)
  SelectedInnerTimeOuterParamTransientSnapshots(sswap,s.selected_indices)
end

space_indices(s::SelectedInnerTimeOuterParamTransientSnapshots) = s.selected_indices[1]
time_indices(s::SelectedInnerTimeOuterParamTransientSnapshots) = s.selected_indices[2]
param_indices(s::SelectedInnerTimeOuterParamTransientSnapshots) = s.selected_indices[3]
num_space_dofs(s::SelectedInnerTimeOuterParamTransientSnapshots) = length(space_indices(s))
FEM.num_times(s::SelectedInnerTimeOuterParamTransientSnapshots) = length(time_indices(s))
FEM.num_params(s::SelectedInnerTimeOuterParamTransientSnapshots) = length(param_indices(s))

function tensor_getindex(
  s::SelectedInnerTimeOuterParamTransientSnapshots,
  ispace::Integer,itime::Integer,iparam::Integer)
  is = space_indices(s)[ispace]
  it = time_indices(s)[itime]
  ip = param_indices(s)[iparam]
  tensor_getindex(s.snaps,is,it,ip)
end

function tensor_setindex!(
  s::SelectedInnerTimeOuterParamTransientSnapshots,
  v,ispace::Integer,itime::Integer,iparam::Integer)
  is = space_indices(s)[ispace]
  it = time_indices(s)[itime]
  ip = param_indices(s)[iparam]
  tensor_setindex!(s.snaps,v,is,it,ip)
end

function FEM.get_values(s::SelectedInnerTimeOuterParamTransientSnapshots)
  @check space_indices(s) == Base.OneTo(num_space_dofs(s))
  v = get_values(s.snaps)
  values = Vector{typeof(first(v))}(undef,num_params(s))
  @inbounds for (i,ip) in enumerate(param_indices(s))
    values[i] = ParamArray([v[ip][jt] for jt in time_indices(s)])
  end
  values
end

function get_realization(s::SelectedInnerTimeOuterParamTransientSnapshots)
  r = get_realization(s.snaps)
  r[param_indices(s),time_indices(s)]
end

get_mode(s::SelectedInnerTimeOuterParamTransientSnapshots) = get_mode(s.snaps)

const BasicNnzSnapshots = BasicSnapshots{M,T,P,R} where {M,T,P<:SparseParamMatrix,R}
const TransientNnzSnapshots = TransientSnapshots{M,T,P,R} where {M,T,P<:SparseParamMatrix,R}
const SelectedNnzSnapshotsAtIndices = SelectedSnapshotsAtIndices{M,T,S,I} where {M,T,S<:Union{BasicNnzSnapshots,TransientNnzSnapshots},I}
const GenericNnzSnapshots = Union{
  BasicNnzSnapshots{M,T},
  TransientNnzSnapshots{M,T},
  SelectedNnzSnapshotsAtIndices{M,T}} where {M,T}

const InnerTimeOuterParamTransientNnzSnapshots = InnerTimeOuterParamTransientSnapshots{T,P,R} where {T,P<:SparseParamMatrix,R}
const SelectedInnerTimeOuterParamTransientNnzSnapshots = SelectedInnerTimeOuterParamTransientSnapshots{T,S} where {T,S<:InnerTimeOuterParamTransientNnzSnapshots}
const NnzSnapshotsSwappedColumns = Union{
  InnerTimeOuterParamTransientNnzSnapshots{T},
  SelectedInnerTimeOuterParamTransientNnzSnapshots{T}
} where T

const NnzSnapshots = Union{
  GenericNnzSnapshots{M,T},
  NnzSnapshotsSwappedColumns{T}} where {M,T}

num_space_dofs(s::BasicNnzSnapshots) = nnz(first(s.values))

function tensor_getindex(s::BasicNnzSnapshots,ispace::Integer,itime::Integer,iparam::Integer)
  nonzeros(s.values[iparam+(itime-1)*num_params(s)])[ispace]
end

function tensor_setindex!(s::BasicNnzSnapshots,v,ispace::Integer,itime::Integer,iparam::Integer)
  nonzeros(s.values[iparam+(itime-1)*num_params(s)])[ispace] = v
end

num_space_dofs(s::TransientNnzSnapshots) = nnz(first(first(s.values)))

function tensor_getindex(
  s::TransientNnzSnapshots,
  ispace::Integer,itime::Integer,iparam::Integer)
  nonzeros(s.values[itime][iparam])[ispace]
end

function tensor_setindex!(
  s::TransientNnzSnapshots,
  v,ispace::Integer,itime::Integer,iparam::Integer)
  nonzeros(s.values[itime][iparam])[ispace] = v
end

function tensor_getindex(
  s::InnerTimeOuterParamTransientNnzSnapshots,
  ispace::Integer,itime::Integer,iparam::Integer)
  nonzeros(s.values[iparam][itime])[ispace]
end

function tensor_setindex!(
  s::InnerTimeOuterParamTransientNnzSnapshots,
  v,ispace::Integer,itime::Integer,iparam::Integer)
  nonzeros(s.values[iparam][itime])[ispace] = v
end

function tensor_getindex(
  s::SelectedInnerTimeOuterParamTransientNnzSnapshots,
  ispace::Integer,itime::Integer,iparam::Integer)
  snaps = s.snaps
  is = space_indices(s)[ispace]
  it = time_indices(s)[itime]
  ip = param_indices(s)[iparam]
  nonzeros(snaps.values[ip][it])[is]
end

function tensor_setindex!(
  s::SelectedInnerTimeOuterParamTransientNnzSnapshots,
  v,ispace::Integer,itime::Integer,iparam::Integer)
  snaps = s.snaps
  is = space_indices(s)[ispace]
  it = time_indices(s)[itime]
  ip = param_indices(s)[iparam]
  tensor_setindex!(snaps,v,is,it,ip)
end

function get_nonzero_indices(s::NnzSnapshots)
  v = isa(s,BasicSnapshots) ? first(s.values) : first(first(s.values))
  i,j, = findnz(v)
  return i .+ (j .- 1)*v.m
end

function recast(a::AbstractMatrix,s::NnzSnapshots{Mode1Axis})
  s1 = first(s.values)
  r = get_realization(s)
  r1 = r[1,axes(a,2)]
  i,j, = findnz(s1)
  m,n = size(s1)
  asparse = map(eachcol(a)) do v
    sparse(i,j,v,m,n)
  end
  return Snapshots(ParamArray(asparse),r1,s.mode)
end

const BasicBlockSnapshots = BasicSnapshots{M,T,P,R} where {M,T,P<:ParamBlockArray,R}
const TransientBlockSnapshots = TransientSnapshots{M,T,P,R} where {M,T,P<:ParamBlockArray,R}
const SelectedBlockSnapshotsAtIndices = SelectedSnapshotsAtIndices{M,T,S,I} where {M,T,S<:Union{BasicBlockSnapshots,TransientBlockSnapshots},I}
const GenericBlockSnapshots = Union{
  BasicBlockSnapshots{M,T},
  TransientBlockSnapshots{M,T},
  SelectedBlockSnapshotsAtIndices{M,T}} where {M,T}

const InnerTimeOuterParamTransientBlockSnapshots = InnerTimeOuterParamTransientSnapshots{T,P,R} where {T,P<:ParamBlockArray,R}
const SelectedInnerTimeOuterParamTransientBlockSnapshots = SelectedInnerTimeOuterParamTransientSnapshots{T,S} where {T,S<:InnerTimeOuterParamTransientBlockSnapshots}
const BlockSnapshotsSwappedColumns = Union{
  InnerTimeOuterParamTransientBlockSnapshots{T},
  SelectedInnerTimeOuterParamTransientBlockSnapshots{T}
} where T

const BlockSnapshots = Union{
  GenericBlockSnapshots{M,T},
  BlockSnapshotsSwappedColumns{T}} where {M,T}

BlockArrays.blocklength(s::BlockSnapshots) = blocklength(_get_values(s))

_get_values(s::BlockSnapshots) = s.values # just here as a helper
_get_snaps(s::BlockSnapshots) = s.snaps # just here as a helper
function _get_offsets(blocks::AbstractArray) # just here as a helper
  offsets = zeros(Int,length(blocks))
  for (n,block) in enumerate(blocks[2:end])
    offsets[n+1] = offsets[n] + num_space_dofs(block[n])
  end
  return offsets
end
function _get_selected_indices(blocks::AbstractArray{<:AbstractTransientSnapshots}) # just here as a helper
  s1 = first(blocks)
  _,ids_time,ids_param = s1.selected_indices
  offsets = _get_offsets(blocks)
  ids_space = map(blocks,offsets) do s,offset
    _ids_space,_ids_time,_ids_param = s.selected_indices
    @check ids_time == _ids_time
    @check ids_param == _ids_param
    _ids_space .+ offset
  end
  ids_space,ids_time,ids_param
end
function _get_selected_indices(s::BlockSnapshots) # just here as a helper
  get_blocks(s::BasicSnapshots) = blocks(first(s.values))
  get_blocks(s::TransientSnapshots) = blocks(first(first(s.values)))
  val_blocks = get_blocks(s.snaps)
  ax = map(x->axes(x,1),val_blocks)
  offsets = zeros(Int,length(val_blocks))
  for (n,block) in enumerate(val_blocks[2:end])
    offsets[n+1] = offsets[n] + size(block[n],1)
  end
  _ids_space,ids_time,ids_param = s.selected_indices
  map(ax,offsets) do axis,offset
    ids_space = intersect(axis,_ids_space .- offset)
    ids_space,ids_time,ids_param
  end
end

function BlockArrays.blocks(s::BasicBlockSnapshots)
  map(blocks(s.values)) do values
    BasicSnapshots(s.mode,values,s.realization)
  end
end

function BlockArrays.mortar(blocks::AbstractArray{<:BasicSnapshots})
  s1 = first(blocks)
  BasicSnapshots(mortar(map(_get_values,blocks)),s1.realization,s1.mode)
end

function BlockArrays.blocks(s::TransientBlockSnapshots)
  block_values = map(blocks,s.values)
  nblocks = blocklength(first(s.values))
  map(1:nblocks) do i
    TransientSnapshots(s.mode,map(x->getindex(x,i),block_values),s.realization)
  end
end

function BlockArrays.mortar(blocks::AbstractArray{<:TransientSnapshots})
  s1 = first(blocks)
  TransientSnapshots(map(mortar,map(_get_values,blocks)),s1.realization,s1.mode)
end

function BlockArrays.blocks(s::SelectedBlockSnapshotsAtIndices)
  selected_indices = _get_selected_indices(s)
  map(blocks(s.snaps),selected_indices) do snaps,selected_indices
    SelectedSnapshotsAtIndices(snaps,selected_indices)
  end
end

function BlockArrays.mortar(blocks::AbstractArray{<:SelectedSnapshotsAtIndices})
  selected_indices = _get_selected_indices(blocks)
  SelectedSnapshotsAtIndices(mortar(map(_get_snaps,blocks)),selected_indices)
end

function BlockArrays.blocks(s::InnerTimeOuterParamTransientBlockSnapshots{T,<:ParamBlockArray}) where T
  block_values = map(blocks,s.values)
  nblocks = blocklength(first(s.values))
  map(1:nblocks) do i
    InnerTimeOuterParamTransientSnapshots(map(x->getindex(x,i),block_values),s.realization)
  end
end

function BlockArrays.mortar(blocks::AbstractArray{<:InnerTimeOuterParamTransientSnapshots})
  s1 = first(blocks)
  InnerTimeOuterParamTransientSnapshots(map(mortar,map(_get_values,blocks)),s1.realization)
end

function BlockArrays.blocks(s::SelectedInnerTimeOuterParamTransientBlockSnapshots)
  selected_indices = _get_selected_indices(s)
  map(blocks(s.snaps),selected_indices) do snaps,selected_indices
    SelectedInnerTimeOuterParamTransientSnapshots(snaps,selected_indices)
  end
end

function BlockArrays.mortar(blocks::AbstractArray{<:SelectedInnerTimeOuterParamTransientSnapshots})
  selected_indices = _get_selected_indices(blocks)
  SelectedInnerTimeOuterParamTransientSnapshots(mortar(map(_get_snaps,blocks)),selected_indices)
end

const AbstractSubTransientSnapshots = Union{
  AbstractTransientSnapshots{M,T},
  SubArray{T,2,AbstractTransientSnapshots{M,T}}
  } where {M,T}

function (*)(
  a::Adjoint{<:Any,<:AbstractTransientSnapshots},
  b::AbstractMatrix
  )

  (*)(collect(a),b)
end

function (*)(
  a::Adjoint{<:Any,<:AbstractVecOrMat},
  b::AbstractTransientSnapshots
  )

  (*)(a,collect(b))
end

function (*)(
  a::Adjoint{<:Any,<:AbstractTransientSnapshots},
  b::AbstractTransientSnapshots
  )

  (*)(collect(a),collect(b))
end

function (*)(
  a::Adjoint{<:Any,<:AbstractTransientSnapshots},
  b::Diagonal
  )

  (*)(collect(a),b)
end

function (*)(
  a::Diagonal,
  b::AbstractTransientSnapshots
  )

  (*)(a,collect(b))
end

for op in (:*,:\)
  @eval begin
    function ($op)(
      a::A,
      b::AbstractMatrix
      ) where A<:AbstractSubTransientSnapshots

      ($op)(a,collect(b))
    end

    function ($op)(
      a::AbstractMatrix,
      b::A
      ) where A<:AbstractSubTransientSnapshots

      ($op)(a,collect(b))
    end

    function ($op)(
      a::A,
      b::B
      ) where {A<:AbstractSubTransientSnapshots,B<:AbstractSubTransientSnapshots}

      ($op)(collect(a),collect(b))
    end
  end
end
