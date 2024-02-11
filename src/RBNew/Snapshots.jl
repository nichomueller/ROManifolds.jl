function collect_solutions(
  solver::RBSolver,
  op::TransientParamFEOperator,
  uh0::Function;
  kwargs...)

  info = get_info(solver)
  fesolver = get_fe_solver(solver)
  nparams = num_params(info)
  sol = solve(fesolver,op,uh0;nparams)
  odesol = sol.odesol
  realization = odesol.r

  stats = @timed begin
    values = collect(odesol)
  end
  snaps = Snapshots(values,realization)
  cinfo = ComputationInfo(stats,nparams)

  return snaps,cinfo
end

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
Base.IndexStyle(::Type{<:AbstractTransientSnapshots}) = IndexCartesian() #IndexLinear() #

FEM.get_values(s::AbstractTransientSnapshots) = s.values
get_realization(s::AbstractTransientSnapshots) = s.realization

FEM.num_times(s::AbstractTransientSnapshots) = num_times(get_realization(s))
FEM.num_params(s::AbstractTransientSnapshots) = num_params(get_realization(s))
get_mode(s::AbstractTransientSnapshots) = s.mode
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
  (mode2_index-1)*num_params(s)+param_index
end
function col_index(s::AbstractTransientSnapshots{Mode1Axis},::Colon,param_index)
  map(i->col_index(s,i,param_index),1:num_times(s))
end
function col_index(s::AbstractTransientSnapshots{Mode2Axis},::Colon,param_index)
  map(i->col_index(s,i,param_index),1:num_space_dofs(s))
end

# function Base.getindex(s::AbstractTransientSnapshots,i)
#   ncol = num_cols(s)
#   irow = slow_index(i,ncol)
#   icol = fast_index(i,ncol)
#   getindex(s,irow,icol)
# end

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

# function Base.setindex!(s::AbstractTransientSnapshots,v,i)
#   ncol = num_cols(s)
#   irow = slow_index(i,ncol)
#   icol = fast_index(i,ncol)
#   setindex!(s,v,irow,icol)
# end

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

function flatten(s::AbstractTransientSnapshots)
  get_values(BasicSnapshots(s))
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

struct TransientSnapshots{M,T,P,R} <: AbstractTransientSnapshots{M,T}
  mode::M
  values::AbstractVector{P}
  realization::R
  function TransientSnapshots(
    mode::M,
    values::AbstractVector{P},
    realization::R
    ) where {M,P<:AbstractParamContainer,R<:TransientParamRealization}

    T = eltype(P)
    new{M,T,P,R}(mode,values,realization)
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

struct CompressedTransientSnapshots{M,N,T,R} <: AbstractTransientSnapshots{M,T}
  current_mode::M
  initial_mode::N
  values::AbstractMatrix{T}
  realization::R
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

function select_snapshots(s::AbstractTransientSnapshots,paramrange,timerange;spacerange=:)
  select_snapshots(s,spacerange,timerange,paramrange)
end

function select_snapshots(s::AbstractTransientSnapshots,paramrange;spacerange=:,timerange=:)
  select_snapshots(s,spacerange,timerange,paramrange)
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

function FEM.get_values(s::SelectedSnapshotsAtIndices{Mode1Axis,T,<:TransientSnapshots}) where T
  @check space_indices(s) == Base.OneTo(num_space_dofs(s))
  v = get_values(s.snaps)
  values = Vector{typeof(first(v))}(undef,num_times(s))
  @inbounds for (i,it) in enumerate(time_indices(s))
    values[i] = ParamArray([v[it][jp] for jp in num_params(s)])
  end
  values
end

function get_realization(s::SelectedSnapshotsAtIndices)
  r = get_realization(s.snaps)
  r[param_indices(s),time_indices(s)]
end

function BasicSnapshots(s::SelectedSnapshotsAtIndices{Mode1Axis,T,<:BasicSnapshots}) where T
  values = get_values(s)
  r = get_realization(s)
  mode = s.snaps.mode
  BasicSnapshots(values,r,mode)
end

function BasicSnapshots(s::SelectedSnapshotsAtIndices{Mode1Axis,T,<:TransientSnapshots}) where T
  @check space_indices(s) == Base.OneTo(num_space_dofs(s))
  v = get_values(s.snaps)
  basic_values = Vector{typeof(first(first(v)))}(undef,num_cols(s))
  @inbounds for (i,it) in enumerate(time_indices(s))
    for (j,jp) in enumerate(param_indices(s))
      basic_values[(i-1)*num_params(s)+j] = v[it][jp]
    end
  end
  r = get_realization(s)
  BasicSnapshots(s.snaps.mode,ParamArray(basic_values),r)
end

#= mode-1 representation of a snapshot of type InnerTimeOuterParamTransientSnapshots
   [ [u(x1,t1,μ1) ⋯ u(x1,tT,μ1)] [u(x1,t1,μ2) ⋯ u(x1,tT,μ2)] [u(x1,t1,μ3) ⋯] [⋯] [u(x1,t1,μP) ⋯ u(x1,tT,μP)] ]
         ⋮             ⋮          ⋮            ⋮           ⋮              ⋮             ⋮
   [ [u(xN,t1,μ1) ⋯ u(xN,tT,μ1)] [u(xN,t1,μ2) ⋯ u(xN,tT,μ2)] [u(xN,t1,μ3) ⋯] [⋯] [u(xN,t1,μP) ⋯ u(xN,tT,μP)] ]
=#
#= mode-2 representation of a snapshot of type InnerTimeOuterParamTransientSnapshots
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

struct InnerTimeOuterParamTransientSnapshots{T,P,R} <: TransientSnapshotsSwappedColumns{T}
  values::AbstractVector{P}
  realization::R
  function InnerTimeOuterParamTransientSnapshots(
    values::AbstractVector{P},
    realization::R
    ) where {P<:AbstractParamContainer,R<:TransientParamRealization}

    T = eltype(P)
    new{T,P,R}(values,realization)
  end
end

function InnerTimeOuterParamTransientSnapshots(s::AbstractTransientSnapshots)
  @abstractmethod
end

function InnerTimeOuterParamTransientSnapshots(s::InnerTimeOuterParamTransientSnapshots)
  s
end

function InnerTimeOuterParamTransientSnapshots(s::SelectedSnapshotsAtIndices)
  SelectedInnerTimeOuterParamTransientSnapshots(s)
end

function InnerTimeOuterParamTransientSnapshots(
  s::BasicSnapshots{M,T,<:ParamArray{T,N,A}}
  ) where {M,T,N,A}

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

function InnerTimeOuterParamTransientSnapshots(
  s::TransientSnapshots{M,T,<:ParamArray{T,N,A}}
  ) where {M,T,N,A}

  nt = num_times(s)
  np = num_params(s)
  P = ParamArray{T,N,A,nt}
  array = Vector{P}(undef,np)
  @inbounds for ip = 1:np
    array[ip] = ParamArray(map(i->s.values[i][ip],1:nt))
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
  sswap = InnerTimeOuterParamTransientSnapshots(s.snaps)
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

function Snapshots(a::ArrayContribution,args...)
  b = array_contribution()
  for (trian,values) in a.dict
    b[trian] = Snapshots(values,args...)
  end
  b
end

# for testing / visualization purposes
function FESpaces.FEFunction(
  fs::SingleFieldParamFESpace,s::AbstractTransientSnapshots{Mode1Axis})
  r = get_realization(s)
  @assert FEM.length_free_values(fs) == length(r)
  free_values = _to_param_array(s.values)
  diri_values = get_dirichlet_dof_values(fs)
  FEFunction(fs,free_values,diri_values)
end

function FESpaces.FEFunction(
  fs::SingleFieldParamFESpace,s2::AbstractTransientSnapshots{Mode2Axis})
  @warn "This snapshot has a mode-2 representation, the resulting FEFunction(s) might be incorrect"
  s = change_mode(s2)
  FEFunction(fs,s)
end

function _plot(trial::TransientTrialParamFESpace,s::AbstractTransientSnapshots;dir=pwd(),varname="u")
  r = get_realization(s)
  r0 = FEM.get_at_time(r,:initial)
  times = get_times(r)
  createpvd(r0,dir) do pvd
    for (it,t) = enumerate(times)
      rt = FEM.get_at_time(r,t)
      free_values = s.values[it]
      sht = FEFunction(trial(rt),free_values)
      files = ParamString(dir,rt)
      trian = get_triangulation(sht)
      vtk = createvtk(trian,files,cellfields=[varname=>sht])
      pvd[rt] = vtk
    end
  end
end
