function collect_solutions(
  rbinfo::RBInfo,
  solver::ODESolver,
  op::TransientParamFEOperator,
  uh0::Function;
  kwargs...)

  nparams = num_params(rbinfo)
  sol = solve(solver,op,uh0;nparams)
  odesol = sol.odesol
  realization = odesol.r

  stats = @timed begin
    values,initial_values = collect(odesol)
  end
  snaps = Snapshots(values,initial_values,realization)
  cinfo = ComputationInfo(stats,nparams)

  return snaps,cinfo
end

#= mode-1 representation of a snapshot tensor (default)
   [ [u(x1,t1,μ1) ⋯ u(x1,t1,μP)] [u(x1,t2,μP) ⋯ u(x1,t2,μP)] [u(x1,t3,μP) ⋯] [⋯] [u(x1,tT,μ1) ⋯ u(x1,tT,μP)] ]
         ⋮             ⋮          ⋮            ⋮           ⋮              ⋮             ⋮
   [ [u(xN,t1,μ1) ⋯ u(xN,t1,μP)] [u(xN,t2,μP) ⋯ u(xN,t2,μP)] [u(xN,t3,μP) ⋯] [⋯] [u(xN,tT,μ1) ⋯ u(xN,tT,μP)] ]
=#
#= mode-2 representation of a snapshot tensor
   [ [u(x1,t1,μ1) ⋯ u(xN,t1,μP)] [u(x1,t1,μP) ⋯ u(xN,t1,μP)] [u(x1,t1,μP) ⋯] [⋯] [u(x1,t1,μ1) ⋯ u(xN,t1,μP)] ]
         ⋮             ⋮          ⋮            ⋮           ⋮              ⋮             ⋮
   [ [u(x1,tT,μ1) ⋯ u(xN,tT,μP)] [u(x1,tT,μP) ⋯ u(xN,tT,μP)] [u(x1,tT,μP) ⋯] [⋯] [u(x1,tT,μ1) ⋯ u(xN,tT,μP)] ]
=#

struct Mode1Axis end
struct Mode2Axis end

abstract type AbstractTransientSnapshots{M,T} <: AbstractParamContainer{T,2} end

Base.ndims(::AbstractTransientSnapshots) = 2
Base.ndims(::Type{<:AbstractTransientSnapshots}) = 2
Base.IndexStyle(::Type{<:AbstractTransientSnapshots}) = IndexCartesian()

FEM.num_times(s::AbstractTransientSnapshots) = num_times(s.realization)
FEM.num_params(s::AbstractTransientSnapshots) = num_params(s.realization)
get_mode(s::AbstractTransientSnapshots) = s.mode
row_size(s::AbstractTransientSnapshots{Mode1Axis}) = num_space_dofs(s)
row_size(s::AbstractTransientSnapshots{Mode2Axis}) = num_times(s)
col_size(s::AbstractTransientSnapshots{Mode1Axis}) = num_times(s)*num_params(s)
col_size(s::AbstractTransientSnapshots{Mode2Axis}) = num_space_dofs(s)*num_params(s)

Base.length(s::AbstractTransientSnapshots) = row_size(s)*col_size(s)
Base.size(s::AbstractTransientSnapshots) = (row_size(s),col_size(s))
Base.axes(s::AbstractTransientSnapshots) = Base.OneTo.(size(s))
Base.eltype(::AbstractTransientSnapshots{M,T}) where {M,T} = T
Base.eltype(::Type{<:AbstractTransientSnapshots{M,T}}) where {M,T} = T

slow_index(i,N::Int) = Int.(floor.((i .- 1) ./ N) .+ 1)
slow_index(i::Colon,::Int) = i
fast_index(i,N::Int) = mod.(i .- 1,N) .+ 1
fast_index(i::Colon,::Int) = i

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

function compress(a::AbstractMatrix,s::AbstractTransientSnapshots{Mode1Axis})
  @fastmath compressed_values = a'*s
  Snapshots(compressed_values,s.realization,Mode1Axis())
end

function compress(a::AbstractMatrix,s::AbstractTransientSnapshots{Mode2Axis})
  @fastmath compressed_values = a'*s
  compressed_realization = s.realization[:,axes(a,2)]
  Snapshots(compressed_values,compressed_realization,Mode2Axis())
end

function FESpaces.FEFunction(
  fs::SingleFieldParamFESpace,s::AbstractTransientSnapshots{Mode1Axis})
  @assert FEM.length_free_values(fs) == length(s.realization)
  free_values = _to_param_array(s.values)
  diri_values = get_dirichlet_dof_values(fs)
  FEFunction(fs,free_values,diri_values)
end

function FESpaces.FEFunction(
  fs::SingleFieldParamFESpace,s2::AbstractTransientSnapshots{Mode2Axis})
  @warn "This snapshot has a mode-2 representation, the resulting FEFunction(s) might be incorrect"
  s = change_mode!(s2)
  FEFunction(fs,s)
end

struct TransientSnapshotsWithInitialValues{M,T,P,R} <: AbstractTransientSnapshots{M,T}
  mode::M
  values::AbstractVector{P}
  initial_values::P
  realization::R
  function TransientSnapshotsWithInitialValues(
    mode::M,
    values::AbstractVector{P},
    initial_values::P,
    realization::R
    ) where {M,P<:AbstractParamContainer,R<:TransientParamRealization}

    T = eltype(P)
    new{M,T,P,R}(mode,values,initial_values,realization)
  end
end

function Snapshots(
  values::AbstractVector{P},
  initial_values::P,
  realization::R,
  mode::M=Mode1Axis()
  ) where {M,P,R}

  TransientSnapshotsWithInitialValues(mode,values,initial_values,realization)
end

function Base.copy(s::TransientSnapshotsWithInitialValues)
  TransientSnapshotsWithInitialValues(
    copy(s.mode),
    copy(s.values),
    copy(s.initial_values),
    copy(s.realization))
end

num_space_dofs(s::TransientSnapshotsWithInitialValues) = length(first(s.initial_values))

function change_mode!(s::TransientSnapshotsWithInitialValues{Mode1Axis})
  TransientSnapshotsWithInitialValues(Mode2Axis(),s.values,s.initial_values,s.realization)
end

function change_mode!(s::TransientSnapshotsWithInitialValues{Mode2Axis})
  TransientSnapshotsWithInitialValues(Mode1Axis(),s.values,s.initial_values,s.realization)
end

function tensor_getindex(s::TransientSnapshotsWithInitialValues,ispace,itime,iparam)
  if itime == 0
    s.initial_values[iparam][ispace]
  else
    s.values[itime][iparam][ispace]
  end
end

function tensor_setindex!(s::TransientSnapshotsWithInitialValues,v,ispace,itime,iparam)
  if itime == 0
    s.initial_values[iparam][ispace] = v
  else
    s.values[itime][iparam][ispace] = v
  end
end

function Base.view(
  s::TransientSnapshotsWithInitialValues,
  timerange,
  paramrange)

  rrange = s.realization[paramrange,timerange]
  TransientSnapshotsWithInitialValues(s.mode,s.values,s.initial_values,rrange)
end

function FEM.shift_time!(s::TransientSnapshotsWithInitialValues,dt::Number,θ::Number)
  mode = get_mode(s)
  v_forward = s.values
  v_backward = [s.initial_values,s.values[1:end-1]...]
  v_middle = θ*v_forward + (1-θ)*v_backward
  r = s.realization
  shift_time!(r,dt*θ)
  TransientSnapshots(mode,v_middle,r)
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
  ) where {M,P,R}

  TransientSnapshots(mode,values,realization)
end

function Base.copy(s::TransientSnapshots)
  TransientSnapshots(
    copy(s.mode),
    copy(s.values),
    copy(s.realization))
end

num_space_dofs(s::TransientSnapshots) = length(first(first(s.values)))

function change_mode!(s::TransientSnapshots{Mode1Axis})
  TransientSnapshots(s.values,s.realization,Mode2Axis())
end

function change_mode!(s::TransientSnapshots{Mode2Axis})
  TransientSnapshots(s.values,s.realization,Mode1Axis())
end

function tensor_getindex(s::TransientSnapshots,ispace,itime,iparam)
  if itime == 0
    @notimplemented
  else
    s.values[itime][iparam][ispace]
  end
end

function tensor_setindex!(s::TransientSnapshots,v,ispace,itime,iparam)
  if itime == 0
    @notimplemented
  else
    s.values[itime][iparam][ispace] = v
  end
end

function Base.view(
  s::TransientSnapshots,
  timerange,
  paramrange)

  rrange = s.realization[paramrange,timerange]
  TransientSnapshots(s.mode,s.values,rrange)
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

function Base.copy(s::CompressedTransientSnapshots)
  CompressedTransientSnapshots(
    copy(s.current_mode),
    copy(s.initial_mode),
    copy(s.values),
    copy(s.realization))
end

num_space_dofs(s::CompressedTransientSnapshots{Mode1Axis,Mode1Axis}) = size(s.values,1)
num_space_dofs(s::CompressedTransientSnapshots{Mode2Axis,Mode1Axis}) = size(s.values,1)
num_space_dofs(s::CompressedTransientSnapshots{Mode1Axis,Mode2Axis}) = Int(size(s.values,2) / num_params(s))
num_space_dofs(s::CompressedTransientSnapshots{Mode2Axis,Mode2Axis}) = Int(size(s.values,2) / num_params(s))

function change_mode!(s::CompressedTransientSnapshots{Mode1Axis})
  CompressedTransientSnapshots(Mode2Axis(),Mode1Axis(),s.values,s.realization)
end

function change_mode!(s::CompressedTransientSnapshots{Mode2Axis})
  CompressedTransientSnapshots(Mode1Axis(),Mode2Axis(),s.values,s.realization)
end

column_index(a,b,Na,Nb) = (a .- 1) .* Nb .+ b
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

function Base.copy(s::TransientSnapshotsWithDirichletValues)
  TransientSnapshotsWithDirichletValues(
    copy(s.snaps),
    copy(s.dirichlet_values))
end

num_space_dofs(s::TransientSnapshotsWithDirichletValues) = num_space_free_dofs(s.snaps) + num_space_dirichlet_dofs(s)
num_space_free_dofs(s::TransientSnapshotsWithDirichletValues) = num_space_dofs(s.snaps)
num_space_dirichlet_dofs(s::TransientSnapshotsWithDirichletValues) = length(first(s.dirichlet_values))

function change_mode!(s::TransientSnapshotsWithDirichletValues)
  TransientSnapshotsWithDirichletValues(change_mode!(s.snaps),s.dirichlet_values)
end

function tensor_getindex(s::TransientSnapshotsWithDirichletValues,ispace::Int,itime,iparam)
  if ispace > num_space_free_dofs(s)
    s.dirichlet_values[itime][iparam][ispace-num_space_free_dofs(s)]
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

function Base.view(
  s::TransientSnapshotsWithDirichletValues,
  timerange,
  paramrange)

  TransientSnapshotsWithDirichletValues(
    view(s.snaps,timerange,paramrange),
    s.dirichlet_values)
end

struct NnzTransientSnapshots{M,T,P,R} <: AbstractTransientSnapshots{M,T}
  mode::M
  values::AbstractVector{P}
  realization::R
  function NnzTransientSnapshots(
    mode::M,
    values::AbstractVector{P},
    realization::R
    ) where {M,P<:ParamArray{<:AbstractSparseMatrix},R<:TransientParamRealization}

    T = eltype(P)
    new{M,T,P,R}(mode,values,realization)
  end
end

function Base.copy(s::NnzTransientSnapshots)
  NnzTransientSnapshots(
    copy(s.mode),
    copy(s.values),
    copy(s.realization))
end

num_space_dofs(s::NnzTransientSnapshots) = nnz(first(first(s.values)))

function change_mode!(s::NnzTransientSnapshots{Mode1Axis})
  NnzTransientSnapshots(s.values,s.realization,Mode2Axis())
end

function change_mode!(s::NnzTransientSnapshots{Mode2Axis})
  NnzTransientSnapshots(s.values,s.realization,Mode1Axis())
end

function tensor_getindex(s::NnzTransientSnapshots,ispace,itime,iparam)
  if itime == 0
    @notimplemented
  else
    nonzeros(s.values[itime][iparam])[ispace]
  end
end

function tensor_setindex!(s::NnzTransientSnapshots,v,ispace,itime,iparam)
  if itime == 0
    @notimplemented
  else
    nonzeros(s.values[itime][iparam])[ispace] = v
  end
end

function Base.view(
  s::NnzTransientSnapshots,
  timerange,
  paramrange)

  rrange = s.realization[paramrange,timerange]
  NnzTransientSnapshots(s.mode,s.values,rrange)
end

# for testing / visualization purposes
function _plot(trial::TransientTrialParamFESpace,s::AbstractTransientSnapshots;dir=pwd(),varname="u")
  r = s.realization
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
