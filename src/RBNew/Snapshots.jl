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
Base.IndexStyle(::Type{<:AbstractTransientSnapshots}) = IndexCartesian()

get_realization(s::AbstractTransientSnapshots) = s.realization

FEM.num_times(s::AbstractTransientSnapshots) = num_times(get_realization(s))
FEM.num_params(s::AbstractTransientSnapshots) = num_params(get_realization(s))
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
  s = change_mode(s2)
  FEFunction(fs,s)
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

function tensor_getindex(s::BasicSnapshots,ispace,itime,iparam)
  s.values[iparam.+(itime.-1)*num_params(s)][ispace]
end

function tensor_setindex!(s::BasicSnapshots,v,ispace,itime,iparam)
  s.values[iparam.+(itime.-1)*num_params(s)][ispace] = v
end

function select_snapshots(s::BasicSnapshots,paramrange,timerange=:)
  BasicSnapshots(s.mode,s.values,s.realization[paramrange,timerange])
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

function tensor_getindex(s::TransientSnapshots,ispace,itime,iparam)
  s.values[itime][iparam][ispace]
end

function tensor_setindex!(s::TransientSnapshots,v,ispace,itime,iparam)
  s.values[itime][iparam][ispace] = v
end

function select_snapshots(s::TransientSnapshots,paramrange,timerange=:)
  TransientSnapshots(s.mode,s.values,s.realization[paramrange,timerange])
end

function TransientToBasicSnapshots(
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
  ) where {M,P<:AbstractParamContainer,R}

  TransientSnapshotsWithInitialValues(mode,values,initial_values,realization)
end

num_space_dofs(s::TransientSnapshotsWithInitialValues) = length(first(s.initial_values))

function change_mode(s::TransientSnapshotsWithInitialValues{Mode1Axis})
  TransientSnapshotsWithInitialValues(Mode2Axis(),s.values,s.initial_values,s.realization)
end

function change_mode(s::TransientSnapshotsWithInitialValues{Mode2Axis})
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

function select_snapshots(s::TransientSnapshotsWithInitialValues,paramrange,timerange=:)
  TransientSnapshotsWithInitialValues(
    s.mode,
    s.values,
    s.initial_values,
    s.realization[paramrange,timerange])
end

function FEM.shift_time!(s::TransientSnapshotsWithInitialValues,dt::Number,θ::Number)
  mode = get_mode(s)
  v_forward = s.values
  v_backward = [s.initial_values,s.values[1:end-1]...]
  v_middle = θ*v_forward + (1-θ)*v_backward
  r = s.realization
  FEM.shift_time!(r,dt*θ)
  sshift = TransientSnapshots(mode,v_middle,r)
  TransientToBasicSnapshots(sshift)
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

num_space_dofs(s::TransientSnapshotsWithDirichletValues) = num_space_free_dofs(s.snaps) + num_space_dirichlet_dofs(s)
num_space_free_dofs(s::TransientSnapshotsWithDirichletValues) = num_space_dofs(s.snaps)
num_space_dirichlet_dofs(s::TransientSnapshotsWithDirichletValues) = length(first(s.dirichlet_values))

function change_mode(s::TransientSnapshotsWithDirichletValues)
  TransientSnapshotsWithDirichletValues(change_mode(s.snaps),s.dirichlet_values)
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

function select_snapshots(s::TransientSnapshotsWithDirichletValues,paramrange,timerange=:)
  TransientSnapshotsWithDirichletValues(
    select_snapshots(s.snaps,paramrange,timerange),
    s.dirichlet_values)
end

const BasicNnzSnapshots = BasicSnapshots{M,T,P,R} where {M,T,P<:SparseParamMatrix,R}
const TransientNnzSnapshots = TransientSnapshots{M,T,P,R} where {M,T,P<:SparseParamMatrix,R}
const NnzSnapshots = Union{BasicNnzSnapshots{M,T},TransientNnzSnapshots{M,T}} where {M,T}

num_space_dofs(s::BasicNnzSnapshots) = nnz(first(s.values))

function tensor_getindex(s::BasicNnzSnapshots,ispace,itime,iparam)
  nonzeros(s.values[iparam.+(itime.-1)*num_params(s)])[ispace]
end

function tensor_setindex!(s::BasicNnzSnapshots,v,ispace,itime,iparam)
  nonzeros(s.values[iparam.+(itime.-1)*num_params(s)])[ispace] = v
end

num_space_dofs(s::TransientNnzSnapshots) = nnz(first(first(s.values)))

function tensor_getindex(s::TransientNnzSnapshots,ispace,itime,iparam)
  nonzeros(s.values[itime][iparam])[ispace]
end

function tensor_setindex!(s::TransientNnzSnapshots,v,ispace,itime,iparam)
  nonzeros(s.values[itime][iparam])[ispace] = v
end

function recast(a::AbstractMatrix,s::NnzSnapshots{Mode1Axis})
  s1 = first(s.values)
  r1 = s.realization[1,axes(a,2)]
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
