function collect_solutions(
  rbinfo::RBInfo,
  solver::ODESolver,
  op::TransientParamFEOperator,
  uh0::Function;
  kwargs...)

  nparams = rbinfo.nsnaps
  sol = solve(solver,op,uh0;nparams)
  iv = sol.odesol.u0
  r = sol.odesol.r

  trial = evaluate(get_trial(op),r)
  V = get_vector_type(trial)
  fv = V[]

  stats = @timed for (uht,rt) in sol
    push!(fv,uht.free_values)
  end
  snaps = Snapshots(fv,iv,r)
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
Base.IndexStyle(::Type{<:AbstractTransientSnapshots}) = IndexLinear()

num_space_dofs(s::AbstractTransientSnapshots) = length(first(first(s.values)))
num_times(::AbstractTransientSnapshots) = num_times(s.realization)
num_params(::AbstractTransientSnapshots) = num_params(s.realization)
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

function Base.show(io::IO,::MIME"text/plain",s::AbstractTransientSnapshots{M}) where M
  println(io, "Transient snapshot matrix of size $(size(s)) ordered in $M representation")
end

function Base.getindex(s::AbstractTransientSnapshots{Mode1Axis},ispace::Int,j::Int)
  np = num_params(s)
  itime = Int(floor((j-1)/np) + 1)
  iparam = mod(j-1,np) + 1
  tensor_getindex(s,ispace,itime,iparam)
end

function Base.getindex(s::AbstractTransientSnapshots{Mode2Axis},itime::Int,j::Int)
  np = num_params(s)
  ispace = mod(j-1,np) + 1
  iparam = Int(floor((j-1)/np) + 1)
  tensor_getindex(s,ispace,itime,iparam)
end

struct TransientSnapshotsWithInitialValues{M,T,R} <: AbstractTransientSnapshots{M,T}
  mode::M
  values::AbstractVector{T}
  initial_values::T
  realization::R
  function TransientSnapshotsWithInitialValues(
    mode::M,
    values::AbstractVector{T},
    initial_values::T,
    realization::R
    ) where {M,T<:AbstractParamContainer,R<:TransientParamRealization}

    new{M,T,R}(mode,values,initial_values,realization)
  end
end

function Snapshots(
  values::AbstractVector{T},
  initial_values::T,
  realization::R,
  mode::M=Mode1Axis()
  ) where {M,T,R}

  TransientSnapshotsWithInitialValues(mode,values,initial_values,realization)
end

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

function Base.view(
  s::TransientSnapshotsWithInitialValues,
  rowrange::Base.Slice{Base.OneTo{Ti}},
  colrange::UnitRange{Ti}) where Ti

  # colrange refers exclusively to the parameter
  rrange = s.realization[colrange]
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

struct TransientSnapshots{M,T,R} <: AbstractTransientSnapshots{M,T}
  mode::M
  values::AbstractVector{T}
  realization::R
  function TransientSnapshots(
    mode::M,
    values::AbstractVector{T},
    realization::R
    ) where {M,T<:AbstractParamContainer,R<:TransientParamRealization}

    new{M,T,R}(mode,values,realization)
  end
end

function Snapshots(
  values::AbstractVector{T},
  realization::R,
  mode::M=Mode1Axis()
  ) where {M,T,R}

  TransientSnapshots(mode,values,realization)
end

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

function Base.view(
  s::TransientSnapshots,
  rowrange::Base.Slice{Base.OneTo{Ti}},
  colrange::UnitRange{Ti}) where Ti

  # colrange refers exclusively to the parameter
  rrange = s.realization[colrange]
  TransientSnapshots(s.mode,s.values,rrange)
end
