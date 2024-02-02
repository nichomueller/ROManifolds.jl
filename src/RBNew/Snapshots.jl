function collect_solutions(
  rbinfo::RBInfo,
  solver::ODESolver,
  op::TransientParamFEOperator,
  uh0::Function;
  kwargs...)

  nparams = num_snaps(rbinfo)
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
Base.IndexStyle(::Type{<:AbstractTransientSnapshots}) = IndexLinear()

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

# function Base.show(io::IO,::MIME"text/plain",s::AbstractTransientSnapshots{M}) where M
#   println(io, "Transient snapshot matrix of size $(size(s)) ordered in $M representation")
# end

function Base.getindex(s::AbstractTransientSnapshots{Mode1Axis},ispace::Int,j::Int)
  np = num_params(s)
  itime = Int(floor((j-1)/np) + 1)
  iparam = mod(j-1,np) + 1
  tensor_getindex(s,ispace,itime,iparam)
end

function Base.getindex(s::AbstractTransientSnapshots{Mode2Axis},itime::Int,j::Int)
  np = num_params(s)
  ispace = Int(floor((j-1)/np) + 1)
  iparam = mod(j-1,np) + 1
  tensor_getindex(s,ispace,itime,iparam)
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

function Base.view(
  s::TransientSnapshotsWithInitialValues,
  timerange::Colon,
  paramrange::UnitRange{Ti}) where Ti

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

function Base.view(
  s::TransientSnapshots,
  timerange::Colon,
  paramrange::UnitRange{Ti}) where Ti

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

function tensor_getindex(
  s::CompressedTransientSnapshots{M,Mode1Axis},
  ispace,itime,iparam
  ) where M

  np = num_params(s)
  s.values[ispace,(itime-1)*np+iparam]
end
function tensor_getindex(
  s::CompressedTransientSnapshots{M,Mode2Axis},
  ispace,itime,iparam
  ) where M

  np = num_params(s)
  s.values[itime,(ispace-1)*np+iparam]
end
