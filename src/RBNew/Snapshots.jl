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

abstract type AbstractTransientSnapshots{T} <: AbstractParamContainer{T,2} end

Base.ndims(::AbstractTransientSnapshots) = 2
Base.ndims(::Type{<:AbstractTransientSnapshots}) = 2
Base.IndexStyle(::Type{<:AbstractTransientSnapshots}) = IndexLinear()

struct TransientSnapshotsWithInitialValues{M,T,R} <: AbstractTransientSnapshots{T}
  mode::M
  values::AbstractVector{T}
  initial_values::T
  realization::R
  function TransientSnapshotsWithInitialValues(
    values::AbstractVector{T},
    initial_values::T,
    realization::R,
    mode::M=Mode1Axis()
    ) where {M,T<:AbstractParamContainer,R<:TransientParamRealization}

    new{M,T,R}(mode,values,initial_values,realization)
  end
end

num_space_dofs(s::TransientSnapshots) = length(first(s.initial_values))
num_times(::TransientSnapshots) = num_times(s.realization)
num_params(::TransientSnapshots) = num_params(s.realization)
get_mode(s::TransientSnapshots) = s.mode
row_size(s::TransientSnapshots{Mode1Axis}) = num_space_dofs(s)
row_size(s::TransientSnapshots{Mode2Axis}) = num_times(s)
col_size(s::TransientSnapshots{Mode1Axis}) = num_times(s)*num_params(s)
col_size(s::TransientSnapshots{Mode2Axis}) = num_space_dofs(s)*num_params(s)

# space_axis(s::TransientSnapshots) = Base.OneTo(num_space_dofs(s))
# time_axis(s::TransientSnapshots) = Base.OneTo(num_times(s))
# param_axis(s::TransientSnapshots) = 1:num_times(s):length(s)

Base.length(s::TransientSnapshots) = row_size(s)*col_size(s)
Base.size(s::TransientSnapshots) = (row_size(s),col_size(s))
Base.axes(s::TransientSnapshots) = Base.OneTo.(size(s))
Base.eltype(::TransientSnapshots{M,T}) where {M,T} = T
Base.eltype(::Type{<:TransientSnapshots{M,T}}) where {M,T} = T

function change_mode!(s::TransientSnapshots{Mode1Axis})
  TransientSnapshots(s.values,s.initial_values,s.realization,Mode2Axis())
end

function change_mode!(s::TransientSnapshots{Mode2Axis})
  TransientSnapshots(s.values,s.initial_values,s.realization,Mode1Axis())
end

function Base.show(io::IO,::MIME"text/plain",s::TransientSnapshots{M}) where M
  println(io, "Transient snapshot matrix of size $(size(s)) ordered in $M representation")
end

function tensor_getindex(s::TransientSnapshots,ispace::Int,itime::Int,iparam::Int)
  if itime == 0
    s.initial_values[iparam][ispace]
  else
    s.values[itime][iparam][ispace]
  end
end

function Base.getindex(s::TransientSnapshots{Mode1Axis},ispace::Int,j::Int)
  np = num_params(s)
  itime = Int(floor((j-1)/np) + 1)
  iparam = mod(j-1,np) + 1
  tensor_getindex(s,ispace,itime,iparam)
end

function Base.getindex(s::TransientSnapshots{Mode2Axis},itime::Int,j::Int)
  np = num_params(s)
  ispace = mod(j-1,np) + 1
  iparam = Int(floor((j-1)/np) + 1)
  tensor_getindex(s,ispace,itime,iparam)
end

function Base.view(
  s::TransientSnapshots,
  rowrange::Base.Slice{Base.OneTo{Ti}},
  colrange::UnitRange{Ti})

  # colrange refers exclusively to the parameter
  rrange = s.realization[colrange]
  TransientSnapshots(s.values,s.initial_values,rrange)
end

function shift_time!(s::TransientSnapshots,δ::Number)
  r = s.realization
  shift_time!(r,δ)
end

# struct TransientSnapshotsView{M,T,R,Ti} <: SubArray{T,1,TransientSnapshots{M,T,R},Tuple{Base.Slice{Base.OneTo{Ti}},UnitRange{Ti}},true}
#   snapshots::TransientSnapshots{M,T,R}
#   paramrange::UnitRange{Ti}
# end

# num_space_dofs(s::TransientSnapshotsView) = num_space_dofs(s.snapshots)
# num_times(::TransientSnapshotsView) = num_times(s.snapshots)
# num_params(::TransientSnapshotsView) = length(s.paramrange)

# Base.length(s::TransientSnapshotsView{Mode1Axis})  = num_times(s)*num_params(s)
# Base.size(s::TransientSnapshotsView{Mode1Axis}) = (num_space_dofs(s),num_times(s)*num_params(s))
# Base.length(s::TransientSnapshotsView{Mode2Axis}) = num_space_dofs(s)*num_params(s)
# Base.size(s::TransientSnapshotsView{Mode2Axis}) = (num_times(s),num_space_dofs(s)*num_params(s))
# Base.axes(s::TransientSnapshotsView) = Base.OneTo.(size(s))
# Base.eltype(::TransientSnapshotsView{M,T}) where {M,T} = T
# Base.eltype(::Type{<:TransientSnapshotsView{M,T}}) where {M,T} = T
# Base.ndims(::TransientSnapshotsView) = 2
# Base.ndims(::Type{<:TransientSnapshotsView}) = 2
# Base.IndexStyle(::Type{<:TransientSnapshotsView}) = IndexLinear()

# function change_mode!(s::TransientSnapshotsView)
#   snapshots = change_mode!(s.snapshots)
#   TransientSnapshotsView(snapshots,s.paramrange)
# end

# function Base.show(io::IO,::MIME"text/plain",s::TransientSnapshotsView{M}) where M
#   println(io, "Transient snapshot view matrix of size $(size(s)) ordered in $M representation")
# end

# function Base.getindex(s::TransientSnapshotsView{Mode1Axis},ispace::Int,j::Int)
#   np = num_params(s)
#   itime = Int(floor((j-1)/np) + 1)
#   iparam = mod(j-1,np) + 1
#   tensor_getindex(s,ispace,itime,iparam)
# end

# function Base.getindex(s::TransientSnapshotsView{Mode2Axis},itime::Int,j::Int)
#   np = num_params(s)
#   ispace = mod(j-1,np) + 1
#   iparam = Int(floor((j-1)/np) + 1)
#   tensor_getindex(s,ispace,itime,iparam)
# end

function reduced_basis(
  rbinfo::RBInfo,
  feop::TransientParamFEOperator,
  s::TransientSnapshots)

  ϵ = rbinfo.ϵ
  nsnaps_state = rbinfo.nsnaps_state
  norm_matrix = get_norm_matrix(rbinfo,feop)
  return reduced_basis(s,norm_matrix;ϵ,nsnaps_state)
end

function reduced_basis(
  s::TransientSnapshots,
  norm_matrix;
  nsnaps_state=50,
  kwargs...)

  if size(s,1) < size(s,2)
    change_mode!(s)
  end
  sview = view(s,:,1:nsnaps_state)
  b1 = tpod(sview,norm_matrix;kwargs...)
  compressed_sview = b1'*sview
  change_mode!(compressed_sview)
  b2 = tpod(compressed_sview;kwargs...)
  if get_mode(s) == Mode1Axis()
    basis_space = b1
    basis_time = b2
  else
    basis_space = b2
    basis_time = b1
  end
  return basis_space,basis_time
end
