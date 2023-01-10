abstract type RBVarOperator{Top,Ttr} end
abstract type RBLinOperator{Top} <: RBVarOperator{Top,nothing} end
abstract type RBBilinOperator{Top,Ttr} <: RBVarOperator{Top,Ttr} end
abstract type RBLiftingOperator{Top,Ttr} <: RBLinOperator{Top} end

mutable struct RBSteadyLinOperator{Top} <: RBLinOperator{Top}
  feop::ParamSteadyLinOperator{Top}
  rbspace_row::RBSpaceSteady
end

mutable struct RBUnsteadyLinOperator{Top} <: RBLinOperator{Top}
  feop::ParamUnsteadyLinOperator{Top}
  rbspace_row::RBSpaceUnsteady
end

mutable struct RBSteadyBilinOperator{Top,Ttr} <: RBBilinOperator{Top,Ttr}
  feop::ParamSteadyBilinOperator{Top,Ttr}
  rbspace_row::RBSpaceSteady
  rbspace_col::RBSpaceSteady
end

mutable struct RBUnsteadyBilinOperator{Top,Ttr} <: RBBilinOperator{Top,Ttr}
  feop::ParamUnsteadyBilinOperator{Top,Ttr}
  rbspace_row::RBSpaceUnsteady
  rbspace_col::RBSpaceUnsteady
end

mutable struct RBSteadyLiftingOperator{Top,Ttr} <: RBLiftingOperator{Top,Ttr}
  feop::ParamSteadyLiftingOperator{Top,Ttr}
  rbspace_row::RBSpaceSteady

  function RBSteadyLiftingOperator(
    feop::ParamSteadyLiftingOperator{Top,Ttr},
    rbspace_row::RBSpaceSteady) where {Top,Ttr}
    new{Top,Ttr}(feop,rbspace_row)
  end
end

mutable struct RBUnsteadyLiftingOperator{Top,Ttr} <: RBLiftingOperator{Top,Ttr}
  feop::ParamUnsteadyLiftingOperator{Top,Ttr}
  rbspace_row::RBSpaceUnsteady

  function RBUnsteadyLiftingOperator(
    feop::ParamUnsteadyLiftingOperator{Top,Ttr},
    rbspace_row::RBSpaceUnsteady) where {Top,Ttr}
    new{Top,Ttr}(feop,rbspace_row)
  end
end

function RBVarOperator(
  feop::ParamSteadyLinOperator{Top},
  rbspace_row::RBSpaceSteady) where Top

  RBSteadyLinOperator{Top}(feop,rbspace_row)
end

function RBVarOperator(
  feop::ParamUnsteadyLinOperator{Top},
  rbspace_row::RBSpaceUnsteady) where Top

  RBUnsteadyLinOperator{Top}(feop,rbspace_row)
end

function RBVarOperator(
  feop::ParamSteadyBilinOperator{Top,Ttr},
  rbspace_row::RBSpaceSteady,
  rbspace_col::RBSpaceSteady) where {Top,Ttr}

  RBSteadyBilinOperator{Top,Ttr}(feop,rbspace_row,rbspace_col)
end

function RBVarOperator(
  feop::ParamUnsteadyBilinOperator{Top,Ttr},
  rbspace_row::RBSpaceUnsteady,
  rbspace_col::RBSpaceUnsteady) where {Top,Ttr}

  RBUnsteadyBilinOperator{Top,Ttr}(feop,rbspace_row,rbspace_col)
end

function RBLiftingOperator(op::RBSteadyBilinOperator)
  feop = ParamLiftingOperator(get_background_feop(op))
  rbs = get_rbspace_row(op)
  rbs_lift = RBSpace(get_id(rbs)*:_lift,get_basis_space(rbs))
  RBSteadyLiftingOperator(feop,rbs_lift)
end

function RBLiftingOperator(op::RBUnsteadyBilinOperator)
  feop = ParamLiftingOperator(get_background_feop(op))
  rbs = get_rbspace_row(op)
  rbs_lift = RBSpace(get_id(rbs)*:_lift,get_basis_space(rbs),get_basis_time(rbs))
  RBUnsteadyLiftingOperator(feop,rbs_lift)
end

const RBSteadyVarOperator{Top,Ttr} =
  Union{RBSteadyLinOperator{Top},RBSteadyBilinOperator{Top,Ttr},RBSteadyLiftingOperator{Top,Ttr}}
const RBUnsteadyVarOperator{Top,Ttr} =
  Union{RBUnsteadyLinOperator{Top},RBUnsteadyBilinOperator{Top,Ttr},RBUnsteadyLiftingOperator{Top,Ttr}}

get_background_feop(rbop::RBVarOperator) = rbop.feop
get_param_function(op::RBVarOperator) = get_param_function(op.feop)
get_fe_function(op::RBVarOperator) = get_fe_function(op.feop)
get_rbspace_row(rbop::RBVarOperator) = rbop.rbspace_row
get_rbspace_col(rbop::RBBilinOperator) = rbop.rbspace_col
get_id(rbop::RBVarOperator) = get_id(get_background_feop(rbop))
get_basis_space_row(rbop::RBVarOperator) = get_basis_space(get_rbspace_row(rbop))
get_basis_space_col(rbop::RBVarOperator) = get_basis_space(get_rbspace_col(rbop))
get_tests(op::RBVarOperator) = get_tests(op.feop)
get_trials(op::RBBilinOperator) = get_trials(op.feop)
get_trials(op::RBLiftingOperator) = get_trials(op.feop)
Gridap.FESpaces.get_test(op::RBVarOperator) = get_test(op.feop)
Gridap.FESpaces.get_trial(op::RBBilinOperator) = get_trial(op.feop)
Gridap.FESpaces.get_trial(op::RBLiftingOperator) = get_trial(op.feop)
get_test_no_bc(op::RBVarOperator) = get_test_no_bc(op.feop)
get_trial_no_bc(op::RBBilinOperator) = get_trial_no_bc(op.feop)
get_trial_no_bc(op::RBLiftingOperator) = get_trial_no_bc(op.feop)

get_basis_time_row(rbop::RBVarOperator) = get_basis_time(get_rbspace_row(rbop))
get_basis_time_col(rbop::RBVarOperator) = get_basis_time(get_rbspace_col(rbop))
get_Nt(op::RBVarOperator) = get_Nt(op.rbspace_row)

get_nrows(op::RBVarOperator) = get_ns(get_rbspace_row(op))
get_nrows(op::RBUnsteadyLinOperator) = get_ns(get_rbspace_row(op))*get_nt(get_rbspace_row(op))
get_nrows(op::RBUnsteadyBilinOperator) = get_ns(get_rbspace_row(op))*get_nt(get_rbspace_row(op))
get_nrows(op::RBUnsteadyLiftingOperator) = get_ns(get_rbspace_row(op))*get_nt(get_rbspace_row(op))

function Gridap.FESpaces.get_cell_dof_ids(
  rbop::RBVarOperator,
  trian::Triangulation)
  get_cell_dof_ids(get_background_feop(rbop),trian)
end

Gridap.FESpaces.assemble_vector(op::RBLinOperator,args...) = assemble_vector(op.feop,args...)
Gridap.FESpaces.assemble_matrix(op::RBBilinOperator,args...) = assemble_matrix(op.feop,args...)
assemble_matrix_and_lifting(op::RBBilinOperator,args...) = assemble_matrix_and_lifting(op.feop,args...)

function Gridap.FESpaces.assemble_vector(op::RBLinOperator{Affine},args...)
  assemble_vector(op.feop,args...)(realization(op))
end

function Gridap.FESpaces.assemble_matrix(op::RBBilinOperator{Affine,Ttr},args...) where Ttr
  assemble_matrix(op.feop,args...)(realization(op))
end

function assemble_matrix_and_lifting(op::RBBilinOperator{Affine,Ttr},args...) where Ttr
  assemble_matrix_and_lifting(op.feop,args...)(realization(op))
end

function assemble_affine_vector(op::RBSteadyLinOperator)
  assemble_vector(op.feop)(realization(op))
end

function assemble_affine_vector(op::RBUnsteadyLinOperator)
  assemble_vector(op.feop,realization(op.feop.tinfo))(realization(op))
end

function assemble_affine_vector(op::RBSteadyLiftingOperator)
  assemble_vector(op.feop)(realization(op))
end

function assemble_affine_vector(op::RBUnsteadyLiftingOperator)
  assemble_vector(op.feop,realization(op.feop.tinfo))(realization(op))
end

function assemble_affine_matrix(op::RBSteadyBilinOperator)
  assemble_matrix(op.feop)(realization(op))
end

function assemble_affine_matrix(op::RBUnsteadyBilinOperator)
  t = realization(op.feop.tinfo)
  assemble_matrix(op.feop,t)(realization(op))
end

function assemble_affine_matrix(op::RBSteadyBilinOperator{Nonlinear,Ttr}) where Ttr
  u = realization(op.feop.tests)
  assemble_matrix(op.feop)(u)
end

function assemble_affine_matrix(op::RBUnsteadyBilinOperator{Nonlinear,Ttr}) where Ttr
  t = realization(op.feop.tinfo)
  u = realization(op.feop.tests)
  assemble_matrix(op.feop,t)(u)
end

get_dirichlet_function(op::RBVarOperator) = get_dirichlet_function(op.feop)

get_pspace(op::RBVarOperator) = get_pspace(op.feop)
realization(op::RBVarOperator) = realization(get_pspace(op))
get_time_info(op::RBUnsteadyLinOperator) = get_time_info(op.feop)
get_time_info(op::RBUnsteadyBilinOperator) = get_time_info(op.feop)
get_dt(op::RBVarOperator) = get_dt(op.feop)
get_Nt(op::RBVarOperator) = get_Nt(op.feop)
get_θ(op::RBVarOperator) = get_θ(op.feop)
get_timesθ(op::RBVarOperator) = get_timesθ(op.feop)
get_reduced_timesθ(op::RBVarOperator,idx::Vector{Int}) = get_timesθ(op)[idx]

function get_reduced_timesθ(
  op::RBUnsteadyVarOperator,
  idx::NTuple{N,Vector{Int}}) where N

  Broadcasting(i->get_reduced_timesθ(op,i))(idx)
end

function compute_in_timesθ(snaps::Snapshots,args...;kwargs...)
  id = get_id(snaps)
  snap = get_snap(snaps)
  nsnap = get_nsnap(snaps)
  Snapshots(id,compute_in_timesθ(snap,args...;kwargs...),nsnap)
end

"Small, full vector -> large, sparse vector"
function get_findnz_map(
  op::RBBilinOperator,
  q::Vector{T}) where {T<:Union{<:Param,<:FEFunction}}

  get_findnz_map(op,first(q))
end

function get_findnz_map(
  op::RBSteadyBilinOperator,
  μ::Param)

  M = assemble_matrix(op)(μ)
  first(findnz(M[:]))
end

function get_findnz_map(
  op::RBUnsteadyBilinOperator,
  μ::Param)

  dtθ = get_dt(op)*get_θ(op)
  M = assemble_matrix(op,dtθ)(μ)
  first(findnz(M[:]))
end

function get_findnz_map(
  op::RBVarOperator{Nonlinear,Ttr},
  f::FEFunction) where Ttr

  M = assemble_matrix(op)(f)
  first(findnz(M[:]))
end

"Viceversa"
function get_inverse_findnz_map(op::RBBilinOperator,q::T) where T
  findnz_map = get_findnz_map(op,q)
  inv_map(i::Int) = findall(x -> x == i,findnz_map)[1]
  inv_map
end

function unfold_spacetime(
  op::RBUnsteadyBilinOperator,
  vals::AbstractVector{T}) where T

  Ns = get_Ns(op)
  Nt = get_Nt(op)
  @assert size(vals,1) == Ns*Nt "Wrong space-time dimensions"

  space_vals = Matrix{Tv}(reshape(vals,Ns,Nt))
  time_vals = Matrix{Tv}(reshape(vals,Nt,Ns))
  space_vals,time_vals
end

function unfold_spacetime(
  op::RBUnsteadyBilinOperator,
  vals::AbstractMatrix{T}) where T

  unfold_vec(k::Int) = unfold_spacetime(op,vals[:,k])
  vals = Broadcasting(unfold_vec)(axes(vals,2))
  Matrix(first.(vals)),Matrix(last.(vals))
end

function rb_space_projection(op::RBLinOperator)
  vec = assemble_affine_vector(op)
  rbrow = get_rbspace_row(op)

  rb_space_projection(rbrow,vec)
end

function rb_space_projection(op::RBBilinOperator)
  mat = assemble_affine_matrix(op)
  rbrow = get_rbspace_row(op)
  rbcol = get_rbspace_col(op)

  rb_space_projection(rbrow,rbcol,mat)
end
