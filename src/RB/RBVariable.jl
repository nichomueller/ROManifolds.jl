abstract type RBVariable{Top,Ttr} end
abstract type RBLinVariable{Top} <: RBVariable{Top,nothing} end
abstract type RBBilinVariable{Top,Ttr} <: RBVariable{Top,Ttr} end
abstract type RBLiftVariable{Top,Ttr} <: RBLinVariable{Top} end

mutable struct RBSteadyLinVariable{Top} <: RBLinVariable{Top}
  feop::ParamSteadyLinOperator{Top}
  rbspace_row::RBSpaceSteady
end

mutable struct RBUnsteadyLinVariable{Top} <: RBLinVariable{Top}
  feop::ParamUnsteadyLinOperator{Top}
  rbspace_row::RBSpaceUnsteady
end

mutable struct RBSteadyBilinVariable{Top,Ttr} <: RBBilinVariable{Top,Ttr}
  feop::ParamSteadyBilinOperator{Top,Ttr}
  rbspace_row::RBSpaceSteady
  rbspace_col::RBSpaceSteady
end

mutable struct RBUnsteadyBilinVariable{Top,Ttr} <: RBBilinVariable{Top,Ttr}
  feop::ParamUnsteadyBilinOperator{Top,Ttr}
  rbspace_row::RBSpaceUnsteady
  rbspace_col::RBSpaceUnsteady
end

mutable struct RBSteadyLiftVariable{Top,Ttr} <: RBLiftVariable{Top,Ttr}
  feop::ParamSteadyLiftOperator{Top,Ttr}
  rbspace_row::RBSpaceSteady

  function RBSteadyLiftVariable(
    feop::ParamSteadyLiftOperator{Top,Ttr},
    rbspace_row::RBSpaceSteady) where {Top,Ttr}
    new{Top,Ttr}(feop,rbspace_row)
  end
end

mutable struct RBUnsteadyLiftVariable{Top,Ttr} <: RBLiftVariable{Top,Ttr}
  feop::ParamUnsteadyLiftOperator{Top,Ttr}
  rbspace_row::RBSpaceUnsteady

  function RBUnsteadyLiftVariable(
    feop::ParamUnsteadyLiftOperator{Top,Ttr},
    rbspace_row::RBSpaceUnsteady) where {Top,Ttr}
    new{Top,Ttr}(feop,rbspace_row)
  end
end

function RBVariable(
  feop::ParamSteadyLinOperator{Top},
  rbspace_row::RBSpaceSteady) where Top

  RBSteadyLinVariable{Top}(feop,rbspace_row)
end

function RBVariable(
  feop::ParamUnsteadyLinOperator{Top},
  rbspace_row::RBSpaceUnsteady) where Top

  RBUnsteadyLinVariable{Top}(feop,rbspace_row)
end

function RBVariable(
  feop::ParamSteadyBilinOperator{Top,Ttr},
  rbspace_row::RBSpaceSteady,
  rbspace_col::RBSpaceSteady) where {Top,Ttr}

  RBSteadyBilinVariable{Top,Ttr}(feop,rbspace_row,rbspace_col)
end

function RBVariable(
  feop::ParamUnsteadyBilinOperator{Top,Ttr},
  rbspace_row::RBSpaceUnsteady,
  rbspace_col::RBSpaceUnsteady) where {Top,Ttr}

  RBUnsteadyBilinVariable{Top,Ttr}(feop,rbspace_row,rbspace_col)
end

function RBLiftVariable(op::RBSteadyBilinVariable)
  feop = ParamLiftOperator(get_background_feop(op))
  rbs = get_rbspace_row(op)
  rbs_lift = RBSpace(get_id(rbs)*:_lift,get_basis_space(rbs))
  RBSteadyLiftVariable(feop,rbs_lift)
end

function RBLiftVariable(op::RBUnsteadyBilinVariable)
  feop = ParamLiftOperator(get_background_feop(op))
  rbs = get_rbspace_row(op)
  rbs_lift = RBSpace(get_id(rbs)*:_lift,get_basis_space(rbs),get_basis_time(rbs))
  RBUnsteadyLiftVariable(feop,rbs_lift)
end

const RBSteadyVariable{Top,Ttr} =
  Union{RBSteadyLinVariable{Top},RBSteadyBilinVariable{Top,Ttr},RBSteadyLiftVariable{Top,Ttr}}

const RBUnsteadyVariable{Top,Ttr} =
  Union{RBUnsteadyLinVariable{Top},RBUnsteadyBilinVariable{Top,Ttr},RBUnsteadyLiftVariable{Top,Ttr}}

get_background_feop(rbop::RBVariable) = rbop.feop
get_param_function(op::RBVariable) = get_param_function(op.feop)
get_fe_function(op::RBVariable) = get_fe_function(op.feop)
get_rbspace_row(rbop::RBVariable) = rbop.rbspace_row
get_rbspace_col(rbop::RBBilinVariable) = rbop.rbspace_col
get_id(rbop::RBVariable) = get_id(get_background_feop(rbop))
get_basis_space_row(rbop::RBVariable) = get_basis_space(get_rbspace_row(rbop))
get_basis_space_col(rbop::RBVariable) = get_basis_space(get_rbspace_col(rbop))
get_tests(op::RBVariable) = get_tests(op.feop)
get_trials(op::RBBilinVariable) = get_trials(op.feop)
get_trials(op::RBLiftVariable) = get_trials(op.feop)
Gridap.FESpaces.get_test(op::RBVariable) = get_test(op.feop)
Gridap.FESpaces.get_trial(op::RBBilinVariable) = get_trial(op.feop)
Gridap.FESpaces.get_trial(op::RBLiftVariable) = get_trial(op.feop)
get_test_no_bc(op::RBVariable) = get_test_no_bc(op.feop)
get_trial_no_bc(op::RBBilinVariable) = get_trial_no_bc(op.feop)
get_trial_no_bc(op::RBLiftVariable) = get_trial_no_bc(op.feop)

get_basis_time_row(rbop::RBVariable) = get_basis_time(get_rbspace_row(rbop))
get_basis_time_col(rbop::RBVariable) = get_basis_time(get_rbspace_col(rbop))
get_Nt(op::RBVariable) = get_Nt(op.rbspace_row)

get_nrows(op::RBSteadyVariable) = get_ns(get_rbspace_row(op))
get_nrows(op::RBUnsteadyVariable) = get_ns(get_rbspace_row(op))*get_nt(get_rbspace_row(op))

issteady(::RBSteadyVariable) = true
issteady(::RBUnsteadyVariable) = false

islinear(::RBVariable) = true
islinear(::RBBilinVariable) = false

function Gridap.FESpaces.get_cell_dof_ids(
  rbop::RBVariable,
  trian::Triangulation)
  get_cell_dof_ids(get_background_feop(rbop),trian)
end

Gridap.FESpaces.assemble_vector(op::RBLinVariable,args...) = assemble_vector(op.feop,args...)
Gridap.FESpaces.assemble_matrix(op::RBBilinVariable,args...) = assemble_matrix(op.feop,args...)
assemble_matrix_and_lifting(op::RBBilinVariable,args...) = assemble_matrix_and_lifting(op.feop,args...)

function Gridap.FESpaces.assemble_vector(op::RBLinVariable{Affine},args...)
  assemble_vector(op.feop,args...)(realization(op))
end

function Gridap.FESpaces.assemble_matrix(op::RBBilinVariable{Affine,Ttr},args...) where Ttr
  assemble_matrix(op.feop,args...)(realization(op))
end

function assemble_matrix_and_lifting(op::RBBilinVariable{Affine,Ttr},args...) where Ttr
  assemble_matrix_and_lifting(op.feop,args...)(realization(op))
end

function assemble_functional_vector(op::RBLinVariable)
  assemble_functional_vector(op.feop)
end

function assemble_functional_matrix_and_lifting(op::RBUnsteadyBilinVariable)
  assemble_functional_matrix_and_lifting(op.feop)
end

function assemble_affine_vector(op::RBSteadyLinVariable)
  assemble_vector(op.feop)(realization(op))
end

function assemble_affine_vector(op::RBUnsteadyLinVariable)
  assemble_vector(op.feop,realization(op.feop.tinfo))(realization(op))
end

function assemble_affine_vector(op::RBSteadyLiftVariable)
  assemble_vector(op.feop)(realization(op))
end

function assemble_affine_vector(op::RBUnsteadyLiftVariable)
  assemble_vector(op.feop,realization(op.feop.tinfo))(realization(op))
end

function assemble_affine_matrix(op::RBSteadyBilinVariable)
  assemble_matrix(op.feop)(realization(op))
end

function assemble_affine_matrix(op::RBUnsteadyBilinVariable)
  t = realization(op.feop.tinfo)
  assemble_matrix(op.feop,t)(realization(op))
end

function assemble_affine_matrix(op::RBSteadyBilinVariable{Nonlinear,Ttr}) where Ttr
  u = realization(op.feop.tests)
  assemble_matrix(op.feop)(u)
end

function assemble_affine_matrix(op::RBUnsteadyBilinVariable{Nonlinear,Ttr}) where Ttr
  t = realization(op.feop.tinfo)
  u = realization(op.feop.tests)
  assemble_matrix(op.feop,t)(u)
end

function assemble_fe_structure(op::RBLinVariable,args...)
  assemble_vector(op,args...)
end

function assemble_fe_structure(op::RBBilinVariable,args...)
  assemble_matrix(op,args...)
end

function assemble_fe_structure(
  op::RBSteadyBilinVariable{Top,<:ParamTrialFESpace},
  args...) where Top

  assemble_matrix_and_lifting(op,args...)
end

function assemble_fe_structure(
  op::RBUnsteadyBilinVariable{Top,<:ParamTransientTrialFESpace},
  args...) where Top

  assemble_matrix_and_lifting(op,args...)
end

get_dirichlet_function(op::RBVariable) = get_dirichlet_function(op.feop)

get_pspace(op::RBVariable) = get_pspace(op.feop)
realization(op::RBVariable) = realization(get_pspace(op))
get_time_info(op::RBUnsteadyLinVariable) = get_time_info(op.feop)
get_time_info(op::RBUnsteadyBilinVariable) = get_time_info(op.feop)
get_dt(op::RBVariable) = get_dt(op.feop)
get_Nt(op::RBVariable) = get_Nt(op.feop)
get_θ(op::RBVariable) = get_θ(op.feop)
get_timesθ(op::RBVariable) = get_timesθ(op.feop)
get_reduced_timesθ(op::RBVariable,idx::Vector{Int}) = get_timesθ(op)[idx]
get_phys_quad_points(op::RBVariable) = get_phys_quad_points(op.feop)

function compute_in_timesθ(snaps::Snapshots,args...;kwargs...)
  id = get_id(snaps)*:θ
  snap = get_snap(snaps)
  nsnap = get_nsnap(snaps)
  Snapshots(id,compute_in_timesθ(snap,args...;kwargs...),nsnap)
end

"Small, full vector -> large, sparse vector"
function get_findnz_map(
  op::RBBilinVariable,
  q::Vector{T}) where {T<:Union{<:Param,<:FEFunction}}

  get_findnz_map(op,first(q))
end

function get_findnz_map(
  op::RBSteadyBilinVariable,
  μ::Param)

  M = assemble_matrix(op)(μ)
  first(findnz(M[:]))
end

function get_findnz_map(
  op::RBUnsteadyBilinVariable,
  μ::Param)

  dtθ = get_dt(op)*get_θ(op)
  M = assemble_matrix(op,dtθ)(μ)
  first(findnz(M[:]))
end

function get_findnz_map(
  op::RBVariable{Nonlinear,Ttr},
  f::FEFunction) where Ttr

  M = assemble_matrix(op)(f)
  first(findnz(M[:]))
end

"Viceversa"
function get_inverse_findnz_map(op::RBBilinVariable,q::T) where T
  findnz_map = get_findnz_map(op,q)
  inv_map(i::Int) = findall(x -> x == i,findnz_map)[1]
  inv_map
end

function unfold_spacetime(
  op::RBUnsteadyVariable,
  vals::AbstractVector{T}) where T

  Nt = get_Nt(op)
  Ns = Int(size(vals,1)/Nt)

  space_vals = Matrix{T}(reshape(vals,Ns,Nt))
  time_vals = Matrix{T}(reshape(vals,Nt,Ns))
  space_vals,time_vals
end

function unfold_spacetime(
  op::RBUnsteadyVariable,
  vals::AbstractMatrix{T}) where T

  unfold_vec(k::Int) = unfold_spacetime(op,vals[:,k])
  vals = Broadcasting(unfold_vec)(axes(vals,2))
  Matrix(first.(vals)),Matrix(last.(vals))
end

function rb_space_projection(
  op::RBLinVariable;
  mv=assemble_affine_vector(op))

  rbrow = get_rbspace_row(op)
  rb_space_projection(rbrow,mv)
end

function rb_space_projection(
  op::RBBilinVariable;
  mv=assemble_affine_matrix(op))

  rbrow = get_rbspace_row(op)
  rbcol = get_rbspace_col(op)
  rb_space_projection(rbrow,rbcol,mv)
end

function rb_time_projection(
  op::RBLinVariable;
  mv=assemble_vector(op)(realization(op)))

  rbrow = get_rbspace_row(op)
  rb_time_projection(rbrow,mv)
end

function rb_time_projection(
  op::RBBilinVariable;
  mv=assemble_matrix(op)(realization(op)))

  rbrow = get_rbspace_row(op)
  rbcol = get_rbspace_col(op)
  rb_time_projection(rbrow,rbcol,mv)
end

function rb_spacetime_projection(
  op::RBLinVariable;
  mv=assemble_vector(op)(realization(op)))

  proj_space = [rb_space_projection(op;mv=mv[:,i]) for i=axes(mv,2)]
  resh_proj = Matrix(proj_space)'
  proj_spacetime_block = rb_time_projection(op;mv=resh_proj)
  rbrow = get_rbspace_row(op)
  ns,nt = get_ns(rbrow),get_nt(rbrow)
  proj_spacetime = zeros(ns*nt,1)
  for i = 1:ns
    proj_spacetime[1+(i-1)*nt:i*nt,1] = proj_spacetime_block[i]
  end
  proj_spacetime
end

function rb_spacetime_projection(
  op::RBBilinVariable;
  mv=assemble_matrix(op)(realization(op)))

  proj_space = [rb_space_projection(op;mv=mv[i])[:] for i=eachindex(mv)]
  resh_proj = Matrix(proj_space)'
  proj_spacetime_block = rb_time_projection(op;mv=resh_proj)
  rbrow = get_rbspace_row(op)
  rbcol = get_rbspace_col(op)
  nsrow,ntrow = get_ns(rbrow),get_nt(rbrow)
  nscol,ntcol = get_ns(rbcol),get_nt(rbcol)
  proj_spacetime = zeros(nsrow*ntrow,nscol*ntcol)
  for i = 1:nscol
    for j = 1:nsrow
      proj_spacetime[1+(j-1)*ntrow:j*ntrow,1+(i-1)*ntcol:i*ntcol] =
        proj_spacetime_block[(i-1)*nsrow+j]
    end
  end
  proj_spacetime
end

function rb_projection(
  op::RBSteadyVariable,
  mv::AbstractArray)

  rb_space_projection(op;mv=mv)
end

function rb_projection(
  op::RBUnsteadyVariable,
  mv::AbstractArray)

  rb_spacetime_projection(op;mv=mv)
end

function rb_projection(
  op::RBVariable,
  mv::NTuple{2,AbstractArray})

  op_lift = RBLiftVariable(op)
  rb_projection(op,first(mv)),rb_projection(op_lift,last(mv))
end
