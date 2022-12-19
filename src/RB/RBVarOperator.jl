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

function assemble_affine_matrix(op::RBSteadyBilinOperator)
  assemble_matrix(op.feop)(realization(op))
end

function assemble_affine_matrix(op::RBUnsteadyBilinOperator)
  assemble_matrix(op.feop,realization(op.tinfo))(realization(op))
end

function assemble_affine_matrix(op::RBSteadyBilinOperator{Nonlinear,Ttr}) where Ttr
  u = realization(op.feop.tests)
  assemble_matrix(op.feop)(u)
end

function assemble_affine_matrix(op::RBUnsteadyBilinOperator{Nonlinear,Ttr}) where Ttr
  u = realization(op.feop.tests)
  assemble_matrix(op.feop,realization(op.tinfo))(u)
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

"Small, full vector -> large, sparse vector"
function get_findnz_mapping(op::RBBilinOperator)
  M = assemble_affine_matrix(op)
  first(findnz(M[:]))
end

"Viceversa"
function get_inverse_findnz_mapping(op::RBBilinOperator)
  findnz_map = get_findnz_mapping(op)
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

function rb_projection(op::RBSteadyLinOperator)
  vec = assemble_vector(op)
  brow = get_basis_space_row(op)

  Matrix(brow'*vec)
end

function rb_projection(op::RBUnsteadyLinOperator)
  vec = assemble_vector(op,get_dt(op))
  brow = get_basis_space_row(op)

  Matrix(brow'*vec)
end

function rb_projection(op::RBSteadyBilinOperator)
  mat = assemble_matrix(op)
  brow = get_basis_space_row(op)
  bcol = get_basis_space_col(op)

  Matrix((brow'*mat*bcol)[:])
end

function rb_projection(op::RBUnsteadyBilinOperator)
  mat = assemble_matrix(op,get_dt(op))
  brow = get_basis_space_row(op)
  bcol = get_basis_space_col(op)

  Matrix((brow'*mat*bcol)[:])
end
