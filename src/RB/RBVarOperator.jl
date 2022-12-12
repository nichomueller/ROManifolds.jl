abstract type RBVarOperator{Top,TT} end
abstract type RBLinOperator{Top} <: RBVarOperator{Top,nothing} end
abstract type RBBilinOperator{Top,TT} <: RBVarOperator{Top,TT} end

mutable struct RBSteadyLinOperator{Top} <: RBLinOperator{Top}
  feop::ParamSteadyLinOperator{Top}
  rbspace_row::RBSpaceSteady
end

mutable struct RBUnsteadyLinOperator{Top} <: RBLinOperator{Top}
  feop::ParamUnsteadyLinOperator{Top}
  rbspace_row::RBSpaceUnsteady
end

mutable struct RBSteadyBilinOperator{Top,TT} <: RBBilinOperator{Top,TT}
  feop::ParamSteadyBilinOperator{Top,TT}
  rbspace_row::RBSpaceSteady
  rbspace_col::RBSpaceSteady
end

mutable struct RBUnsteadyBilinOperator{Top,TT} <: RBBilinOperator{Top,TT}
  feop::ParamUnsteadyBilinOperator{Top,TT}
  rbspace_row::RBSpaceUnsteady
  rbspace_col::RBSpaceUnsteady
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
  feop::ParamSteadyBilinOperator{Top,TT},
  rbspace_row::RBSpaceSteady,
  rbspace_col::RBSpaceSteady) where {Top,TT}

  RBSteadyBilinOperator{Top,TT}(feop,rbspace_row,rbspace_col)
end

function RBVarOperator(
  feop::ParamUnsteadyBilinOperator{Top,TT},
  rbspace_row::RBSpaceUnsteady,
  rbspace_col::RBSpaceUnsteady) where {Top,TT}

  RBUnsteadyBilinOperator{Top,TT}(feop,rbspace_row,rbspace_col)
end

abstract type RBLiftingOperator{Top,TT} <: RBBilinOperator{Top,TT} end

mutable struct RBSteadyLiftingOperator{Top,TT} <: RBLiftingOperator{Top,TT}
  feop::ParamSteadyLiftingOperator{Top,TT}
  rbspace_row::RBSpaceSteady
end

mutable struct RBUnsteadyLiftingOperator{Top,TT} <: RBLiftingOperator{Top,TT}
  feop::ParamUnsteadyLiftingOperator{Top,TT}
  rbspace_row::RBSpaceUnsteady
end

function RBLiftingOperator(op::RBSteadyBilinOperator{Top,TT}) where {OT,TT}
  feop = get_background_feop(op)
  rbs = get_rbspace_row(op)
  rbs_lift = RBSpace(get_id(rbs)*:_lift,get_basis_space(rbs))
  ParamSteadyLiftingOperator{OT,TT}(feop,rbs_lift)
end

function RBLiftingOperator(op::RBUnsteadyBilinOperator{Top,TT}) where {OT,TT}
  feop = get_background_feop(op)
  rbs = get_rbspace_row(op)
  rbs_lift = RBSpace(get_id(rbs)*:_lift,get_basis_space(rbs),get_basis_time(rbs))
  ParamUnsteadyLiftingOperator{OT,TT}(feop,rbs_lift)
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
Gridap.FESpaces.get_test(op::RBVarOperator) = get_test(op.feop)
Gridap.FESpaces.get_trial(op::RBBilinOperator) = get_trial(op.feop)
get_test_no_bc(op::RBVarOperator) = get_test_no_bc(op.feop)
get_trial_no_bc(op::RBBilinOperator) = get_trial_no_bc(op.feop)

get_basis_time_row(rbop::RBVarOperator) = get_basis_time(get_rbspace_row(rbop))
get_basis_time_col(rbop::RBVarOperator) = get_basis_time(get_rbspace_col(rbop))
get_Nt(op::RBVarOperator) = get_Nt(op.rbspace_row)

get_nrows(op::RBVarOperator) = get_ns(get_rbspace_row(op))
get_nrows(op::RBUnsteadyLinOperator) = get_ns(get_rbspace_row(op))*get_nt(get_rbspace_row(op))
get_nrows(op::RBUnsteadyBilinOperator) = get_ns(get_rbspace_row(op))*get_nt(get_rbspace_row(op))

function Gridap.FESpaces.get_cell_dof_ids(
  rbop::RBVarOperator,
  trian::Triangulation)
  get_cell_dof_ids(get_background_feop(rbop),trian)
end

Gridap.FESpaces.assemble_vector(op::RBLinOperator,args...) = assemble_vector(op.feop,args...)
Gridap.FESpaces.assemble_matrix(op::RBBilinOperator,args...) = assemble_matrix(op.feop,args...)
assemble_matrix_and_lifting(op::RBBilinOperator,args...) = assemble_matrix_and_lifting(op.feop,args...)
assemble_lift(op::RBLiftingOperator,args...) = assemble_lift(op.feop,args...)

function Gridap.FESpaces.assemble_vector(op::RBLinOperator{Affine},args...)
  assemble_vector(op,args...)(realization(op))
end

function Gridap.FESpaces.assemble_matrix(op::RBBilinOperator{Affine,TT},args...) where TT
  assemble_matrix(op,args...)(realization(op))
end

function assemble_matrix_and_lifting(op::RBBilinOperator{Affine,TT},args...) where TT
  assemble_matrix_and_lifting(op,args...)(realization(op))
end

get_pspace(op::RBVarOperator) = get_pspace(op.feop)
realization(op::RBVarOperator) = realization(get_pspace(op))
get_time_info(op::RBUnsteadyLinOperator) = get_time_info(op.feop)
get_time_info(op::RBUnsteadyBilinOperator) = get_time_info(op.feop)
get_dt(op::RBVarOperator) = get_dt(op.feop)
get_Nt(op::RBVarOperator) = get_Nt(op.feop)
get_θ(op::RBVarOperator) = get_θ(op.feop)
get_timesθ(op::RBVarOperator) = get_timesθ(op.feop)

"Small, full vector -> large, sparse vector"
function get_findnz_mapping(op::RBSteadyBilinOperator)
  M = assemble_matrix(op)(realization(op))
  first(findnz(M[:]))
end

"Small, full vector -> large, sparse vector"
function get_findnz_mapping(op::RBUnsteadyBilinOperator)
  M = assemble_matrix(op,get_dt(op))(realization(op))
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
  vals = Broadcasting(unfold_vec)(axis(vals,2))
  Matrix(first.(vals)),Matrix(last.(vals))
end

function rb_projection(op::RBSteadyLinOperator)
  id = get_id(op)
  println("Linear operator $id is affine: computing Φᵀ$id")

  vec = assemble_vector(op)
  brow = get_basis_space_row(op)

  Matrix(brow'*vec)
end

function rb_projection(op::RBUnsteadyLinOperator)
  id = get_id(op)
  println("Linear operator $id is affine: computing Φᵀ$id")

  vec = assemble_vector(op,get_dt(op))
  brow = get_basis_space_row(op)

  Matrix(brow'*vec)
end

function rb_projection(op::RBSteadyBilinOperator)
  id = get_id(op)
  println("Bilinear operator $id is affine: computing Φᵀ$(id)Φ")

  mat = assemble_matrix(op)
  brow = get_basis_space_row(op)
  bcol = get_basis_space_col(op)

  brow'*mat*bcol
end

function rb_projection(op::RBUnsteadyBilinOperator)
  id = get_id(op)
  println("Bilinear operator $id is affine: computing Φᵀ$(id)Φ")

  mat = assemble_matrix(op,get_dt(op))
  brow = get_basis_space_row(op)
  bcol = get_basis_space_col(op)

  brow'*mat*bcol
end

function rb_projection(op::RBSteadyLiftingOperator)
  id = get_id(op)
  println("Lifting operator $id is affine: computing Φᵀ$id")

  vec = assemble_lift(op)
  brow = get_basis_space_row(op)

  Matrix(brow'*vec)
end

function rb_projection(op::RBUnsteadyLiftingOperator)
  id = get_id(op)
  println("Lifting operator $id is affine: computing Φᵀ$id")

  vec = assemble_lift(op,get_dt(op))
  brow = get_basis_space_row(op)

  Matrix(brow'*vec)
end
