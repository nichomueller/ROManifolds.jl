abstract type RBVarOperator{Top,TT,Tsp} end

mutable struct RBLinOperator{Top,Tsp} <: RBVarOperator{Top,nothing,Tsp}
  feop::ParamLinOperator{Top}
  rbspace_row::Tsp
end

mutable struct RBBilinOperator{Top,TT,Tsp} <: RBVarOperator{Top,TT,Tsp}
  feop::ParamBilinOperator{Top,TT}
  rbspace_row::Tsp
  rbspace_col::Tsp
end

function RBVarOperator(
  feop::ParamLinOperator{Top},
  rbspace_row::Tsp) where {Top,Tsp}

  RBLinOperator{Top,Tsp}(feop,rbspace_row)
end

function RBVarOperator(
  feop::ParamBilinOperator{Top,TT},
  rbspace_row::Tsp,
  rbspace_col::Tsp) where {Top,TT,Tsp}

  RBBilinOperator{Top,TT,Tsp}(feop,rbspace_row,rbspace_col)
end

get_background_feop(rbop::RBVarOperator) = rbop.feop
get_param_function(op::RBVarOperator) = get_param_function(op.feop)
get_fe_function(op::RBVarOperator) = get_fe_function(op.feop)
get_rbspace_row(rbop::RBVarOperator) = rbop.rbspace_row
get_rbspace_col(rbop::RBBilinOperator) = rbop.rbspace_col
get_id(rbop::RBVarOperator) = get_id(get_background_feop(rbop))
get_basis_space_row(rbop::RBVarOperator) = get_basis_space(get_rbspace_row(rbop))
get_basis_space_col(rbop::RBVarOperator) = get_basis_space(get_rbspace_col(rbop))
get_basis_time_row(rbop::RBVarOperator{Top,TT,RBSpaceUnsteady}) where {Top,TT} =
  get_basis_time(get_rbspace_row(rbop))
get_basis_time_col(rbop::RBVarOperator{Top,TT,RBSpaceUnsteady}) where {Top,TT} =
  get_basis_time(get_rbspace_col(rbop))
get_tests(op::RBVarOperator) = get_tests(op.feop)
get_trials(op::RBBilinOperator) = get_trials(op.feop)
Gridap.FESpaces.get_test(op::RBVarOperator) = get_test(op.feop)
Gridap.FESpaces.get_trial(op::RBBilinOperator) = get_trial(op.feop)
get_test_no_bc(op::RBVarOperator) = get_test_no_bc(op.feop)
get_trial_no_bc(op::RBBilinOperator) = get_trial_no_bc(op.feop)

function get_nrows(op::RBVarOperator{Top,TT,RBSpaceSteady}) where {Top,TT}
  get_ns(get_rbspace_row(op))
end

function get_nrows(op::RBVarOperator{Top,TT,RBSpaceUnsteady}) where {Top,TT}
  get_ns(get_rbspace_row(op))*get_nt(get_rbspace_row(op))
end

function Gridap.FESpaces.get_cell_dof_ids(
  rbop::RBVarOperator,
  trian::Triangulation)
  get_cell_dof_ids(get_background_feop(rbop),trian)
end

Gridap.FESpaces.assemble_vector(op::RBLinOperator) = assemble_vector(op.feop)
Gridap.FESpaces.assemble_matrix(op::RBBilinOperator) = assemble_matrix(op.feop)
assemble_lifting(op::RBBilinOperator) = assemble_lifting(op.feop)
assemble_matrix_and_lifting(op::RBBilinOperator) = assemble_matrix_and_lifting(op.feop)

get_pspace(op::RBVarOperator) = get_pspace(op.feop)
realization(op::RBVarOperator) = realization(get_pspace(op))

Gridap.Algebra.allocate_vector(op::RBLinOperator) = assemble_vector(op.feop)
Gridap.Algebra.allocate_matrix(op::RBBilinOperator) = assemble_matrix(op.feop)
allocate_structure(op::RBLinOperator) = allocate_vector(op)
allocate_structure(op::RBBilinOperator) = allocate_matrix(op)

function assemble_affine_vector(
  op::RBLinOperator{Affine,RBSpaceSteady})
  assemble_vector(op)(realization(op))
end

function assemble_affine_vector(
  op::RBLinOperator{Affine,RBSpaceUnsteady})
  assemble_vector(op)(realization(op),first(get_timesθ(op)))
end

function assemble_affine_matrix(
  op::RBBilinOperator{Affine,TT,RBSpaceSteady}) where TT
  assemble_matrix(op)(realization(op))
end

function assemble_affine_matrix(
  op::RBBilinOperator{Affine,TT,RBSpaceUnsteady}) where TT
  assemble_matrix(op)(realization(op),first(get_timesθ(op)))
end

function assemble_affine_matrix_and_lifting(
  op::RBBilinOperator{Affine,TT,RBSpaceSteady}) where TT
  assemble_matrix_and_lifting(op)(realization(op))
end

function assemble_affine_matrix_and_lifting(
  op::RBBilinOperator{Affine,TT,RBSpaceUnsteady}) where TT
  assemble_matrix_and_lifting(op)(realization(op),first(get_timesθ(op)))
end

function get_findnz_mapping(op::RBLinOperator)
  v = assemble_vector(op)(realization(op))
  collect(eachindex(v))
end

"Small, full vector -> large, sparse vector"
function get_findnz_mapping(op::RBBilinOperator)
  M = assemble_matrix(op)(realization(op))
  first(findnz(M[:]))
end

"Viceversa"
function get_inverse_findnz_mapping(op::RBVarOperator)
  findnz_map = get_findnz_mapping(op)
  inv_map(i::Int) = findall(x -> x == i,findnz_map)[1]
  inv_map
end

function unfold_spacetime(
  op::RBVarOperator{Top,TT,RBSpaceUnsteady},
  vals::AbstractVector{Tv}) where {Top,TT,Tv}

  Ns = get_Ns(op)
  Nt = get_Nt(op)
  @assert size(vals,1) == Ns*Nt "Wrong space-time dimensions"

  space_vals = Matrix{Tv}(reshape(vals,Ns,Nt))
  time_vals = Matrix{Tv}(reshape(vals,Nt,Ns))
  space_vals,time_vals
end

function unfold_spacetime(
  op::RBVarOperator{Top,TT,RBSpaceUnsteady},
  vals::AbstractMatrix{Tv}) where {Top,TT,Tv}

  unfold_vec(k::Int) = unfold_spacetime(op,vals[:,k])
  vals = Broadcasting(unfold_vec)(axis(vals,2))
  Matrix(first.(vals)),Matrix(last.(vals))
end

function rb_projection(op::RBLinOperator{Affine,Tsp}) where Tsp
  id = get_id(op)
  println("Vector $id is affine: computing Φᵀ$id")

  vec = assemble_affine_vector(op)
  brow = get_basis_space_row(op)

  Matrix(brow'*vec)
end

function rb_projection(op::RBBilinOperator{Affine,TT,Tsp}) where {TT,Tsp}
  id = get_id(op)
  println("Matrix $id and its lifting are affine: computing Φᵀ$(id)Φ and Φᵀlift_$id")

  mat,lift = assemble_affine_matrix_and_lifting(op)
  brow = get_basis_space_row(op)
  bcol = get_basis_space_col(op)

  brow'*mat*bcol,Matrix(brow'*lift)
end

function rb_projection(op::RBBilinOperator{Affine,UnconstrainedFESpace,Tsp}) where Tsp
  id = get_id(op)
  println("Matrix $id is affine: computing Φᵀ$(id)Φ")

  mat = assemble_affine_matrix_and_lifting(op)
  brow = get_basis_space_row(op)
  bcol = get_basis_space_col(op)

  brow'*mat*bcol
end
