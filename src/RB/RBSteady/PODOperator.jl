function reduced_operator(
  solver::RBSolver,
  feop::ParamFEOperator,
  s::AbstractSnapshots)

  red_trial,red_test = reduced_fe_space(solver,feop,s)
  op = get_algebraic_operator(feop)
  reduced_operator(solver,op,red_trial,red_test,s)
end

function reduced_operator(
  solver::RBSolver,
  op::ParamOperator,
  trial::RBSpace,
  test::RBSpace,
  s::AbstractSnapshots)

  pop = PODOperator(op,trial,test)
  reduced_operator(solver,pop,s)
end

abstract type RBOperator{T<:ParamOperatorType} <: ParamOperatorWithTrian{T} end

struct PODOperator{T} <: RBOperator{T}
  op::ParamOperatorWithTrian{T}
  trial::RBSpace
  test::RBSpace
end

FESpaces.get_trial(op::PODOperator) = op.trial
FESpaces.get_test(op::PODOperator) = op.test
ParamDataStructures.realization(op::PODOperator;kwargs...) = realization(op.op;kwargs...)
ParamSteady.get_fe_operator(op::PODOperator) = ParamSteady.get_fe_operator(op.op)
ParamSteady.get_vector_index_map(op::PODOperator) = get_vector_index_map(op.op)
ParamSteady.get_matrix_index_map(op::PODOperator) = get_matrix_index_map(op.op)
get_fe_trial(op::PODOperator) = get_trial(op.op)
get_fe_test(op::PODOperator) = get_test(op.op)

function ParamSteady.get_linear_operator(op::PODOperator)
  PODOperator(get_linear_operator(op.op),op.trial,op.test)
end

function ParamSteady.get_nonlinear_operator(op::PODOperator)
  PODOperator(get_nonlinear_operator(op.op),op.trial,op.test)
end

function ParamSteady.set_triangulation(
  op::PODOperator,
  trians_rhs,
  trians_lhs)

  PODOperator(set_triangulation(op.op,trians_rhs,trians_lhs),op.trial,op.test)
end

function ParamSteady.change_triangulation(
  op::PODOperator,
  trians_rhs,
  trians_lhs)

  PODOperator(change_triangulation(op.op,trians_rhs,trians_lhs),op.trial,op.test)
end

function Algebra.allocate_residual(op::PODOperator,r::AbstractParamRealization,u::AbstractParamVector)
  allocate_residual(op.op,r,u)
end

function Algebra.allocate_jacobian(op::PODOperator,r::AbstractParamRealization,u::AbstractParamVector)
  allocate_jacobian(op.op,r,u)
end

function ParamSteady.allocate_paramcache(
  op::PODOperator,
  r::ParamRealization,
  u::AbstractParamVector)

  allocate_paramcache(op.op,r,u)
end

function ParamSteady.update_paramcache!(
  paramcache,
  op::PODOperator,
  r::ParamRealization)

  update_odeopcache!(paramcache,op.op,r)
end

function Algebra.residual!(
  b::Contribution,
  op::PODOperator,
  r::AbstractParamRealization,
  u::AbstractParamVector,
  paramcache)

  residual!(b,op.op,r,u,paramcache)
  i = get_vector_index_map(op)
  return Snapshots(b,i,r)
end

function Algebra.jacobian!(
  A::Contribution,
  op::PODOperator,
  r::AbstractParamRealization,
  u::AbstractParamVector,
  paramcache)

  jacobian!(A,op.op,r,u,paramcache)
  i = get_matrix_index_map(op)
  return Snapshots(A,i,r)
end

function jacobian_and_residual(solver::RBSolver,op::RBOperator,s::AbstractSnapshots)
  jacobian_and_residual(get_fe_solver(solver),op.op,s)
end

function jacobian_and_residual(fesolver::FESolver,op::ParamOperator,s::AbstractSnapshots)
  u = get_values(s)
  r = get_realization(s)
  A,b = jacobian_and_residual(fesolver,op,r,u)
  iA = get_matrix_index_map(op)
  ib = get_matrix_index_map(op)
  return Snapshots(A,iA,r),Snapshots(b,ib,r)
end

function jacobian_and_residual(::FESolver,op::ParamOperator,r::ParamRealization,u::AbstractVector)
  pcache = allocate_paramcache(fesolver,op,r,u)
  A = jacobian(op,r,u,pcache)
  b = residual(op,r,u,pcache)
  return A,b
end
