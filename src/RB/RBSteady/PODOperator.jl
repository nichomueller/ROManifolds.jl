function reduced_operator(
  solver::RBSolver,
  feop::ParamFEOperator,
  s::AbstractSnapshots)

  red_trial,red_test = reduced_fe_space(solver,feop,s)
  odeop = get_algebraic_operator(feop)
  reduced_operator(solver,odeop,red_trial,red_test,s)
end

function reduced_operator(
  solver::RBSolver,
  odeop::ParamOperator,
  trial::RBSpace,
  test::RBSpace,
  s::AbstractSnapshots)

  pop = PODOperator(odeop,trial,test)
  reduced_operator(solver,pop,s)
end

abstract type RBOperator{T<:ParamOperatorType} <: ParamOperatorWithTrian{T} end

struct PODOperator{T} <: RBOperator{T}
  odeop::ParamOperatorWithTrian{T}
  trial::RBSpace
  test::RBSpace
end

FESpaces.get_trial(op::PODOperator) = op.trial
FESpaces.get_test(op::PODOperator) = op.test
ParamDataStructures.realization(op::PODOperator;kwargs...) = realization(op.odeop;kwargs...)
ParamFESpaces.get_fe_operator(op::PODOperator) = get_fe_operator(op.odeop)
get_fe_trial(op::PODOperator) = get_trial(op.odeop)
get_fe_test(op::PODOperator) = get_test(op.odeop)

function ParamFESpaces.get_linear_operator(op::PODOperator{LinearNonlinearParamODE})
  PODOperator(get_linear_operator(op.odeop),op.trial,op.test)
end

function ParamFESpaces.get_nonlinear_operator(op::PODOperator{LinearNonlinearParamODE})
  PODOperator(get_nonlinear_operator(op.odeop),op.trial,op.test)
end

function ParamFESpaces.set_triangulation(
  op::PODOperator,
  trians_rhs,
  trians_lhs)

  PODOperator(set_triangulation(op.odeop,trians_rhs,trians_lhs),op.trial,op.test)
end

function ParamFESpaces.change_triangulation(
  op::PODOperator,
  trians_rhs,
  trians_lhs)

  PODOperator(change_triangulation(op.odeop,trians_rhs,trians_lhs),op.trial,op.test)
end

function Algebra.allocate_residual(op::PODOperator,r::AbstractParamRealization,u::AbstractParamVector)
  allocate_residual(op.odeop,r,u)
end

function Algebra.allocate_jacobian(op::PODOperator,r::AbstractParamRealization,u::AbstractParamVector)
  allocate_jacobian(op.odeop,r,u)
end

function Algebra.residual!(
  b::Contribution,
  op::PODOperator,
  r::AbstractParamRealization,
  u::AbstractParamVector)

  residual!(b,op.odeop,r,u)
  return Snapshots(b,r)
end

function Algebra.jacobian!(
  A::Contribution,
  op::PODOperator,
  r::AbstractParamRealization,
  u::AbstractParamVector)

  jacobian!(A,op.odeop,r,u)
  return Snapshots(A,r)
end

function jacobian_and_residual(
  solver::RBSolver,
  op::RBOperator,
  r::AbstractParamRealization,
  s::AbstractSnapshots)

  jacobian_and_residual(get_fe_solver(solver),op.odeop,r,s)
end

function jacobian_and_residual(fesolver::FESolver,odeop::ParamOperator,s::AbstractSnapshots)
  u = get_values(s)
  r = get_realization(s)
  A,b = jacobian_and_residual(fesolver,odeop,r,u)
  return Snapshots(A,r),Snapshots(b,r)
end
