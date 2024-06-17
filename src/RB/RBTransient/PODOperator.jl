function RBSteady.reduced_operator(
  solver::RBSolver,
  feop::TransientParamFEOperator,
  s::AbstractTransientSnapshots)

  red_trial,red_test = reduced_fe_space(solver,feop,s)
  op = get_algebraic_operator(feop)
  reduced_operator(solver,op,red_trial,red_test,s)
end

function RBSteady.reduced_operator(
  solver::RBSolver,
  op::TransientParamFEOperator,
  trial::RBSpace,
  test::RBSpace,
  s::AbstractTransientSnapshots)

  pop = TransientPODOperator(op,trial,test)
  reduced_operator(solver,pop,s)
end

abstract type TransientRBOperator{T<:ODEParamOperatorType} <: ODEParamOperatorWithTrian{T} end

struct TransientPODOperator{T} <: TransientRBOperator{T}
  op::ParamOperatorWithTrian{T}
  trial::RBSpace
  test::RBSpace
end

FESpaces.get_trial(op::TransientPODOperator) = op.trial
FESpaces.get_test(op::TransientPODOperator) = op.test
ParamDataStructures.realization(op::TransientPODOperator;kwargs...) = realization(op.op;kwargs...)
ParamSteady.get_fe_operator(op::TransientPODOperator) = ParamSteady.get_fe_operator(op.op)
ParamSteady.get_vector_index_map(op::TransientPODOperator) = get_vector_index_map(op.op)
ParamSteady.get_matrix_index_map(op::TransientPODOperator) = get_matrix_index_map(op.op)
RBSteady.get_fe_trial(op::TransientPODOperator) = get_trial(op.op)
RBSteady.get_fe_test(op::TransientPODOperator) = get_test(op.op)

function ParamSteady.get_linear_operator(op::TransientPODOperator)
  TransientPODOperator(get_linear_operator(op.op),op.trial,op.test)
end

function ParamSteady.get_nonlinear_operator(op::TransientPODOperator)
  TransientPODOperator(get_nonlinear_operator(op.op),op.trial,op.test)
end

function ParamSteady.set_triangulation(
  op::TransientPODOperator,
  trians_rhs,
  trians_lhs)

  TransientPODOperator(set_triangulation(op.op,trians_rhs,trians_lhs),op.trial,op.test)
end

function ParamSteady.change_triangulation(
  op::TransientPODOperator,
  trians_rhs,
  trians_lhs)

  TransientPODOperator(change_triangulation(op.op,trians_rhs,trians_lhs),op.trial,op.test)
end

function ODEs.allocate_odeopcache(
  op::TransientPODOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}})

  allocate_odeopcache(op.op,r,us)
end

function ODEs.update_odeopcache!(
  odeopcache,
  op::TransientPODOperator,
  r::TransientParamRealization)

  update_odeopcache!(odeopcache,op.op,r)
end

function Algebra.allocate_residual(
  op::TransientPODOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  odeopcache)

  allocate_residual(op.op,r,us,odeopcache)
end

function Algebra.allocate_jacobian(
  op::TransientPODOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  odeopcache)

  allocate_jacobian(op.op,r,us,odeopcache)
end

function Algebra.residual!(
  b::Contribution,
  op::TransientPODOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractParamVector}},
  odeopcache;
  kwargs...)

  residual!(b,op.op,r,us,odeopcache;kwargs...)
  i = get_vector_index_map(op)
  return Snapshots(b,i,r)
end

function Algebra.jacobian!(
  A::TupOfArrayContribution,
  op::TransientPODOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  jacobian!(A,op.op,r,us,ws,odeopcache)
  i = get_matrix_index_map(op)
  return Snapshots(A,i,r)
end

function RBSteady.jacobian_and_residual(solver::RBSolver,op::TransientRBOperator,s::AbstractTransientSnapshots)
  jacobian_and_residual(get_fe_solver(solver),op.op,s)
end

function RBSteady.jacobian_and_residual(fesolver::ODESolver,odeop::ODEParamOperator,s::AbstractTransientSnapshots)
  us = (get_values(s),)
  r = get_realization(s)
  odecache = allocate_odecache(fesolver,odeop,r,us)
  A,b = jacobian_and_residual(fesolver,odeop,r,us,odecache)
  iA = get_matrix_index_map(odeop)
  ib = get_vector_index_map(odeop)
  return Snapshots(A,iA,r),Snapshots(b,ib,r)
end
