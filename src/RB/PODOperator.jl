function reduced_operator(
  solver::RBSolver,
  feop::TransientParamFEOperator,
  s::S) where S

  red_trial,red_test = reduced_fe_space(solver,feop,s)
  odeop = get_algebraic_operator(feop)
  reduced_operator(solver,odeop,red_trial,red_test,s)
end

function reduced_operator(
  solver::RBSolver,
  odeop::ODEParamOperator,
  trial::RBSpace,
  test::RBSpace,
  s::S) where S

  pop = PODOperator(odeop,trial,test)
  reduced_operator(solver,pop,s)
end

abstract type RBOperator{T<:ODEParamOperatorType} <: ODEParamOperatorWithTrian{T} end
const LinearRBOperator = RBOperator{LinearODE}

struct PODOperator{T} <: RBOperator{T}
  odeop::ODEParamOperatorWithTrian{T}
  trial::RBSpace
  test::RBSpace
end

FESpaces.get_trial(op::PODOperator) = op.trial
FESpaces.get_test(op::PODOperator) = op.test
FEM.realization(op::PODOperator;kwargs...) = realization(op.odeop;kwargs...)
FEM.get_fe_operator(op::PODOperator) = FEM.get_fe_operator(op.odeop)
get_fe_trial(op::PODOperator) = get_trial(op.odeop)
get_fe_test(op::PODOperator) = get_test(op.odeop)

function FEM.get_linear_operator(op::PODOperator{LinearNonlinearParamODE})
  PODOperator(get_linear_operator(op.odeop),op.trial,op.test)
end

function FEM.get_nonlinear_operator(op::PODOperator{LinearNonlinearParamODE})
  PODOperator(get_nonlinear_operator(op.odeop),op.trial,op.test)
end

function FEM.set_triangulation(
  op::PODOperator,
  trians_rhs,
  trians_lhs)

  PODOperator(set_triangulation(op.odeop,trians_rhs,trians_lhs),op.trial,op.test)
end

function FEM.change_triangulation(
  op::PODOperator,
  trians_rhs,
  trians_lhs)

  PODOperator(change_triangulation(op.odeop,trians_rhs,trians_lhs),op.trial,op.test)
end

function ODEs.allocate_odeopcache(
  op::PODOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}})

  allocate_odeopcache(op.odeop,r,us)
end

function ODEs.update_odeopcache!(
  ode_cache,
  op::PODOperator,
  r::TransientParamRealization)

  update_odeopcache!(ode_cache,op.odeop,r)
end

function Algebra.allocate_residual(
  op::PODOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache)

  allocate_residual(op.odeop,r,us,odeopcache)
end

function Algebra.allocate_jacobian(
  op::PODOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache)

  allocate_jacobian(op.odeop,r,us,odeopcache)
end

function allocate_jacobian_and_residual(op::PODOperator,r,us,odeopcache)
  A = allocate_jacobian(op,r,us,odeopcache)
  b = allocate_residual(op,r,us,odeopcache)
  return A,b
end

function Algebra.residual!(
  b::Contribution,
  op::PODOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache;
  kwargs...)

  residual!(b,op.odeop,r,us,ode_cache;kwargs...)
  return Snapshots(b,r)
end

function Algebra.jacobian!(
  A::Tuple{Vararg{Contribution}},
  op::PODOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  jacobian!(A,op.odeop,r,us,ws,odeopcache)
  return Snapshots.(A,r)
end

function FEM.residual_and_jacobian(
  solver::RBSolver,
  op::RBOperator,
  s::S) where S

  fesolver = get_fe_solver(solver)
  x = get_values(s)
  r = get_realization(s)
  FEM.residual_and_jacobian(fesolver,op.odeop,r,x)
end
