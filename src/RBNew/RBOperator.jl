
struct RBOperator{T} <: ODEOperator{T}
  feop::ODEOperator{T}
  test::TestRBSpace
  trial::TrialRBSpace
end

ReferenceFEs.get_order(op::RBOperator) = get_order(op.feop)

function TransientFETools.allocate_cache(
  op::RBOperator,
  r::TransientParamRealization)

  allocate_cache(op.feop,r)
end

function TransientFETools.update_cache!(
  ode_cache,
  op::RBOperator,
  r::TransientParamRealization)

  update_cache!(ode_cache,op.feop,r)
end

function Algebra.allocate_residual(
  op::RBOperator,
  r::TransientParamRealization,
  x::AbstractVector,
  ode_cache)

  allocate_residual(op.feop,r,x,ode_cache)
end

function Algebra.allocate_jacobian(
  op::RBOperator,
  r::TransientParamRealization,
  x::AbstractVector,
  ode_cache)

  allocate_jacobian(op.feop,r,x,ode_cache)
end

function Algebra.allocate_jacobian(
  op::RBOperator,
  r::TransientParamRealization,
  x::AbstractVector,
  i::Integer,
  ode_cache)

  allocate_jacobian(op.feop,r,x,i,ode_cache)
end

function Algebra.residual!(
  b::AbstractVector,
  op::RBOperator,
  r::TransientParamRealization,
  xhF::Tuple{Vararg{AbstractVector}},
  ode_cache)

  residual!(b,op.feop,r,xhF,ode_cache)
end

function residual_for_trian!(
  b::AbstractVector,
  op::RBOperator,
  r::TransientParamRealization,
  xhF::Tuple{Vararg{AbstractVector}},
  ode_cache)

  residual_for_trian!(b,op.feop,r,xhF,ode_cache)
end

function Algebra.jacobian!(
  A::AbstractMatrix,
  op::RBOperator,
  r::TransientParamRealization,
  xhF::Tuple{Vararg{AbstractVector}},
  i::Integer,
  γᵢ::Real,
  ode_cache)

  jacobian!(A,op.feop,r,xhF,i,γᵢ,ode_cache)
end

function jacobian_for_trian!(
  A::AbstractMatrix,
  op::RBOperator,
  r::TransientParamRealization,
  xhF::Tuple{Vararg{AbstractVector}},
  i::Integer,
  γᵢ::Real,
  ode_cache)

  jacobian_for_trian!(A,op.feop,r,xhF,i,γᵢ,ode_cache)
end

function ODETools.jacobians!(
  A::AbstractMatrix,
  op::RBOperator,
  r::TransientParamRealization,
  xhF::Tuple{Vararg{AbstractVector}},
  γ::Tuple{Vararg{Real}},
  ode_cache)

  jacobians!(A,op.feop,r,xhF,γ,ode_cache)
end

function Algebra.zero_initial_guess(op::RBOperator,r::TransientParamRealization)
  x0 = zero_initial_guess(op.feop)
  allocate_param_array(x0,length(r))
end

function _init_free_values(op::RBOperator{Affine},r::TransientParamRealization)
  trial = evaluate(get_trial(op.feop),r)
  x = zero_free_values(trial)
  y = zero_free_values(trial)
  return x,y
end

function _init_free_values(op::RBOperator,r::TransientParamRealization)
  trial = evaluate(get_trial(op.feop),r)
  x = random_free_values(trial)
  y = zero_free_values(trial)
  return x,y
end

function FEM.get_method_operator(
  solver::RBSolver{ThetaMethod},
  op::RBOperator,
  r::TransientParamRealization)

  fesolver = get_fe_solver(solver)
  dt = fesolver.dt
  θ = fesolver.θ
  θ == 0.0 ? dtθ = dt : dtθ = dt*θ

  x,y = _init_free_values(op,r)

  ode_cache = allocate_cache(op,r)
  ode_cache = update_cache!(ode_cache,op,r)

  get_method_operator(odeop,r,dtθ,x,ode_cache,y)
end

function collect_residuals_and_jacobians(solver::RBSolver,op::RBOperator)
  nparams = num_mdeim_params(solver.info)
  r = realization(op.feop;nparams)

  nlop = get_method_operator(solver,op,r)
  x = nlop.u0

  b = residual_for_trian(nlop,x)
  A = jacobian_for_trian(nlop,x)

  return b,A
end
