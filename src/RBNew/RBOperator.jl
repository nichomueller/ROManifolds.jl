
struct RBOperator{T} <: ODEOperator{T}
  feop::ODEParamOperator{T}
  trial::TrialRBSpace
  test::TestRBSpace
end

ReferenceFEs.get_order(op::RBOperator) = get_order(op.feop)
FESpaces.get_test(op::RBOperator) = get_test(op.feop)
FESpaces.get_trial(op::RBOperator) = get_trial(op.feop)

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
