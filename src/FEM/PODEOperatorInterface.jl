abstract type PODEOperator{C<:OperatorType} <: ODEOperator{C} end

const ConstantPODEOperator = PODEOperator{Constant}
const ConstantMatrixPODEOperator = PODEOperator{ConstantMatrix}
const AffinePODEOperator = PODEOperator{Affine}
const NonlinearPODEOperator = PODEOperator{Nonlinear}

struct PODEOpFromFEOp{C} <: PODEOperator{C}
  feop::TransientPFEOperator{C}
end

TransientFETools.get_order(op::PODEOpFromFEOp) = get_order(op.feop)

function TransientFETools.allocate_cache(
  op::PODEOpFromFEOp,
  μ::AbstractVector,
  t)

  Ut = get_trial(op.feop)
  U = allocate_trial_space(Ut,μ,t)
  Uts = (Ut,)
  Us = (U,)
  for i in 1:get_order(op)
    Uts = (Uts...,∂ₚt(Uts[i]))
    Us = (Us...,allocate_trial_space(Uts[i+1],μ,t))
  end
  fecache = allocate_cache(op.feop)
  Us,Uts,fecache
end

function TransientFETools.update_cache!(
  ode_cache,
  op::PODEOpFromFEOp,
  μ::AbstractVector,
  t)

  _Us,Uts,fecache = ode_cache
  Us = ()
  for i in 1:get_order(op)+1
    Us = (Us...,evaluate!(_Us[i],Uts[i],μ,t))
  end
  fecache = allocate_cache(op.feop)
  Us,Uts,fecache
end

function Algebra.allocate_residual(op::PODEOpFromFEOp,μ,t,x::AbstractVector,ode_cache)
  Us,Uts,fecache = ode_cache
  xh = EvaluationFunction(Us[1],x)
  allocate_residual(op.feop,μ,t,xh,fecache)
end

function Algebra.allocate_residual(op::PODEOpFromFEOp,μ,t,x::AbstractVector,ode_cache)
  Us, = ode_cache
  xh = EvaluationFunction(Us[1],x)
  allocate_residual(op.feop,μ,t,xh,ode_cache)
end

function Algebra.allocate_jacobian(
  op::TransientPFEOperator,
  μ::P,
  t::T,
  xh,
  i::Integer) where {P,T}

  trial = get_trial(op)(μ,t)
  test = get_test(op)
  u = get_trial_fe_basis(trial)
  v = get_fe_basis(test)
  dc = op.jacs[i](μ,t,xh,u,v)
  matdata = collect_cell_matrix(trial,test,dc)
  allocate_matrix(op.assem,matdata)
end
