# Interface

abstract type ParamODEOperator{C} <: GridapType end
const AffineParamODEOperator = ParamODEOperator{Affine}

"""
A wrapper of `ParamTransientFEOperator` that transforms it to `ParamODEOperator`, i.e.,
takes A(μ,t,uh,∂tuh,∂t^2uh,...,∂t^Nuh,vh) and returns A(μ,t,uF,∂tuF,...,∂t^NuF)
where uF,∂tuF,...,∂t^NuF represent the free values of the `EvaluationFunction`
uh,∂tuh,∂t^2uh,...,∂t^Nuh.
"""
struct ParamODEOpFromFEOp{C} <: ParamODEOperator{C}
  feop::ParamTransientFEOperator{C}
end

Gridap.ODEs.TransientFETools.get_order(op::ParamODEOpFromFEOp) =
  get_order(op.feop)

function Gridap.ODEs.TransientFETools.allocate_cache(op::ParamODEOpFromFEOp)
  Ut = get_trial(op.feop)
  U = allocate_trial_space(Ut)
  Uts = (Ut,)
  Us = (U,)
  for i in 1:get_order(op)
    Uts = (Uts...,∂t(Uts[i]))
    Us = (Us...,allocate_trial_space(Uts[i+1]))
  end
  fecache = allocate_cache(op.feop)
  ode_cache = (Us,Uts,fecache)
  ode_cache
end

function Gridap.ODEs.TransientFETools.allocate_cache(
  op::ParamODEOpFromFEOp,
  v::AbstractVector,
  a::AbstractVector)

  ode_cache = allocate_cache(op)
  (v,a,ode_cache)
end

function Gridap.ODEs.TransientFETools.update_cache!(
  ode_cache,
  op::ParamODEOpFromFEOp,
  μ::AbstractVector,
  t::Real)

  _Us,Uts,fecache = ode_cache
  Us = ()
  for i in 1:get_order(op)+1
    Us = (Us...,evaluate!(_Us[i],Uts[i],μ,t))
  end
  fecache = update_cache!(fecache,op.feop,μ,t)
  (Us,Uts,fecache)
end

function Gridap.ODEs.TransientFETools.allocate_residual(
  op::ParamODEOpFromFEOp,
  uhF::AbstractVector,
  ode_cache)

  Us,_,fecache = ode_cache
  uh = EvaluationFunction(Us[1],uhF)
  allocate_residual(op.feop,uh,fecache)
end

function Gridap.ODEs.TransientFETools.allocate_jacobian(
  op::ParamODEOpFromFEOp,
  uhF::AbstractVector,
  ode_cache)

  Us,_,fecache = ode_cache
  uh = EvaluationFunction(Us[1],uhF)
  allocate_jacobian(op.feop,uh,fecache)
end

"""
It provides A(t,uh,∂tuh,...,∂t^Nuh) for a given (t,uh,∂tuh,...,∂t^Nuh)
"""
function Gridap.ODEs.TransientFETools.residual!(
  b::AbstractVector,
  op::ParamODEOpFromFEOp,
  μ::AbstractVector,
  t::Real,
  xhF::Tuple{Vararg{AbstractVector}},
  ode_cache)
  Xh, = ode_cache
  dxh = ()
  for i in 2:get_order(op)+1
    dxh = (dxh...,EvaluationFunction(Xh[i],xhF[i]))
  end
  xh=TransientCellField(EvaluationFunction(Xh[1],xhF[1]),dxh)
  residual!(b,op.feop,μ,t,xh,ode_cache)
end

"""
It adds contribution to the Jacobian with respect to the i-th time derivative,
with i=0,...,N. That is, adding γ_i*[∂A/∂(∂t^iuh)](t,uh,∂tuh,...,∂t^Nuh) for a
given (t,uh,∂tuh,...,∂t^Nuh) to a given matrix J, where γ_i is a scaling coefficient
provided by the `ODESolver`, e.g., 1/Δt for Backward Euler; It represents
∂(δt^i(uh))/∂(uh), in which δt^i(⋅) is the approximation of ∂t^i(⋅) in the solver.
Note that for i=0, γ_i=1.0.
"""
function Gridap.ODEs.TransientFETools.jacobian!(
  A::AbstractMatrix,
  op::ParamODEOpFromFEOp,
  μ::AbstractVector,
  t::Real,
  xhF::Tuple{Vararg{AbstractVector}},
  i::Integer,
  γᵢ::Real,
  ode_cache)
  Xh, = ode_cache
  dxh = ()
  for i in 2:get_order(op)+1
    dxh = (dxh...,EvaluationFunction(Xh[i],xhF[i]))
  end
  xh=TransientCellField(EvaluationFunction(Xh[1],xhF[1]),dxh)
  jacobian!(A,op.feop,μ,t,xh,i,γᵢ,ode_cache)
end

"""
Add the contribution of all jacobians ,i.e., ∑ᵢ γ_i*[∂A/∂(∂t^iuh)](t,uh,∂tuh,...,∂t^Nuh,vh)
"""
function Gridap.ODEs.TransientFETools.jacobians!(
  J::AbstractMatrix,
  op::ParamODEOpFromFEOp,
  μ::AbstractVector,
  t::Real,
  xhF::Tuple{Vararg{AbstractVector}},
  γ::Tuple{Vararg{Real}},
  ode_cache)
  Xh, = ode_cache
  dxh = ()
  for i in 2:get_order(op)+1
    dxh = (dxh...,EvaluationFunction(Xh[i],xhF[i]))
  end
  xh=TransientCellField(EvaluationFunction(Xh[1],xhF[1]),dxh)
  jacobians!(J,op.feop,μ,t,xh,γ,ode_cache)
end
