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
  r::Realization)

  Ut = get_trial(op.feop)
  U = allocate_trial_space(Ut,get_parameters(r),get_times(r))
  Uts = (Ut,)
  Us = (U,)
  for i in 1:get_order(op)
    Uts = (Uts...,∂ₚt(Uts[i]))
    Us = (Us...,allocate_trial_space(Uts[i+1],get_parameters(r),get_times(r)))
  end
  fecache = allocate_cache(op.feop)
  Us,Uts,fecache
end

function TransientFETools.update_cache!(
  ode_cache,
  op::PODEOpFromFEOp,
  r::Realization)

  _Us,Uts,fecache = ode_cache
  Us = ()
  for i in 1:get_order(op)+1
    Us = (Us...,evaluate!(_Us[i],Uts[i],get_parameters(r),get_times(r)))
  end
  fecache = allocate_cache(op.feop)
  Us,Uts,fecache
end

function Algebra.allocate_residual(
  op::PODEOpFromFEOp,
  r::Realization,
  x::AbstractVector,
  ode_cache)

  Us,Uts,fecache = ode_cache
  xh = EvaluationFunction(Us[1],x)
  allocate_residual(op.feop,r,xh,fecache)
end

function Algebra.allocate_jacobian(
  op::PODEOpFromFEOp,
  r::Realization,
  x::AbstractVector,
  ode_cache)

  Us,Uts,fecache = ode_cache
  xh = EvaluationFunction(Us[1],x)
  allocate_residual(op.feop,r,xh,fecache)
end

function Algebra.allocate_jacobian(
  op::PODEOpFromFEOp,
  r::Realization,
  x::AbstractVector,
  i::Integer,
  ode_cache)

  Us,Uts,fecache = ode_cache
  xh = EvaluationFunction(Us[1],x)
  allocate_residual(op.feop,r,xh,i)
end

function Algebra.residual!(
  b::AbstractVector,
  op::PODEOpFromFEOp,
  r::Realization,
  xhF::Tuple{Vararg{AbstractVector}},
  ode_cache)

  Xh, = ode_cache
  dxh = ()
  for i in 2:get_order(op)+1
    dxh = (dxh...,EvaluationFunction(Xh[i],xhF[i]))
  end
  xh=TransientCellField(EvaluationFunction(Xh[1],xhF[1]),dxh)
  residual!(b,op.feop,r,xh,ode_cache)
end

function residual_for_trian!(
  b::AbstractVector,
  op::PODEOpFromFEOp,
  r::Realization,
  xhF::Tuple{Vararg{AbstractVector}},
  ode_cache)

  Xh, = ode_cache
  dxh = ()
  for i in 2:get_order(op)+1
    dxh = (dxh...,EvaluationFunction(Xh[i],xhF[i]))
  end
  xh=TransientCellField(EvaluationFunction(Xh[1],xhF[1]),dxh)
  residual_for_trian!(b,op.feop,r,xh,ode_cache)
end

function Algebra.jacobian!(
  A::AbstractMatrix,
  op::PODEOpFromFEOp,
  r::Realization,
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
  jacobian!(A,op.feop,r,xh,i,γᵢ,ode_cache)
end

function jacobian_for_trian!(
  A::AbstractMatrix,
  op::PODEOpFromFEOp,
  r::Realization,
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
  jacobian_for_trian!(A,op.feop,r,xh,i,γᵢ,ode_cache)
end

function ODETools.jacobians!(
  A::AbstractMatrix,
  op::PODEOpFromFEOp,
  r::Realization,
  xhF::Tuple{Vararg{AbstractVector}},
  γ::Tuple{Vararg{Real}},
  ode_cache)

  Xh, = ode_cache
  dxh = ()
  for i in 2:get_order(op)+1
    dxh = (dxh...,EvaluationFunction(Xh[i],xhF[i]))
  end
  xh=TransientCellField(EvaluationFunction(Xh[1],xhF[1]),dxh)
  jacobians!(A,op.feop,r,xh,γ,ode_cache)
end
