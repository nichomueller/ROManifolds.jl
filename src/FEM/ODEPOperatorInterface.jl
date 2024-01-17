abstract type ODEPOperator{C<:OperatorType} <: ODEOperator{C} end

const ConstantODEPOperator = ODEPOperator{Constant}
const ConstantMatrixODEPOperator = ODEPOperator{ConstantMatrix}
const AffineODEPOperator = ODEPOperator{Affine}
const NonlinearODEPOperator = ODEPOperator{Nonlinear}

struct ODEPOpFromFEOp{C} <: ODEPOperator{C}
  feop::TransientPFEOperator{C}
end

ReferenceFEs.get_order(op::ODEPOpFromFEOp) = get_order(op.feop)

function TransientFETools.allocate_cache(
  op::ODEPOpFromFEOp,
  r::TransientPRealization)

  Upt = get_trial(op.feop)
  U = allocate_trial_space(Upt,r)
  Uts = (Upt,)
  Us = (U,)
  for i in 1:get_order(op)
    Uts = (Uts...,∂ₚt(Uts[i]))
    Us = (Us...,allocate_trial_space(Uts[i+1],r))
  end
  fecache = allocate_cache(op.feop)
  Us,Uts,fecache
end

function TransientFETools.update_cache!(
  ode_cache,
  op::ODEPOpFromFEOp,
  r::TransientPRealization)

  _Us,Uts,fecache = ode_cache
  Us = ()
  for i in 1:get_order(op)+1
    Us = (Us...,evaluate!(_Us[i],Uts[i],r))
  end
  fecache = allocate_cache(op.feop)
  Us,Uts,fecache
end

function Algebra.allocate_residual(
  op::ODEPOpFromFEOp,
  r::TransientPRealization,
  x::AbstractVector,
  ode_cache)

  Us,Uts,fecache = ode_cache
  xh = EvaluationFunction(Us[1],x)
  allocate_residual(op.feop,r,xh,fecache)
end

function Algebra.allocate_jacobian(
  op::ODEPOpFromFEOp,
  r::TransientPRealization,
  x::AbstractVector,
  ode_cache)

  Us,Uts,fecache = ode_cache
  xh = EvaluationFunction(Us[1],x)
  allocate_residual(op.feop,r,xh,fecache)
end

function Algebra.allocate_jacobian(
  op::ODEPOpFromFEOp,
  r::TransientPRealization,
  x::AbstractVector,
  i::Integer,
  ode_cache)

  Us,Uts,fecache = ode_cache
  xh = EvaluationFunction(Us[1],x)
  allocate_residual(op.feop,r,xh,i)
end

function Algebra.residual!(
  b::AbstractVector,
  op::ODEPOpFromFEOp,
  r::TransientPRealization,
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
  op::ODEPOpFromFEOp,
  r::TransientPRealization,
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
  op::ODEPOpFromFEOp,
  r::TransientPRealization,
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
  op::ODEPOpFromFEOp,
  r::TransientPRealization,
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
  op::ODEPOpFromFEOp,
  r::TransientPRealization,
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
