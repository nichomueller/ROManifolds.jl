abstract type ODEParamOperator{C<:OperatorType} <: ODEOperator{C} end

const ConstantODEParamOperator = ODEParamOperator{Constant}
const ConstantMatrixODEParamOperator = ODEParamOperator{ConstantMatrix}
const AffineODEParamOperator = ODEParamOperator{Affine}
const NonlinearODEParamOperator = ODEParamOperator{Nonlinear}

struct ODEParamOpFromFEOp{C} <: ODEParamOperator{C}
  feop::TransientParamFEOperator{C}
end

ReferenceFEs.get_order(op::ODEParamOpFromFEOp) = get_order(op.feop)

function TransientFETools.allocate_cache(
  op::ODEParamOpFromFEOp,
  r::TransientParamRealization)

  Upt = get_trial(op.feop)
  U = allocate_trial_space(Upt,r)
  Uts = (Upt,)
  Us = (U,)
  for i in 1:get_order(op)
    Uts = (Uts...,∂t(Uts[i]))
    Us = (Us...,allocate_trial_space(Uts[i+1],r))
  end
  fecache = allocate_cache(op.feop)
  Us,Uts,fecache
end

function TransientFETools.update_cache!(
  ode_cache,
  op::ODEParamOpFromFEOp,
  r::TransientParamRealization)

  _Us,Uts,fecache = ode_cache
  Us = ()
  for i in 1:get_order(op)+1
    Us = (Us...,evaluate!(_Us[i],Uts[i],r))
  end
  fecache = allocate_cache(op.feop)
  Us,Uts,fecache
end

function Algebra.allocate_residual(
  op::ODEParamOpFromFEOp,
  r::TransientParamRealization,
  x::AbstractVector,
  ode_cache)

  Us,Uts,fecache = ode_cache
  xh = EvaluationFunction(Us[1],x)
  allocate_residual(op.feop,r,xh,fecache)
end

function Algebra.allocate_jacobian(
  op::ODEParamOpFromFEOp,
  r::TransientParamRealization,
  x::AbstractVector,
  ode_cache)

  Us,Uts,fecache = ode_cache
  xh = EvaluationFunction(Us[1],x)
  allocate_jacobian(op.feop,r,xh,fecache)
end

function Algebra.allocate_jacobian(
  op::ODEParamOpFromFEOp,
  r::TransientParamRealization,
  x::AbstractVector,
  i::Integer,
  ode_cache)

  Us,Uts,fecache = ode_cache
  xh = EvaluationFunction(Us[1],x)
  allocate_residual(op.feop,r,xh,i)
end

function Algebra.residual!(
  b::AbstractVector,
  op::ODEParamOpFromFEOp,
  r::TransientParamRealization,
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
  op::ODEParamOpFromFEOp,
  r::TransientParamRealization,
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
  op::ODEParamOpFromFEOp,
  r::TransientParamRealization,
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
  op::ODEParamOpFromFEOp,
  r::TransientParamRealization,
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
  op::ODEParamOpFromFEOp,
  r::TransientParamRealization,
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
