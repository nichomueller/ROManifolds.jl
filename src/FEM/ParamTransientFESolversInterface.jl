abstract type ParamODEOperator{C} <: GridapType end

const AffineParamODEOperator = ParamODEOperator{Affine}

struct ParamODEOpFromFEOp{C} <: ParamODEOperator{C}
  feop::ParamTransientFEOperator{C}
end

get_order(op::ParamODEOpFromFEOp) = get_order(op.feop)

function allocate_cache(op::ParamODEOpFromFEOp,μ::AbstractArray,t::Real)
  n = isa(μ,Table) ? length(μ) : 1
  Ut = get_trial(op.feop)
  U = allocate_trial_space(Ut,n)
  Uts = (Ut,)
  Us = (U,)
  for i in 1:get_order(op)
    Uts = (Uts...,∂ₚt(Uts[i]))
    Us = (Us...,allocate_trial_space(Uts[i+1]))
  end
  fecache = allocate_cache(op.feop)
  ode_cache = (Us,Uts,fecache)
  ode_cache
end

function update_cache!(
  ode_cache,
  op::ParamODEOpFromFEOp,
  μ::AbstractArray,
  t::Real)

  _Us,Uts,fecache = ode_cache
  Us = ()
  for i in 1:get_order(op)+1
    Us = (Us...,evaluate!(_Us[i],Uts[i],μ,t))
  end
  fecache = update_cache!(fecache,op.feop,μ,t)
  (Us,Uts,fecache)
end

function allocate_residual(
  op::ParamODEOpFromFEOp,
  uhF::AbstractVector,
  ode_cache)

  Us,_,fecache = ode_cache
  uh = EvaluationFunction(Us[1],uhF)
  allocate_residual(op.feop,uh,fecache)
end

function allocate_jacobian(
  op::ParamODEOpFromFEOp,
  uhF::AbstractVector,
  ode_cache)

  Us,_,fecache = ode_cache
  uh = EvaluationFunction(Us[1],uhF)
  allocate_jacobian(op.feop,uh,fecache)
end

function residual!(
  b::AbstractArray,
  op::ParamODEOpFromFEOp,
  μ::AbstractArray,
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

function jacobian!(
  A::AbstractArray,
  op::ParamODEOpFromFEOp,
  μ::AbstractArray,
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

function jacobians!(
  J::AbstractArray,
  op::ParamODEOpFromFEOp,
  μ::AbstractArray,
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
