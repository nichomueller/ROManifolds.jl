abstract type ParamODEOperator{C} <: GridapType end

const AffineParamODEOperator = ParamODEOperator{Affine}

struct ParamODEOpFromFEOp{C} <: ParamODEOperator{C}
  feop::ParamTransientFEOperator{C}
end

function Base.length(sol::ParamODESolution)
  get_time_ndofs(sol.solver)
end

get_order(op::ParamODEOpFromFEOp) = get_order(op.feop)

function allocate_cache(op::ParamODEOpFromFEOp)
  Ut = get_trial(op.feop)
  U = allocate_trial_space(Ut)
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

function allocate_cache(
  op::ParamODEOpFromFEOp,
  v::AbstractVector,
  a::AbstractVector)

  ode_cache = allocate_cache(op)
  (v,a,ode_cache)
end

function update_cache!(
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

function jacobian!(
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

function jacobians!(
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

function Arrays.return_value(sol::ParamODESolution)
  sol1 = iterate(sol)
  (uh1,_),_ = sol1
  uh1
end

function postprocess(sol::ParamODESolution,uF::AbstractArray)
  Uh = allocate_trial_space(get_trial(sol.op.feop))
  if isa(Uh,MultiFieldFESpace)
    blocks = map(1:length(Uh.spaces)) do i
      MultiField.restrict_to_field(Uh,uF,i)
    end
    return mortar(blocks)
  else
    return uF
  end
end
