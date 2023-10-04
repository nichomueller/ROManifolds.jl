abstract type PODEOperator{C} <: GridapType end

const AffinePODEOperator = PODEOperator{Affine}

struct PODEOpFromFEOp{C} <: PODEOperator{C}
  feop::PTFEOperator{C}
end

get_order(op::PODEOpFromFEOp) = get_order(op.feop)

function allocate_cache(op::PODEOpFromFEOp,μ::AbstractVector,t::T) where T
  Ut = get_trial(op.feop)
  U = allocate_trial_space(Ut,μ,t)
  Uts = (Ut,)
  Us = (U,)
  for i in 1:get_order(op)
    Uts = (Uts...,∂ₚt(Uts[i]))
    Us = (Us...,allocate_trial_space(Uts[i+1],μ,t))
  end
  fecache = allocate_cache(op.feop)
  ode_cache = (Us,Uts,fecache)
  ode_cache
end

function update_cache!(
  ode_cache,
  op::PODEOpFromFEOp,
  μ::AbstractVector,
  t::T) where T

  _Us,Uts,fecache = ode_cache
  Us = ()
  for i in 1:get_order(op)+1
    Us = (Us...,evaluate!(_Us[i],Uts[i],μ,t))
  end
  fecache = update_cache!(fecache,op.feop,μ,t)
  (Us,Uts,fecache)
end

function allocate_residual(
  op::PODEOpFromFEOp,
  μ::AbstractVector,
  t::T,
  uhF::PTArray,
  ode_cache) where T

  Us,_,fecache = ode_cache
  uh = EvaluationFunction(Us[1],uhF)
  allocate_residual(op.feop,μ,t,uh,fecache)
end

function allocate_jacobian(
  op::PODEOpFromFEOp,
  μ::AbstractVector,
  t::T,
  uhF::PTArray,
  ode_cache) where T

  Us,_,fecache = ode_cache
  uh = EvaluationFunction(Us[1],uhF)
  allocate_jacobian(op.feop,μ,t,uh,fecache)
end

for fun in (:residual!,:residual_for_trian!)
  @eval begin
    function $fun(
      b::PTArray,
      op::PODEOpFromFEOp,
      μ::AbstractVector,
      t::T,
      xhF::Tuple{Vararg{PTArray}},
      ode_cache,
      args...) where T

      Xh, = ode_cache
      dxh = ()
      for i in 2:get_order(op)+1
        dxh = (dxh...,EvaluationFunction(Xh[i],xhF[i]))
      end
      xh=TransientCellField(EvaluationFunction(Xh[1],xhF[1]),dxh)
      $fun(b,op.feop,μ,t,xh,ode_cache,args...)
    end
  end
end

for fun in (:jacobian!,:jacobian_for_trian!)
  @eval begin
    function $fun(
      A::PTArray,
      op::PODEOpFromFEOp,
      μ::AbstractVector,
      t::T,
      xhF::Tuple{Vararg{PTArray}},
      i::Integer,
      γᵢ::Real,
      ode_cache,
      args...) where T

      Xh, = ode_cache
      dxh = ()
      for i in 2:get_order(op)+1
        dxh = (dxh...,EvaluationFunction(Xh[i],xhF[i]))
      end
      xh=TransientCellField(EvaluationFunction(Xh[1],xhF[1]),dxh)
      $fun(A,op.feop,μ,t,xh,i,γᵢ,ode_cache,args...)
    end
  end
end

function jacobians!(
  J::PTArray,
  op::PODEOpFromFEOp,
  μ::AbstractVector,
  t::T,
  xhF::Tuple{Vararg{PTArray}},
  γ::Tuple{Vararg{Real}},
  ode_cache) where T

  Xh, = ode_cache
  dxh = ()
  for i in 2:get_order(op)+1
    dxh = (dxh...,EvaluationFunction(Xh[i],xhF[i]))
  end
  xh=TransientCellField(EvaluationFunction(Xh[1],xhF[1]),dxh)
  jacobians!(J,op.feop,μ,t,xh,γ,ode_cache)
end
