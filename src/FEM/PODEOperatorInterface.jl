abstract type PODEOperator{C} <: GridapType end

const AffinePODEOperator = PODEOperator{Affine}

struct PODEOpFromFEOp{C} <: PODEOperator{C}
  feop::PTFEOperator{C}
end

ReferenceFEs.get_order(op::PODEOperator) = get_order(op.feop)

function TransientFETools.allocate_cache(op::PODEOperator,μ::AbstractVector,t::T) where T
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

function Gridap.ODEs.TransientFETools.update_cache!(
  ode_cache,
  op::PODEOperator,
  μ::AbstractVector,
  t::T) where T

  _Us,Uts,fecache = ode_cache
  Us = ()
  for i in 1:get_order(op)+1
    Us = (Us...,evaluate!(_Us[i],Uts[i],μ,t))
  end
  fecache = update_cache!(fecache,op.feop,μ,t)
  Us,Uts,fecache
end

function cache_at_idx(ode_cache,idx::Int)
  _Us,_Uts,fecache = ode_cache
  Us,Uts = (),()
  for i in eachindex(_Us)
    Us = (Us...,_Us[i][idx])
    Uts = (Uts...,_Uts[i][idx])
  end
  Us,Uts,fecache
end

function Gridap.Algebra.allocate_residual(
  op::PODEOperator,
  μ::AbstractVector,
  t::T,
  uhF::PTArray,
  ode_cache) where T

  Us,_,fecache = ode_cache
  uh = EvaluationFunction(Us[1],uhF)
  allocate_residual(op.feop,μ,t,uh,fecache)
end

function Gridap.Algebra.allocate_jacobian(
  op::PODEOperator,
  μ::AbstractVector,
  t::T,
  uhF::PTArray,
  i::Integer,
  ode_cache) where T

  Us,_,fecache = ode_cache
  uh = EvaluationFunction(Us[1],uhF)
  allocate_jacobian(op.feop,μ,t,uh,i,fecache)
end

for fun in (:(Algebra.residual!),:residual_for_trian!)
  @eval begin
    function $fun(
      b::PTArray,
      op::PODEOperator,
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
      xh = TransientCellField(EvaluationFunction(Xh[1],xhF[1]),dxh)
      $fun(b,op.feop,μ,t,xh,ode_cache,args...)
    end
  end
end

for fun in (:(Algebra.jacobian!),:jacobian_for_trian!)
  @eval begin
    function $fun(
      A::PTArray,
      op::PODEOperator,
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
      xh = TransientCellField(EvaluationFunction(Xh[1],xhF[1]),dxh)
      $fun(A,op.feop,μ,t,xh,i,γᵢ,ode_cache,args...)
    end
  end
end

function ODETools.jacobians!(
  J::PTArray,
  op::PODEOperator,
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
  xh = TransientCellField(EvaluationFunction(Xh[1],xhF[1]),dxh)
  jacobians!(J,op.feop,μ,t,xh,γ,ode_cache)
end
