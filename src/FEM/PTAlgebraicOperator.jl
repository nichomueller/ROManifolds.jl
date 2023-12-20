abstract type PTNonlinearOperator <: NonlinearOperator end
abstract type PTAlgebraicOperator{T<:OperatorType} <: PTNonlinearOperator end

function Base.getindex(op::PTAlgebraicOperator,row::Int,col=:)
  feop = op.feop
  offsets = field_offsets(feop.test)
  feop_idx = feop[row,col]
  odeop_idx = get_algebraic_operator(feop_idx)
  if isa(col,Colon)
    return get_algebraic_operator(odeop_idx,op.μ,op.t,op.dtθ,op.u0,op.ode_cache,op.vθ)
  else
    u0_idx = get_at_offsets(op.u0,offsets,col)
    vθ_idx = get_at_offsets(op.vθ,offsets,col)
    ode_cache_idx = cache_at_idx(op.ode_cache,col)
    return get_algebraic_operator(odeop_idx,op.μ,op.t,op.dtθ,u0_idx,ode_cache_idx,vθ_idx)
  end
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

function Algebra.allocate_residual(op::PTAlgebraicOperator,x::AbstractVector)
  Xh, = op.ode_cache
  uh = EvaluationFunction(Xh[1],x)
  dxh = ()
  for _ in 1:get_order(op.feop)
    dxh = (dxh...,uh)
  end
  xh = TransientCellField(uh,dxh)
  allocate_residual(op.feop,op.μ,op.t,xh)
end

function Algebra.allocate_jacobian(op::PTAlgebraicOperator,x::AbstractVector,i=1)
  Xh, = op.ode_cache
  uh = EvaluationFunction(Xh[1],x)
  dxh = ()
  for _ in 1:get_order(op.feop)
    dxh = (dxh...,uh)
  end
  xh = TransientCellField(uh,dxh)
  allocate_jacobian(op.feop,op.μ,op.t,xh,i)
end

for fun in (:(Algebra.residual!),:residual_for_trian!)
  @eval begin
    function $fun(
      b::AbstractVector,
      op::PTAlgebraicOperator,
      xhF::Tuple{Vararg{AbstractVector}},
      args...)

      Xh, = op.ode_cache
      dxh = ()
      for i in 2:get_order(op.feop)+1
        dxh = (dxh...,EvaluationFunction(Xh[i],xhF[i]))
      end
      xh = TransientCellField(EvaluationFunction(Xh[1],xhF[1]),dxh)
      $fun(b,op.feop,op.μ,op.t,xh,args...)
    end
  end
end

for fun in (:(Algebra.jacobian!),:jacobian_for_trian!)
  @eval begin
    function $fun(
      A::AbstractMatrix,
      op::PTAlgebraicOperator,
      xhF::Tuple{Vararg{AbstractVector}},
      args...)

      Xh, = op.ode_cache
      dxh = ()
      for i in 2:get_order(op.feop)+1
        dxh = (dxh...,EvaluationFunction(Xh[i],xhF[i]))
      end
      xh = TransientCellField(EvaluationFunction(Xh[1],xhF[1]),dxh)
      $fun(A,op.feop,op.μ,op.t,xh,args...)
    end
  end
end

function ODETools.jacobians!(
  A::AbstractMatrix,
  op::PTAlgebraicOperator,
  xhF::Tuple{Vararg{AbstractVector}},
  args...)

  Xh, = op.ode_cache
  dxh = ()
  for i in 2:get_order(op.feop)+1
    dxh = (dxh...,EvaluationFunction(Xh[i],xhF[i]))
  end
  xh = TransientCellField(EvaluationFunction(Xh[1],xhF[1]),dxh)
  jacobians!(A,op.feop,op.μ,op.t,xh,args...)
end

function update_algebraic_operator(op::PTAlgebraicOperator,x::AbstractVector)
  @unpack feop,μ,t,dtθ,u0,ode_cache,vθ = op
  get_algebraic_operator(feop,μ,t,dtθ,x,ode_cache,vθ)
end
