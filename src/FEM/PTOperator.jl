abstract type PTOperator{T<:OperatorType} <: NonlinearOperator end

function Base.getindex(op::PTOperator,row::Int,col=:)
  feop = op.odeop.feop
  offsets = field_offsets(feop.test)
  feop_idx = feop[row,col]
  odeop_idx = get_algebraic_operator(feop_idx)
  if isa(col,Colon)
    return get_ptoperator(odeop_idx,op.μ,op.tθ,op.dtθ,op.u0,op.ode_cache,op.vθ)
  else
    u0_idx = get_at_offsets(op.u0,offsets,col)
    vθ_idx = get_at_offsets(op.vθ,offsets,col)
    ode_cache_idx = cache_at_idx(op.ode_cache,col)
    return get_ptoperator(odeop_idx,op.μ,op.tθ,op.dtθ,u0_idx,ode_cache_idx,vθ_idx)
  end
end

for f in (:linear_operator,:nonlinear_operator,:auxiliary_operator)
  @eval begin
    function $f(op::PTOperator)
      feop = $f(op.odeop.feop)
      odeop = get_algebraic_operator(feop)
      return get_ptoperator(odeop,op.μ,op.tθ,op.dtθ,op.u0,op.ode_cache,op.vθ)
    end
  end
end

function Algebra.allocate_residual(op::PTOperator,x::AbstractVector)
  allocate_residual(op.odeop,op.μ,op.tθ,x,op.ode_cache)
end

function Algebra.allocate_jacobian(op::PTOperator,x::AbstractVector,i=1)
  allocate_jacobian(op.odeop,op.μ,op.tθ,x,i,op.ode_cache)
end

function update_ptoperator(op::PTOperator,x::AbstractVector)
  @unpack odeop,μ,tθ,dtθ,u0,ode_cache,vθ = op
  get_ptoperator(odeop,μ,tθ,dtθ,x,ode_cache,vθ)
end
