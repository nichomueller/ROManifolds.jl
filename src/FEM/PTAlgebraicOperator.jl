abstract type PTAlgebraicOperator{T<:OperatorType} <: NonlinearOperator end

struct PTAffineOperator <: PTAlgebraicOperator{Affine}
  matrix::PTArray
  vector::PTArray
end

function get_at_offsets(x::PTArray,offsets::Vector{Int},row::Int)
  map(y->y[offsets[row]+1:offsets[row+1]],x)
end

function get_at_offsets(x::PTArray,offsets::Vector{Int},row::Int,col::Int)
  map(y->y[offsets[row]+1:offsets[row+1],offsets[col]+1:offsets[col+1]],x)
end

function Base.getindex(op::PTAlgebraicOperator,row::Int,col=:)
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

function update_ptoperator(op::PTAlgebraicOperator,x::PTArray)
  odeop,μ,tθ,dtθ,ode_cache,vθ = op.odeop,op.μ,op.tθ,op.dtθ,op.ode_cache,op.vθ
  get_ptoperator(odeop,μ,tθ,dtθ,x,ode_cache,vθ)
end

for f in (:linear_operator,:nonlinear_operator,:auxiliary_operator)
  @eval begin
    function $f(op::PTAlgebraicOperator)
      feop = $f(op.odeop.feop)
      odeop = get_algebraic_operator(feop)
      return get_ptoperator(odeop,op.μ,op.tθ,op.dtθ,op.u0,op.ode_cache,op.vθ)
    end
  end
end

struct PTThetaAffineMethodOperator <: PTAlgebraicOperator{Affine}
  odeop::AffinePODEOperator
  μ
  tθ
  dtθ::Float
  u0::PTArray
  ode_cache
  vθ::PTArray
end

function get_ptoperator(
  odeop::AffinePODEOperator,μ,tθ,dtθ::Float,u0::PTArray,ode_cache,vθ::PTArray)
  PTThetaAffineMethodOperator(odeop,μ,tθ,dtθ,u0,ode_cache,vθ)
end

struct PTThetaMethodOperator <: PTAlgebraicOperator{Nonlinear}
  odeop::PODEOperator
  μ
  tθ
  dtθ::Float
  u0::PTArray
  ode_cache
  vθ::PTArray
end

function get_ptoperator(
  odeop::PODEOperator,μ,tθ,dtθ::Float,u0::PTArray,ode_cache,vθ::PTArray)
  PTThetaMethodOperator(odeop,μ,tθ,dtθ,u0,ode_cache,vθ)
end

function Algebra.allocate_residual(
  op::PTAlgebraicOperator,
  x::PTArray)

  allocate_residual(op.odeop,op.μ,op.tθ,x,op.ode_cache)
end

function Algebra.allocate_jacobian(
  op::PTAlgebraicOperator,
  x::PTArray)

  allocate_jacobian(op.odeop,op.μ,op.tθ,x,op.ode_cache)
end

function Algebra.residual(op::PTAlgebraicOperator,x::PTArray,args...)
  b = allocate_residual(op,x)
  residual!(b,op,x,args...)
end

function Algebra.jacobian(op::PTAlgebraicOperator,x::PTArray,args...)
  A = allocate_jacobian(op,x)
  jacobian!(A,op,x,args...)
end
