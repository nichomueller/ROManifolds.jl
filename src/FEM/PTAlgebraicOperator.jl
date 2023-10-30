abstract type PTAlgebraicOperator{T<:OperatorType} <: NonlinearOperator end

struct PTAffineOperator <: PTAlgebraicOperator{Affine}
  matrix::PTArray
  vector::PTArray
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

for f in (:linear_operator,:nonlinear_operator)
  @eval begin
    function $f(op::PTThetaMethodOperator)
      feop = $f(op.odeop.feop)
      odeop = get_algebraic_operator(feop)
      return PTThetaMethodOperator(odeop,op.μ,op.tθ,op.dtθ,op.u0,op.ode_cache,op.vθ)
    end
  end
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
