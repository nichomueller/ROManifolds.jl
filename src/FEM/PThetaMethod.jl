function solution_step!(
  uf::PTArray,
  solver::ThetaMethod,
  op::PODEOperator,
  μ::AbstractVector,
  u0::PTArray,
  t0::Real,
  cache)

  dt = solver.dt
  solver.θ == 0.0 ? dtθ = dt : dtθ = dt*solver.θ
  tθ = t0+dtθ

  if isnothing(cache)
    ode_cache = allocate_cache(op,μ,tθ)
    vθ = similar(u0)
    vθ .= 0.0
    nl_cache = nothing
  else
    ode_cache,vθ,nl_cache = cache
  end

  ode_cache = update_cache!(ode_cache,op,μ,tθ)

  nlop = PThetaMethodNonlinearOperator(op,μ,tθ,dtθ,u0,ode_cache,vθ)

  nl_cache = solve!(uf,solver.nls,nlop,nl_cache)

  if 0.0 < solver.θ < 1.0
    @. uf = uf*(1.0/solver.θ)-u0*((1-solver.θ)/solver.θ)
  end

  cache = (ode_cache,vθ,nl_cache)

  return (uf,cache)
end

"""
Nonlinear operator that represents the θ-method nonlinear operator at a
given time step, i.e., A(t,u_n+θ,(u_n+θ-u_n)/dt)
"""
struct PThetaMethodNonlinearOperator <: PNonlinearOperator
  odeop::PODEOperator
  μ
  tθ
  dtθ::Float
  u0::PTArray
  ode_cache
  vθ::PTArray
end

function Algebra.residual(op::PNonlinearOperator,x::PTArray)
  b = allocate_residual(op,x)
  residual!(b,op,x)
  b
end

function Algebra.residual!(
  b::PTArray,
  op::PThetaMethodNonlinearOperator,
  x::PTArray)

  uθ = x
  vθ = op.vθ
  @. vθ = (x-op.u0)/op.dtθ
  z = zero(eltype(b))
  fill!(b,z)
  residual!(b,op.odeop,op.μ,op.tθ,(uθ,vθ),op.ode_cache)
end

function Algebra.jacobian(op::PNonlinearOperator,x::PTArray,args...)
  A = allocate_jacobian(op,x)
  jacobian!(A,op,x,args...)
  A
end

function Algebra.jacobian!(
  A::PTArray,
  op::PThetaMethodNonlinearOperator,
  x::PTArray)

  uF = x
  vθ = op.vθ
  @. vθ = (x-op.u0)/op.dtθ
  z = zero(eltype(A))
  fillstored!(A,z)
  jacobians!(A,op.odeop,op.μ,op.tθ,(uF,vθ),(1.0,1/op.dtθ),op.ode_cache)
end

function Algebra.jacobian!(
  A::PTArray,
  op::PThetaMethodNonlinearOperator,
  x::PTArray,
  i::Int)

  uF = x
  vθ = op.vθ
  @. vθ = (x-op.u0)/op.dtθ
  z = zero(eltype(A))
  fillstored!(A,z)
  jacobian!(A,op.odeop,op.μ,op.tθ,(uF,vθ),i,(1.0,1/op.dtθ)[i],op.ode_cache)
end

function Algebra.allocate_residual(
  op::PThetaMethodNonlinearOperator,
  x::PTArray)

  allocate_residual(op.odeop,x,op.ode_cache)
end

function Algebra.allocate_jacobian(
  op::PThetaMethodNonlinearOperator,
  x::PTArray)

  allocate_jacobian(op.odeop,x,op.ode_cache)
end
