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
    ode_cache = allocate_cache(op,μ)
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
  μ::AbstractVector
  tθ::Float64
  dtθ::Float64
  u0::PTArray
  ode_cache
  vθ::PTArray
end

function residual!(
  b::PTArray,
  op::PThetaMethodNonlinearOperator,
  x::PTArray)

  uθ = x
  vθ = op.vθ
  @. vθ = (x-op.u0)/op.dtθ
  residual!(b,op.odeop,op.μ,op.tθ,(uθ,vθ),op.ode_cache)
end

function jacobian!(
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

function allocate_residual(
  op::PThetaMethodNonlinearOperator,
  x::PTArray)

  allocate_residual(op.odeop,x,op.ode_cache)
end

function allocate_jacobian(
  op::PThetaMethodNonlinearOperator,
  x::PTArray)

  allocate_jacobian(op.odeop,x,op.ode_cache)
end
