function Gridap.ODEs.TransientFETools.solve_step!(
  uf::AbstractVector,
  solver::ThetaMethod,
  op::ParamODEOperator,
  μ::AbstractVector,
  u0::AbstractVector,
  t0::Real,
  cache)

  dt = solver.dt
  solver.θ == 0.0 ? dtθ = dt : dtθ = dt*solver.θ
  tθ = t0+dtθ

  if isnothing(cache)
    ode_cache = allocate_cache(op)
    vθ = similar(u0)
    nl_cache = nothing
  else
    ode_cache,vθ,nl_cache = cache
  end

  ode_cache = update_cache!(ode_cache,op,μ,tθ)

  nlop = ParamThetaMethodNonlinearOperator(op,μ,tθ,dtθ,u0,ode_cache,vθ)

  nl_cache = solve!(uf,solver.nls,nlop,nl_cache)

  if 0.0 < solver.θ < 1.0
    uf = uf*(1.0/solver.θ)-u0*((1-solver.θ)/solver.θ)
  end

  cache = (ode_cache,vθ,nl_cache)

  tf = t0+dt
  return (uf,tf,cache)
end

"""
Nonlinear operator that represents the θ-method nonlinear operator at a
given time step, i.e., A(t,u_n+θ,(u_n+θ-u_n)/dt)
"""
struct ParamThetaMethodNonlinearOperator <: NonlinearOperator
  odeop::ParamODEOperator
  μ::AbstractVector
  tθ::Float64
  dtθ::Float64
  u0::AbstractVector
  ode_cache
  vθ::AbstractVector
end

function Gridap.ODEs.TransientFETools.residual!(
  b::AbstractVector,
  op::ParamThetaMethodNonlinearOperator,
  x::AbstractVector)

  uθ = x
  vθ = op.vθ
  vθ = (x-op.u0)/op.dtθ
  residual!(b,op.odeop,op.μ,op.tθ,(uθ,vθ),op.ode_cache)
end

function Gridap.ODEs.TransientFETools.jacobian!(
  A::AbstractMatrix,
  op::ParamThetaMethodNonlinearOperator,
  x::AbstractVector)

  uF = x
  vθ = op.vθ
  vθ = (x-op.u0)/op.dtθ
  z = zero(eltype(A))
  LinearAlgebra.fillstored!(A,z)
  jacobians!(A,op.odeop,op.μ,op.tθ,(uF,vθ),(1.0,1/op.dtθ),op.ode_cache)
end

function Gridap.ODEs.ODETools.allocate_residual(
  op::ParamThetaMethodNonlinearOperator,
  x::AbstractVector)

  allocate_residual(op.odeop,x,op.ode_cache)
end

function Gridap.ODEs.ODETools.allocate_jacobian(
  op::ParamThetaMethodNonlinearOperator,
  x::AbstractVector)

  allocate_jacobian(op.odeop,x,op.ode_cache)
end

function zero_initial_guess(op::ParamThetaMethodNonlinearOperator)
  x0 = similar(op.u0)
  fill!(x0,zero(eltype(x0)))
  x0
end
