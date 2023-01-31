function Gridap.ODEs.TransientFETools.solve_step!(
  uf::AbstractVector,
  solver::BDF,
  op::ParamODEOperator,
  μ::Param,
  u0::AbstractVector,
  u1::AbstractVector,
  t1::Real,
  cache)

  dt = solver.dt
  t2 = t1+dt

  if isnothing(cache)
    ode_cache = allocate_cache(op)
    v0,v1 = similar(u0),similar(u1)
    nl_cache = nothing
  else
    ode_cache,v0,v1,nl_cache = cache
  end

  ode_cache = update_cache!(ode_cache,op,μ,t2)

  nlop = ParamBDFNonlinearOperator(op,μ,t2,u1,u0,ode_cache)

  nl_cache = solve!(uf,solver.nls,nlop,nl_cache)

  uf = uf-4/3*u1+1/3*u0

  cache = (ode_cache,vθ,nl_cache)

  tf = t0+dt
  return (uf,tf,cache)
end

"""
Nonlinear operator that represents the BDF nonlinear operator at a
given time step
"""
struct ParamBDFNonlinearOperator <: NonlinearOperator
  odeop::ParamODEOperator
  μ::Param
  tθ::Float64
  dtθ::Float64
  u0::AbstractVector
  ode_cache
  vθ::AbstractVector
end

function Gridap.ODEs.TransientFETools.residual!(
  b::AbstractVector,
  op::ParamBDFNonlinearOperator,
  x::AbstractVector)

  uθ = x
  vθ = op.vθ
  vθ = (x-op.u0)/op.dtθ
  residual!(b,op.odeop,op.μ,op.tθ,(uθ,vθ),op.ode_cache)
end

function Gridap.ODEs.TransientFETools.jacobian!(
  A::AbstractMatrix,
  op::ParamBDFNonlinearOperator,
  x::AbstractVector)

  uF = x
  vθ = op.vθ
  vθ = (x-op.u0)/op.dtθ
  z = zero(eltype(A))
  LinearAlgebra.fillstored!(A,z)
  jacobians!(A,op.odeop,op.μ,op.tθ,(uF,vθ),(1.0,1/op.dtθ),op.ode_cache)
end

function Gridap.ODEs.ODETools.allocate_residual(
  op::ParamBDFNonlinearOperator,
  x::AbstractVector)

  allocate_residual(op.odeop,x,op.ode_cache)
end

function Gridap.ODEs.ODETools.allocate_jacobian(
  op::ParamBDFNonlinearOperator,
  x::AbstractVector)

  allocate_jacobian(op.odeop,x,op.ode_cache)
end

function zero_initial_guess(op::ParamBDFNonlinearOperator)
  x0 = similar(op.u0)
  fill!(x0,zero(eltype(x0)))
  x0
end
