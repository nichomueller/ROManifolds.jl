function Gridap.ODEs.ODETools.solve_step!(
  uf::AbstractVector,
  solver::ThetaMethod,
  op::AffineParamODEOperator,
  μ::Vector{Float},
  u0::AbstractVector,
  t0::Real,
  cache) # -> (uF,tF)

  dt = solver.dt
  solver.θ == 0.0 ? dtθ = dt : dtθ = dt*solver.θ
  tθ = t0+dtθ

  if isnothing(cache)
    ode_cache = allocate_cache(op)
    vθ = similar(u0)
    vθ .= 0.0
    l_cache = nothing
    A,b = _allocate_matrix_and_vector(op,u0,ode_cache)
  else
    ode_cache,vθ,A,b,l_cache = cache
  end

  ode_cache = Gridap.ODEs.TransientFETools.update_cache!(ode_cache,op,μ,tθ)

  _matrix_and_vector!(A,b,op,μ,tθ,dtθ,u0,ode_cache,vθ)
  afop = Gridap.Algebra.AffineOperator(A,b)

  newmatrix = true
  l_cache = Gridap.ODEs.ODETools.solve!(uf,solver.nls,afop,l_cache,newmatrix)

  uf = uf + u0
  if 0.0 < solver.θ < 1.0
    uf = uf*(1.0/solver.θ)-u0*((1-solver.θ)/solver.θ)
  end

  cache = (ode_cache,vθ,A,b,l_cache)

  tf = t0+dt
  (uf,tf,cache)
end

"""
Affine operator that represents the θ-method affine operator at a
given time step, i.e., M(t)(u_n+θ-u_n)/dt + K(t)u_n+θ + b(t)
"""
function ParamThetaMethodAffineOperator(
  odeop::AffineParamODEOperator,
  μ::Vector{Float},
  tθ::Float64,
  dtθ::Float64,
  u0::AbstractVector,
  ode_cache,
  vθ::AbstractVector)

  A,b = _allocate_matrix_and_vector(odeop,u0,ode_cache)
  _matrix_and_vector!(A,b,odeop,μ,tθ,dtθ,u0,ode_cache,vθ)
  Gridap.Algebra.AffineOperator(A,b)
end

function _matrix_and_vector!(A,b,odeop,μ,tθ,dtθ,u0,ode_cache,vθ)
  _matrix!(A,odeop,μ,tθ,dtθ,u0,ode_cache,vθ)
  _vector!(b,odeop,μ,tθ,dtθ,u0,ode_cache,vθ)
end

function _matrix!(A,odeop,μ,tθ,dtθ,u0,ode_cache,vθ)
  z = zero(eltype(A))
  LinearAlgebra.fillstored!(A,z)
  Gridap.ODEs.TransientFETools.jacobians!(A,odeop,μ,tθ,(vθ,vθ),(1.0,1/dtθ),ode_cache)
end

function _mass_matrix!(A,odeop,μ,tθ,dtθ,u0,ode_cache,vθ)
  z = zero(eltype(A))
  LinearAlgebra.fillstored!(A,z)
  Gridap.ODEs.TransientFETools.jacobian!(A,odeop,μ,tθ,(vθ,vθ),2,(1/dtθ),ode_cache)
end

function _vector!(b,odeop,μ,tθ,dtθ,u0,ode_cache,vθ)
  Gridap.ODEs.TransientFETools.residual!(b,odeop,μ,tθ,(u0,vθ),ode_cache)
  b .*= -1.0
end

function _allocate_matrix(odeop,u0,ode_cache)
  A = allocate_jacobian(odeop,u0,ode_cache)
  A
end

function _allocate_matrix_and_vector(odeop,u0,ode_cache)
  b = allocate_residual(odeop,u0,ode_cache)
  A = allocate_jacobian(odeop,u0,ode_cache)
  A,b
end
