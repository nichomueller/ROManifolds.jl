function solve_step!(
  uf::AbstractVector,
  solver::θMethod,
  op::AffineParamODEOperator,
  μ::AbstractVector,
  u0::AbstractVector,
  t0::Real,
  cache)

  dt = solver.dt
  solver.θ == 0.0 ? dtθ = dt : dtθ = dt*solver.θ
  tθ = t0+dtθ

  if isnothing(cache)
    ode_cache = isa(μ,Table) ? allocate_cache(op,length(μ)) : allocate_cache(op)
    vθ = allocate_intermediate_step(u0)
    l_cache = nothing
    A,b = _allocate_matrix_and_vector(op,u0,ode_cache)
  else
    ode_cache,vθ,A,b,l_cache = cache
  end

  ode_cache = update_cache!(ode_cache,op,μ,tθ)

  _matrix_and_vector!(A,b,op,μ,tθ,dtθ,u0,ode_cache,vθ)
  afop = AffineOperator(A,b)

  newmatrix = true
  l_cache = solve!(uf,solver.nls,afop,l_cache,newmatrix)

  uf = uf + u0
  if 0.0 < solver.θ < 1.0
    uf = uf*(1.0/solver.θ)-u0*((1-solver.θ)/solver.θ)
  end

  cache = (ode_cache,vθ,A,b,l_cache)

  return (uf,cache)
end

function _matrix_and_vector!(
  A::AbstractArray,
  b::AbstractArray,
  op::AffineParamODEOperator,
  μ::AbstractArray,
  tθ::Real,
  dtθ::Real,
  u0,
  ode_cache,
  vθ)

  _matrix!(A,op,μ,tθ,dtθ,u0,ode_cache,vθ)
  _vector!(b,op,μ,tθ,dtθ,u0,ode_cache,vθ)
end

function _matrix!(
  A::AbstractArray,
  op::AffineParamODEOperator,
  μ::AbstractArray,
  tθ::Real,
  dtθ::Real,
  u0,
  ode_cache,
  vθ)

  fill_with_zeros!(A)
  jacobians!(A,op,μ,tθ,(vθ,vθ),(1.0,1/dtθ),ode_cache)
end

function _mass_matrix!(
  A::AbstractArray,
  op::AffineParamODEOperator,
  μ::AbstractArray,
  tθ::Real,
  dtθ::Real,
  u0,
  ode_cache,
  vθ)

  fill_with_zeros!(A)
  jacobian!(A,op,μ,tθ,(vθ,vθ),2,(1/dtθ),ode_cache)
end

function _vector!(
  b::AbstractArray,
  op::AffineParamODEOperator,
  μ::AbstractArray,
  tθ::Real,
  ::Real,
  u0,
  ode_cache,
  vθ)

  residual!(b,op,μ,tθ,(u0,vθ),ode_cache)
  b .*= -1.0
end

function _allocate_matrix(
  op::AffineParamODEOperator,
  u0,
  ode_cache)

  allocate_jacobian(op,u0,ode_cache)
end

function _allocate_matrix_and_vector(
  op::AffineParamODEOperator,
  u0,
  ode_cache)

  b = allocate_residual(op,u0,ode_cache)
  A = allocate_jacobian(op,u0,ode_cache)
  A,b
end
