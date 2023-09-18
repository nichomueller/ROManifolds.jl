function solution_step!(
  uf::PTArray,
  solver::ThetaMethod,
  op::AffinePODEOperator,
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
    l_cache = nothing
    A,b = _allocate_matrix_and_vector(op,u0,ode_cache)
  else
    ode_cache,vθ,A,b,l_cache = cache
  end

  ode_cache = update_cache!(ode_cache,op,μ,tθ)

  _matrix_and_vector!(A,b,op,μ,tθ,dtθ,u0,ode_cache,vθ)
  afop = PAffineOperator(A,b)

  l_cache = solve!(uf,solver.nls,afop,l_cache)

  uf = uf + u0
  if 0.0 < solver.θ < 1.0
    @. uf = uf*(1.0/solver.θ)-u0*((1-solver.θ)/solver.θ)
  end

  cache = (ode_cache,vθ,A,b,l_cache)
  tf = t0+dt
  return (uf,tf,cache)
end

function _matrix_and_vector!(
  A::PTArray,
  b::PTArray,
  op::AffinePODEOperator,
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
  A::PTArray,
  op::AffinePODEOperator,
  μ::AbstractArray,
  tθ::Real,
  dtθ::Real,
  u0,
  ode_cache,
  vθ)

  z = zero(eltype(A))
  fillstored!(A,z)
  jacobians!(A,op,μ,tθ,(vθ,vθ),(1.0,1/dtθ),ode_cache)
end

function _vector!(
  b::PTArray,
  op::AffinePODEOperator,
  μ::AbstractArray,
  tθ::Real,
  ::Real,
  u0,
  ode_cache,
  vθ)

  residual!(b,op,μ,tθ,(u0,vθ),ode_cache)
  b.array .*= -1.0
  b
end

# function residual_step!(
#   rf::PTArray,
#   solver::ThetaMethod,
#   op::PODEOperator,
#   μ::AbstractVector,
#   u0::PTArray,
#   t0::Real,
#   cache)

#   dt = solver.dt
#   solver.θ == 0.0 ? dtθ = dt : dtθ = dt*solver.θ
#   tθ = t0+dtθ

#   if isnothing(cache)
#     ode_cache = allocate_cache(op,μ)
#     vθ = similar(u0)
#     vθ .= 0.0
#     b = allocate_residual(op,u0,ode_cache)
#   else
#     ode_cache,vθ,b = cache
#   end

#   ode_cache = update_cache!(ode_cache,op,μ,tθ)

#   _vector!(b,op,μ,tθ,dtθ,u0,ode_cache,vθ)

#   @. rf = b

#   cache = (ode_cache,vθ,b)
#   tf = t0+dt
#   return (rf,tf,cache)
# end

# function jacobian_step!(
#   af::PTArray,
#   solver::ThetaMethod,
#   op::PODEOperator,
#   μ::AbstractVector,
#   A0::PTArray,
#   t0::Real,
#   cache)

#   dt = solver.dt
#   solver.θ == 0.0 ? dtθ = dt : dtθ = dt*solver.θ
#   tθ = t0+dtθ

#   if isnothing(cache)
#     ode_cache = allocate_cache(op,μ)
#     vθ = similar(r0)
#     vθ .= 0.0
#     A = allocate_jacobian(op,A0,ode_cache)
#   else
#     ode_cache,vθ,A = cache
#   end

#   ode_cache = update_cache!(ode_cache,op,μ,tθ)

#   _matrix!(A,op,μ,tθ,dtθ,A0,ode_cache,vθ)
#   compress!(af,A)

#   cache = (ode_cache,vθ,A)
#   tf = t0+dt
#   return (af,tf,cache)
# end
