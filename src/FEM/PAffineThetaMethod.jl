function TransientFETools.solve_step!(
  uf::AbstractVector,
  solver::PThetaMethod,
  op::AffinePODEOperator,
  μ::AbstractVector,
  u0::AbstractVector,
  t0::Real,
  cache)

  dt = solver.dt
  solver.θ == 0.0 ? dtθ = dt : dtθ = dt*solver.θ
  tθ = t0+dtθ

  if isnothing(cache)
    ode_cache = allocate_cache(op,μ,tθ)
    vθ = similar(u0)
    vθ .= 0.0
    l_cache = nothing
    A,b = _allocate_matrix_and_vector(op,μ,t0,u0,ode_cache)
  else
    ode_cache,vθ,A,b,l_cache = cache
  end

  ode_cache = update_cache!(ode_cache,op,μ,tθ)

  _matrix_and_vector!(A,b,op,μ,tθ,dtθ,u0,ode_cache,vθ)
  afop = PTAffineOperator(A,b)

  l_cache = solve!(uf,solver.nls,afop,l_cache)

  uf .+= u0
  if 0.0 < solver.θ < 1.0
    @. uf = uf*(1.0/solver.θ)-u0*((1-solver.θ)/solver.θ)
  end

  cache = (ode_cache,vθ,A,b,l_cache)
  tf = t0+dt
  return (uf,tf,cache)
end

struct PTThetaAffineMethodOperator <: PTOperator{Affine}
  odeop::AffinePODEOperator
  μ
  tθ
  dtθ::Float
  u0::AbstractVector
  ode_cache
  vθ::AbstractVector
end

function get_ptoperator(
  odeop::AffinePODEOperator,
  μ,
  tθ,
  dtθ::Float,
  u0::AbstractVector,
  ode_cache,
  vθ::AbstractVector)

  PTThetaAffineMethodOperator(odeop,μ,tθ,dtθ,u0,ode_cache,vθ)
end

function residual_for_trian!(
  b::AbstractVector,
  op::PTThetaAffineMethodOperator,
  ::AbstractVector,
  args...)

  vθ = op.vθ
  z = zero(eltype(b))
  fill!(b,z)
  residual_for_trian!(b,op.odeop,op.μ,op.tθ,(vθ,vθ),op.ode_cache,args...)
end

function jacobian_for_trian!(
  A::AbstractMatrix,
  op::PTThetaAffineMethodOperator,
  ::AbstractVector,
  i::Int,
  args...)

  vθ = op.vθ
  z = zero(eltype(A))
  fillstored!(A,z)
  jacobian_for_trian!(A,op.odeop,op.μ,op.tθ,(vθ,vθ),i,(1.0,1/op.dtθ)[i],op.ode_cache,args...)
end

# SHORTCUTS
function _allocate_matrix_and_vector(odeop,μ,t0,u0,ode_cache)
  b = allocate_residual(odeop,μ,t0,u0,ode_cache)
  A = allocate_jacobian(odeop,μ,t0,u0,1,ode_cache)
  return A,b
end

function _matrix_and_vector!(
  A::AbstractMatrix,
  b::AbstractVector,
  op::AffinePODEOperator,
  μ,
  tθ,
  dtθ,
  u0,
  ode_cache,
  vθ)

  _matrix!(A,op,μ,tθ,dtθ,u0,ode_cache,vθ)
  _vector!(b,op,μ,tθ,dtθ,u0,ode_cache,vθ)
end

function _matrix!(
  A::AbstractMatrix,
  op::AffinePODEOperator,
  μ,
  tθ,
  dtθ,
  u0,
  ode_cache,
  vθ)

  z = zero(eltype(A))
  fillstored!(A,z)
  jacobians!(A,op,μ,tθ,(vθ,vθ),(1.0,1/dtθ),ode_cache)
end

function _vector!(
  b::AbstractVector,
  op::AffinePODEOperator,
  μ,
  tθ,
  dtθ,
  u0,
  ode_cache,
  vθ)

  z = zero(eltype(b))
  fill!(b,z)
  residual!(b,op,μ,tθ,(u0,vθ),ode_cache)
  b .*= -1.0
  b
end
