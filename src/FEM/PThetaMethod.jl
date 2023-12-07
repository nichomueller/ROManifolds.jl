function TransientFETools.solve_step!(
  uf::AbstractArray,
  solver::PThetaMethod,
  op::PODEOperator,
  μ::AbstractVector,
  u0::AbstractArray,
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

  nlop = PTThetaMethodOperator(op,μ,tθ,dtθ,u0,ode_cache,vθ)

  nl_cache = solve!(uf,solver.nls,nlop,nl_cache)

  if 0.0 < solver.θ < 1.0
    @. uf = uf*(1.0/solver.θ)-u0*((1-solver.θ)/solver.θ)
  end

  cache = (ode_cache,vθ,nl_cache)
  tf = t0+dt
  return (uf,tf,cache)
end

struct PTThetaMethodOperator <: PTOperator{Nonlinear}
  odeop::PODEOperator
  μ
  tθ
  dtθ::Float
  u0::AbstractVector
  ode_cache
  vθ::AbstractVector
end

function get_ptoperator(
  odeop::PODEOperator,
  μ,
  tθ,
  dtθ::Float,
  u0::AbstractVector,
  ode_cache,
  vθ::AbstractVector)

  PTThetaMethodOperator(odeop,μ,tθ,dtθ,u0,ode_cache,vθ)
end

function Algebra.residual!(
  b::AbstractVector,
  op::PTThetaMethodOperator,
  x::AbstractVector)

  uF = x
  vθ = op.vθ
  @. vθ = (x-op.u0)/op.dtθ
  z = zero(eltype(b))
  fill!(b,z)
  residual!(b,op.odeop,op.μ,op.tθ,(uF,vθ),op.ode_cache)
end

function residual_for_trian!(
  b::AbstractVector,
  op::PTThetaMethodOperator,
  x::AbstractVector,
  args...)

  uF = x
  vθ = op.vθ
  z = zero(eltype(b))
  fill!(b,z)
  residual_for_trian!(b,op.odeop,op.μ,op.tθ,(uF,vθ),op.ode_cache,args...)
end

function ODETools.jacobian!(
  A::AbstractMatrix,
  op::PTThetaMethodOperator,
  x::AbstractVector)

  uF = x
  vθ = op.vθ
  @. vθ = (x-op.u0)/op.dtθ
  z = zero(eltype(A))
  fillstored!(A,z)
  jacobians!(A,op.odeop,op.μ,op.tθ,(uF,vθ),(1.0,1/op.dtθ),op.ode_cache)
end

function Algebra.jacobian!(
  A::AbstractMatrix,
  op::PTThetaMethodOperator,
  x::AbstractVector,
  i::Int)

  uF = x
  vθ = op.vθ
  @. vθ = (x-op.u0)/op.dtθ
  z = zero(eltype(A))
  fillstored!(A,z)
  γ = (1.0,1/op.dtθ)
  jacobian!(A,op.odeop,op.μ,op.tθ,(uF,vθ),i,γ[i],op.ode_cache)
end

function jacobian_for_trian!(
  A::AbstractMatrix,
  op::PTThetaMethodOperator,
  x::AbstractVector,
  i::Int,
  args...)

  uF = x
  vθ = op.vθ
  z = zero(eltype(A))
  fillstored!(A,z)
  γ = (1.0,1/op.dtθ)
  jacobian_for_trian!(A,op.odeop,op.μ,op.tθ,(uF,vθ),i,γ[i],op.ode_cache,args...)
end
