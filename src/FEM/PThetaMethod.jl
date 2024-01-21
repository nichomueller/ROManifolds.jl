# general nonlinear case
function ODETools.solve_step!(
  uf::AbstractVector,
  solver::ThetaMethod,
  op::ODEPOperator,
  r::TransientPRealization,
  u0::AbstractVector,
  cache)

  dt = solver.dt
  θ = solver.θ
  θ == 0.0 ? dtθ = dt : dtθ = dt*θ
  change_time!(r,dtθ)

  if isnothing(cache)
    ode_cache = allocate_cache(op,r)
    vθ = similar(u0)
    vθ .= 0.0
    nl_cache = nothing
  else
    ode_cache,vθ,nl_cache = cache
  end

  ode_cache = update_cache!(ode_cache,op,r)

  nlop = PThetaMethodOperator(op,r,dtθ,u0,ode_cache,vθ)

  nl_cache = solve!(uf,solver.nls,nlop,nl_cache)

  if 0.0 < θ < 1.0
    @. uf = uf*(1.0/θ)-u0*((1-θ)/θ)
  end

  cache = (ode_cache,vθ,nl_cache)
  change_time!(r,dt*(1-θ))
  return (uf,r,cache)
end

struct PThetaMethodOperator <: NonlinearOperator
  op::ODEPOperator
  r::TransientPRealization
  dtθ::Float
  u0::AbstractVector
  ode_cache
  vθ::AbstractVector
end

function get_method_operator(
  op::ODEPOperator,
  r::TransientPRealization,
  dtθ::Float,
  u0::AbstractVector,
  ode_cache,
  vθ::AbstractVector)

  PThetaMethodOperator(op,r,dtθ,u0,ode_cache,vθ)
end

function Algebra.residual!(
  b::AbstractVector,
  op::PThetaMethodOperator,
  x::AbstractVector)

  uF = x
  vθ = op.vθ
  @. vθ = (x-op.u0)/op.dtθ
  z = zero(eltype(b))
  fill!(b,z)
  residual!(b,op.odeop,op.r,(uF,vθ),op.ode_cache)
end

function residual_for_trian!(
  b::AbstractVector,
  op::PThetaMethodOperator,
  x::AbstractVector,
  args...)

  uF = x
  vθ = op.vθ
  z = zero(eltype(b))
  fill!(b,z)
  residual_for_trian!(b,op.odeop,op.r,(uF,vθ),op.ode_cache,args...)
end

function Algebra.jacobian!(
  A::AbstractMatrix,
  op::PThetaMethodOperator,
  x::AbstractVector)

  uF = x
  vθ = op.vθ
  @. vθ = (x-op.u0)/op.dtθ
  z = zero(eltype(A))
  fillstored!(A,z)
  jacobians!(A,op.odeop,op.r,(uF,vθ),(1.0,1/op.dtθ),op.ode_cache)
end

function Algebra.jacobian!(
  A::AbstractMatrix,
  op::PThetaMethodOperator,
  x::AbstractVector,
  i::Int)

  uF = x
  vθ = op.vθ
  @. vθ = (x-op.u0)/op.dtθ
  z = zero(eltype(A))
  fillstored!(A,z)
  γ = (1.0,1/op.dtθ)
  jacobian!(A,op.odeop,op.r,(uF,vθ),i,γ[i],op.ode_cache)
end

function jacobian_for_trian!(
  A::AbstractMatrix,
  op::PThetaMethodOperator,
  x::AbstractVector,
  i::Int,
  args...)

  uF = x
  vθ = op.vθ
  z = zero(eltype(A))
  fillstored!(A,z)
  γ = (1.0,1/op.dtθ)
  jacobian_for_trian!(A,op.odeop,op.r,(uF,vθ),i,γ[i],op.ode_cache,args...)
end

# specializations for affine case
function ODETools.solve_step!(
  uf::AbstractVector,
  solver::ThetaMethod,
  op::AffineODEPOperator,
  r::TransientPRealization,
  u0::AbstractVector,
  cache)

  dt = solver.dt
  θ = solver.θ
  θ == 0.0 ? dtθ = dt : dtθ = dt*θ
  change_time!(r,dtθ)

  if isnothing(cache)
    ode_cache = allocate_cache(op,r)
    vθ = similar(u0)
    vθ .= 0.0
    l_cache = nothing
    A,b = ODETools._allocate_matrix_and_vector(op,r,u0,ode_cache)
  else
    ode_cache,vθ,A,b,l_cache = cache
  end

  ode_cache = update_cache!(ode_cache,op,r)

  ODETools._matrix_and_vector!(A,b,op,r,dtθ,u0,ode_cache,vθ)
  afop = AffineOperator(A,b)

  newmatrix = true
  l_cache = solve!(uf,solver.nls,afop,l_cache,newmatrix)

  uf = uf + u0
  if 0.0 < θ < 1.0
    @. uf = uf*(1.0/θ)-u0*((1-θ)/θ)
  end

  cache = (ode_cache,vθ,A,b,l_cache)
  change_time!(r,dt*(1-θ))
  return (uf,r,cache)
end

struct AffinePThetaMethodOperator <: NonlinearOperator
  op::AffineODEPOperator
  r::TransientPRealization
  dtθ::Float
  u0::AbstractVector
  ode_cache
  vθ::AbstractVector
end

function get_method_operator(
  op::AffineODEPOperator,
  r::TransientPRealization,
  dtθ::Float,
  u0::AbstractVector,
  ode_cache,
  vθ::AbstractVector)

  AffinePThetaMethodOperator(op,r,dtθ,u0,ode_cache,vθ)
end

function Algebra.residual!(
  b::AbstractVector,
  op::AffinePThetaMethodOperator,
  x::AbstractVector)

  uF = op.u0
  vθ = op.vθ
  z = zero(eltype(b))
  fill!(b,z)
  residual!(b,op.odeop,op.r,(uF,vθ),op.ode_cache)
end

function residual_for_trian!(
  b::AbstractVector,
  op::AffinePThetaMethodOperator,
  x::AbstractVector,
  args...)

  vθ = op.vθ
  z = zero(eltype(b))
  fill!(b,z)
  residual_for_trian!(b,op.odeop,op.r,(vθ,vθ),op.ode_cache,args...)
end

function Algebra.jacobian!(
  A::AbstractMatrix,
  op::AffinePThetaMethodOperator,
  x::AbstractVector)

  vθ = op.vθ
  z = zero(eltype(A))
  fillstored!(A,z)
  jacobians!(A,op.odeop,op.r,(vθ,vθ),(1.0,1/op.dtθ),op.ode_cache)
end

function Algebra.jacobian!(
  A::AbstractMatrix,
  op::AffinePThetaMethodOperator,
  x::AbstractVector,
  i::Int)

  vθ = op.vθ
  z = zero(eltype(A))
  fillstored!(A,z)
  γ = (1.0,1/op.dtθ)
  jacobian!(A,op.odeop,op.r,(vθ,vθ),i,γ[i],op.ode_cache)
end

function jacobian_for_trian!(
  A::AbstractMatrix,
  op::AffinePThetaMethodOperator,
  x::AbstractVector,
  i::Int,
  args...)

  vθ = op.vθ
  z = zero(eltype(A))
  fillstored!(A,z)
  γ = (1.0,1/op.dtθ)
  jacobian_for_trian!(A,op.odeop,op.r,(vθ,vθ),i,γ[i],op.ode_cache,args...)
end
