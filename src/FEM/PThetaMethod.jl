# general nonlinear case
function TransientFETools.solve_step!(
  uf::AbstractArray,
  solver::PThetaMethod,
  op::PTFEOperator,
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

struct PTThetaMethodOperator{P,T} <: PTAlgebraicOperator{Nonlinear}
  feop::PTFEOperator
  μ::P
  t::T
  dtθ::Float
  u0::AbstractVector
  ode_cache
  vθ::AbstractVector
end

function TransientFETools.get_algebraic_operator(
  feop::PTFEOperator,
  μ,
  t,
  dtθ::Float,
  u0::AbstractVector,
  ode_cache,
  vθ::AbstractVector)

  PTThetaMethodOperator(feop,μ,t,dtθ,u0,ode_cache,vθ)
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
  residual!(b,op,(uF,vθ))
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
  residual_for_trian!(b,op,(uF,vθ),args...)
end

function Algebra.jacobian!(
  A::AbstractMatrix,
  op::PTThetaMethodOperator,
  x::AbstractVector)

  uF = x
  vθ = op.vθ
  @. vθ = (x-op.u0)/op.dtθ
  z = zero(eltype(A))
  fillstored!(A,z)
  jacobians!(A,op,(uF,vθ),(1.0,1/op.dtθ))
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
  jacobian!(A,op,(uF,vθ),i,γ[i])
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
  jacobian_for_trian!(A,op,(uF,vθ),i,γ[i],args...)
end

# specializations for affine case
function TransientFETools.solve_step!(
  uf::AbstractVector,
  solver::PThetaMethod,
  op::PTFEOperator{Affine},
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
  else
    ode_cache,vθ,l_cache = cache
  end

  ode_cache = update_cache!(ode_cache,op,μ,tθ)

  lop = PTAffineThetaMethodOperator(op,μ,tθ,dtθ,u0,ode_cache,vθ)

  l_cache = solve!(uf,solver.nls,lop,l_cache)

  uf .+= u0
  if 0.0 < solver.θ < 1.0
    @. uf = uf*(1.0/solver.θ)-u0*((1-solver.θ)/solver.θ)
  end

  cache = (ode_cache,vθ,l_cache)
  tf = t0+dt
  return (uf,tf,cache)
end

# function ODETools._allocate_matrix_and_vector(op,μ,t0,u0,ode_cache)
#   b = allocate_residual(op,μ,t0,xh)
#   A = allocate_jacobian(op,μ,t0,xh,1)
#   return A,b
# end

# function ODETools._matrix_and_vector!(
#   A::AbstractMatrix,
#   b::AbstractVector,
#   op::PTFEOperator{Affine},
#   μ,
#   t,
#   dtθ,
#   u0,
#   ode_cache,
#   vθ)

#   _matrix!(A,op,μ,t,dtθ,u0,ode_cache,vθ)
#   _vector!(b,op,μ,t,dtθ,u0,ode_cache,vθ)
# end

# function ODETools._matrix!(
#   A::AbstractMatrix,
#   op::PTFEOperator{Affine},
#   μ,
#   t,
#   dtθ,
#   u0,
#   ode_cache,
#   vθ)

#   z = zero(eltype(A))
#   fillstored!(A,z)
#   jacobians!(A,op,μ,t,(vθ,vθ),(1.0,1/dtθ),ode_cache)
# end

# function ODETools._vector!(
#   b::AbstractVector,
#   op::PTFEOperator{Affine},
#   μ,
#   t,
#   dtθ,
#   u0,
#   ode_cache,
#   vθ)

#   z = zero(eltype(b))
#   fill!(b,z)
#   residual!(b,op,μ,t,(u0,vθ),ode_cache)
#   b .*= -1.0
#   b
# end

struct PTAffineThetaMethodOperator{P,T} <: PTAlgebraicOperator{Affine}
  feop::PTFEOperator{Affine}
  μ::P
  t::T
  dtθ::Float
  u0::AbstractVector
  ode_cache
  vθ::AbstractVector
end

function TransientFETools.get_algebraic_operator(
  feop::PTFEOperator{Affine},
  μ,
  t,
  dtθ::Float,
  u0::AbstractVector,
  ode_cache,
  vθ::AbstractVector)

  PTAffineThetaMethodOperator(feop,μ,t,dtθ,u0,ode_cache,vθ)
end

function Algebra.residual!(
  b::AbstractVector,
  op::PTAffineThetaMethodOperator,
  x::AbstractVector)

  uF = x
  vθ = op.vθ
  z = zero(eltype(b))
  fill!(b,z)
  residual!(b,op,(uF,vθ))
end

function residual_for_trian!(
  b::AbstractVector,
  op::PTAffineThetaMethodOperator,
  x::AbstractVector,
  args...)

  vθ = op.vθ
  z = zero(eltype(b))
  fill!(b,z)
  residual_for_trian!(b,op,(vθ,vθ),args...)
end

function Algebra.jacobian!(
  A::AbstractMatrix,
  op::PTAffineThetaMethodOperator,
  x::AbstractVector)

  uF = x
  vθ = op.vθ
  z = zero(eltype(A))
  fillstored!(A,z)
  jacobians!(A,op,(uF,vθ),(1.0,1/op.dtθ))
end

function Algebra.jacobian!(
  A::AbstractMatrix,
  op::PTAffineThetaMethodOperator,
  x::AbstractVector,
  i::Int)

  uF = x
  vθ = op.vθ
  z = zero(eltype(A))
  fillstored!(A,z)
  γ = (1.0,1/op.dtθ)
  jacobian!(A,op,(uF,vθ),i,γ[i])
end

function jacobian_for_trian!(
  A::AbstractMatrix,
  op::PTAffineThetaMethodOperator,
  x::AbstractVector,
  i::Int,
  args...)

  vθ = op.vθ
  z = zero(eltype(A))
  fillstored!(A,z)
  γ = (1.0,1/op.dtθ)
  jacobian_for_trian!(A,op,(vθ,vθ),i,γ[i],args...)
end
