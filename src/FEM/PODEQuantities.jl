abstract type PODEQuantity end
abstract type PODESolution <: PODEQuantity end

struct SingleFieldPODESolution <: PODESolution
  solver::ODESolver
  op::PODEOperator
  μ::AbstractVector
  u0::PTArray
  t0::Real
  tF::Real
end

function PODESolution(
  solver::ODESolver,
  op::PODEOperator,
  μ::AbstractVector,
  u0::PTArray,
  t0::Real,
  tF::Real)
  SingleFieldPODESolution(solver,op,μ,u0,t0,tF)
end

struct MultiFieldPODESolution <: PODESolution
  solver::ODESolver
  op::PODEOperator
  μ::AbstractVector
  u0::Vector{<:PTArray}
  t0::Real
  tF::Real
end

function PODESolution(
  solver::ODESolver,
  op::PODEOperator,
  μ::AbstractVector,
  u0::Vector{<:PTArray},
  t0::Real,
  tF::Real)
  MultiFieldPODESolution(solver,op,μ,u0,t0,tF)
end

function Base.iterate(sol::PODESolution)
  uf = copy(sol.u0)
  u0 = copy(sol.u0)
  t0 = sol.t0
  n = 0
  cache = nothing

  uf,tf,cache = solution_step!(uf,sol.solver,sol.op,sol.μ,u0,t0,cache)

  u0 .= uf
  n += 1
  state = (uf,u0,tf,n,cache)

  return (uf,tf,n),state
end

function Base.iterate(sol::PODESolution,state)
  uf,u0,t0,n,cache = state

  if t0 >= sol.tF - ϵ
    return nothing
  end

  uf,tf,cache = solution_step!(uf,sol.solver,sol.op,sol.μ,u0,t0,cache)

  u0 .= uf
  n += 1
  state = (uf,u0,tf,n,cache)

  return (uf,tf,n),state
end

# struct PODEResidual <: PODEQuantity
#   solver::ODESolver
#   op::PODEOperator
#   μ::AbstractVector
#   u0::PTArray
#   t0::Real
#   tF::Real
# end

# function residual_step!(
#   uf::PTArray,
#   solver::ThetaMethod,
#   op::PODEOperator,
#   μ::AbstractVector,
#   u0::PTArray,
#   t0::Real)

#   residual_step!(uf,solver,op,μ,u0,t0,nothing)
# end

# function Base.iterate(sol::PODEResidual)
#   bf = copy(sol.b0)
#   b0 = copy(sol.b0)
#   t0 = sol.t0

#   bf,tf,cache = residual_step!(bf,sol.solver,sol.op,sol.μ,b0,t0)

#   b0 .= bf
#   state = (bf,b0,tf,cache)

#   return (bf,tf),state
# end

# function Base.iterate(sol::PODEResidual,state)
#   bf,b0,t0,cache = state

#   if t0 >= sol.tF - ϵ
#     return nothing
#   end

#   bf,tf,cache = residual_step!(bf,sol.solver,sol.op,sol.μ,b0,t0,cache)

#   b0 .= bf
#   state = (bf,b0,tf,cache)

#   return (bf,tf),state
# end

# struct PODEJacobian <: PODEQuantity
#   solver::ODESolver
#   op::PODEOperator
#   μ::AbstractVector
#   u0::PTArray
#   t0::Real
#   tF::Real
#   i::Int
# end

# function jacobian_step!(
#   uf::PTArray,
#   solver::ThetaMethod,
#   op::PODEOperator,
#   μ::AbstractVector,
#   u0::PTArray,
#   t0::Real)

#   jacobian_step!(uf,solver,op,μ,u0,t0,nothing)
# end

# function Base.iterate(sol::PODEJacobian)
#   Af = copy(sol.A0)
#   A0 = copy(sol.A0)
#   t0 = sol.t0

#   Af,tf,cache = jacobian_step!(Af,sol.solver,sol.op,sol.μ,A0,t0)

#   A0 .= Af
#   state = (Af,A0,tf,cache)

#   return (Af,tf),state
# end

# function Base.iterate(sol::PODEJacobian,state)
#   Af,A0,t0,cache = state

#   if t0 >= sol.tF - ϵ
#     return nothing
#   end

#   Af,tf,cache = jacobian_step!(Af,sol.solver,sol.op,sol.μ,A0,t0,cache)

#   A0 .= Af
#   state = (Af,A0,tf,cache)

#   return (Af,tf),state
# end
