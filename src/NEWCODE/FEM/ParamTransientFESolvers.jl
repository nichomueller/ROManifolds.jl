abstract type ParamODESolution <: GridapType end

struct GenericParamODESolution <: ParamODESolution
  op::ParamTransientFEOperator
  solver::ODESolver
  μ::AbstractVector
  u0::AbstractVector
  t0::Real
  tF::Real
end

function solve_step!(
  uF::Union{AbstractVector,Tuple{Vararg{AbstractVector}}},
  op::ParamTransientFEOperator,
  solver::ODESolver,
  μ::AbstractVector,
  u0::Union{AbstractVector,Tuple{Vararg{AbstractVector}}},
  t0::Real) # -> (uF,tF,cache)

  solve_step!(uF,op,solver,μ,u0,t0,nothing)
end

function Gridap.Algebra.solve(
  op::ParamTransientFEOperator,
  solver::ODESolver,
  μ::AbstractVector,
  uh0)

  u0 = get_free_dof_values(uh0)
  t0,tF = solver.t0,solver.tF
  GenericParamODESolution(op,solver,μ,u0,t0,tF)
end

function Base.iterate(
  sol::GenericParamODESolution)

  uF = copy(sol.u0)
  u0 = copy(sol.u0)
  t0 = sol.t0

  # Solve step
  uF,tF,cache = solve_step!(uF,sol.op,sol.solver,sol.μ,u0,t0)

  # Update
  u0 .= uF
  state = (uF,u0,tF,cache)

  # Multi field
  ode_cache, = cache
  Uh, = first(ode_cache)
  uh = _split_solutions(Uh,uF)

  return uh,state
end

function Base.iterate(
  sol::GenericParamODESolution,
  state)

  uF,u0,t0,cache = state

  if t0 >= sol.tF - 100*eps()
    return nothing
  end

  # Solve step
  uF,tF,cache = solve_step!(uF,sol.op,sol.solver,sol.μ,u0,t0,cache)

  # Update
  u0 .= uF
  state = (uF,u0,tF,cache)

  # Multi field
  ode_cache, = cache
  Uh, = first(ode_cache)
  uh = _split_solutions(Uh,uF)

  return uh,state
end
