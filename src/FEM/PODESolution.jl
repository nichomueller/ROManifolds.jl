abstract type PODESolution <: ODESolution end

struct GenericPODESolution
  solver::ODESolver
  op::TransientPFEOperator
  u0::AbstractVector
  r::Realization
end

# Base.length(sol::PODESolution) = Int((sol.tf-sol.t0)/sol.solver.dt)

function Base.iterate(sol::PODESolution)
  uf = copy(sol.u0)
  u0 = copy(sol.u0)
  r0 = get_at_time(sol.r,:initial)
  cache = nothing

  uf,rf,cache = solve_step!(uf,sol.solver,sol.op,r0,u0,cache)

  u0 .= uf
  state = (uf,u0,rf,cache)

  return (uf,tf),state
end

function Base.iterate(sol::PODESolution,state)
  uf,u0,r0,cache = state

  if get_times(r0) >= 100*eps()*get_final_time(sol.r)
    return nothing
  end

  uf,rf,cache = solve_step!(uf,sol.solver,sol.op,r0,u0,cache)

  u0 .= uf
  state = (uf,u0,rf,cache)

  return (uf,rf),state
end

function solve(
  solver::ODESolver,
  op::PODEOperator,
  u0::T,
  r::Realization)

  GenericODESolution(solver,op,u0,r)
end
