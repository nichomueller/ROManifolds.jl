abstract type ODEPSolution <: ODESolution end

struct GenericODEPSolution <: ODEPSolution
  solver::ODESolver
  op::ODEPOperator
  u0::AbstractVector
  r::TransientPRealization
end

function Base.iterate(sol::ODEPSolution)
  uf = copy(sol.u0)
  u0 = copy(sol.u0)
  r0 = get_at_time(sol.r,:initial)
  cache = nothing

  uf,rf,cache = solve_step!(uf,sol.solver,sol.op,r0,u0,cache)

  u0 .= uf
  state = (uf,u0,rf,cache)

  return (uf,rf),state
end

function Base.iterate(sol::ODEPSolution,state)
  uf,u0,r0,cache = state

  if get_times(r0) >= 100*eps()*get_final_time(sol.r)
    return nothing
  end

  uf,rf,cache = solve_step!(uf,sol.solver,sol.op,r0,u0,cache)

  u0 .= uf
  state = (uf,u0,rf,cache)

  return (uf,rf),state
end

function Algebra.solve(
  solver::ODESolver,
  op::ODEPOperator,
  u0::T,
  r::TransientPRealization) where T

  GenericODEPSolution(solver,op,u0,r)
end

function ODETools.test_ode_solution(sol::ODEPSolution)
  for (u_n,r_n) in sol
    @test isa(r_n,TransientPRealization)
    @test isa(u_n,ParamVector)
  end
  true
end
