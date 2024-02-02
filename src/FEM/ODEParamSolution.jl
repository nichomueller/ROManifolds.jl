abstract type ODEParamSolution <: ODESolution end

struct GenericODEParamSolution <: ODEParamSolution
  solver::ODESolver
  op::ODEParamOperator
  u0::AbstractVector
  r::TransientParamRealization
end

function Base.iterate(sol::ODEParamSolution)
  uf = copy(sol.u0)
  u0 = copy(sol.u0)
  r0 = get_at_time(sol.r,:initial)
  cache = nothing

  uf,rf,cache = solve_step!(uf,sol.solver,sol.op,r0,u0,cache)

  u0 .= uf
  state = (uf,u0,rf,cache)

  return (uf,rf),state
end

function Base.iterate(sol::ODEParamSolution,state)
  uf,u0,r0,cache = state

  if get_times(r0) >= get_final_time(sol.r) - 100*eps()
    return nothing
  end

  uf,rf,cache = solve_step!(uf,sol.solver,sol.op,r0,u0,cache)

  u0 .= uf
  state = (uf,u0,rf,cache)

  return (uf,rf),state
end

function Base.collect(sol::ODEParamSolution)
  ntimes = num_times(sol.r)

  initial_values = sol.u0
  V = typeof(initial_values)
  free_values = Vector{V}(undef,ntimes)
  for (k,(ut,rt)) in enumerate(sol)
    free_values[k] = copy(ut)
  end
  return free_values,initial_values
end

function Algebra.solve(
  solver::ODESolver,
  op::ODEParamOperator,
  u0::T,
  r::TransientParamRealization) where T

  GenericODEParamSolution(solver,op,u0,r)
end

function ODETools.test_ode_solution(sol::ODEParamSolution)
  for (u_n,r_n) in sol
    @test isa(r_n,TransientParamRealization)
    @test isa(u_n,ParamVector)
  end
  true
end
