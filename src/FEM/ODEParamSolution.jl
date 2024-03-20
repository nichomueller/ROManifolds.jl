function ode_start(
  solver::ODESolver,
  odeop::ODEOperator,
  r0::TransientParamRealization,
  us0::Tuple{Vararg{AbstractVector}},
  odecache)

  state0 = copy.(us0)
  (state0,odecache)
end

function ODEs.ode_finish!(
  uF::AbstractVector,
  solver::ODESolver,
  op::ODEOperator,
  r0::TransientParamRealization,
  rf::TransientParamRealization,
  statef::Tuple{Vararg{AbstractVector}},
  odecache)

  copy!(uF,first(statef))
  (uF,odecache)
end

abstract type ODEParamSolution <: ODESolution end

struct GenericODEParamSolution <: ODEParamSolution
  solver::ODESolver
  op::ODEParamOperator
  r::TransientParamRealization
  us0::AbstractVector
end

function Base.iterate(sol::ODEParamSolution)
  r0 = get_at_time(sol.r,:initial)
  cache = allocate_odecache(sol.solver,sol.op,r0,sol.us0)

  state0,cache = ode_start(sol.solver,sol.op,r0,sol.us0,cache)

  statef = copy.(state0)
  rf,statef,cache = ode_march!(statef,sol.solver,sol.op,r0,state0,cache)

  uf = copy(first(sol.us0))
  uf,cache = ode_finish!(uf,sol.solver,sol.op,r0,rf,statef,cache)

  state = (rf,statef,state0,cache)
  return (rf,uf),state
end

function Base.iterate(sol::ODEParamSolution,state)
  r0,state0,statef,cache = state

  if get_times(r0) >= get_final_time(sol.r) - ε
    return nothing
  end

  rf,statef,cache = ode_march!(statef,sol.solver,sol.op,r0,state0,cache)

  uf,cache = ode_finish!(uf,sol.solver,sol.op,r0,rf,statef,cache)

  state = (rf,statef,state0,cache)
  return (rf,uf),state
end

function Base.collect(sol::ODEParamSolution)
  ntimes = num_times(sol.r)

  initial_values = sol.u0
  V = typeof(initial_values)
  free_values = Vector{V}(undef,ntimes)
  for (k,(rt,ut)) in enumerate(sol)
    free_values[k] = copy(ut)
  end
  return free_values
end

function Algebra.solve(
  solver::ODESolver,
  op::ODEParamOperator,
  r::TransientParamRealization,
  u0::T) where T

  GenericODEParamSolution(solver,op,r,u0)
end

# for testing purposes

function ODEs.test_ode_solution(sol::ODEParamSolution)
  for (r_n,u_n) in sol
    @test isa(r_n,TransientParamRealization)
    @test isa(u_n,ParamVector)
  end
  true
end
