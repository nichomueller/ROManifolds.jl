function ODEs.ode_start(
  solver::ODESolver,
  odeop::ODEParamOperator,
  r0::TransientRealization,
  us0::Tuple{Vararg{AbstractVector}},
  odeparamcache)

  state0 = copy.(us0)
  (state0,odeparamcache)
end

function ODEs.ode_finish!(
  uF::AbstractVector,
  solver::ODESolver,
  odeop::ODEParamOperator,
  r0::TransientRealization,
  rf::TransientRealization,
  statef::Tuple{Vararg{AbstractVector}},
  odeparamcache)

  copy!(uF,first(statef))
  (uF,odeparamcache)
end

"""
    struct ODEParamSolution{V} <: ODESolution end

Generic wrapper for the evolution of an `ODEParamOperator` with an `ODESolver`.

"""
struct ODEParamSolution{V} <: ODESolution
  solver::ODESolver
  odeop::ODEParamOperator
  r::TransientRealization
  us0::Tuple{Vararg{V}}
  tracker::CostTracker
end

function ODEParamSolution(
  solver::ODESolver,
  odeop::ODEParamOperator,
  r::TransientRealization,
  us0::Tuple{Vararg{V}}) where V

  tracker = CostTracker(name="FEM time marching")
  ODEParamSolution(solver,odeop,r,us0,tracker)
end

function Base.iterate(sol::ODEParamSolution)
  r0 = ParamDataStructures.get_at_time(sol.r,:initial)
  cache = allocate_odeparamcache(sol.solver,sol.odeop,r0,sol.us0)

  state0,cache = ode_start(sol.solver,sol.odeop,r0,sol.us0,cache)

  statef = copy.(state0)
  t = @timed rf,statef,cache = ode_march!(statef,sol.solver,sol.odeop,r0,state0,cache)
  update_tracker!(sol.tracker,t)

  uf = copy(first(sol.us0))
  uf,cache = ode_finish!(uf,sol.solver,sol.odeop,r0,rf,statef,cache)

  state = (rf,statef,state0,uf,cache)
  return (rf,uf),state
end

function Base.iterate(sol::ODEParamSolution,state)
  r0,state0,statef,uf,cache = state

  if get_times(r0) >= get_final_time(sol.r) - ODEs.ε
    return nothing
  end

  t = @timed rf,statef,cache = ode_march!(statef,sol.solver,sol.odeop,r0,state0,cache)
  update_tracker!(sol.tracker,t)

  uf,cache = ode_finish!(uf,sol.solver,sol.odeop,r0,rf,statef,cache)

  state = (rf,statef,state0,uf,cache)
  return (rf,uf),state
end

function Base.collect(sol::ODEParamSolution{V}) where V
  ntimes = num_times(sol.r)

  free_values = Vector{V}(undef,ntimes)
  for (k,(rt,ut)) in enumerate(sol)
    free_values[k] = copy(ut)
  end

  return free_values,sol.tracker
end

function Algebra.solve(
  solver::ODESolver,
  odeop::ODEParamOperator,
  r::TransientRealization,
  u0::T) where T

  ODEParamSolution(solver,odeop,r,u0)
end

# for testing purposes

function ODEs.test_ode_solution(sol::ODEParamSolution)
  for (r_n,u_n) in sol
    @test isa(r_n,TransientRealization)
    @test isa(u_n,ParamVector)
  end
  true
end
