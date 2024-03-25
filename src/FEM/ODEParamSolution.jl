function ODEs.ode_start(
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
  odeop::ODEOperator,
  r0::TransientParamRealization,
  rf::TransientParamRealization,
  statef::Tuple{Vararg{AbstractVector}},
  odecache)

  copy!(uF,first(statef))
  (uF,odecache)
end

abstract type ODEParamSolution{V} <: ODESolution end

struct GenericODEParamSolution{V} <: ODEParamSolution{V}
  solver::ODESolver
  odeop::ODEParamOperator
  r::TransientParamRealization
  us0::Tuple{Vararg{V}}
end

function Base.iterate(sol::ODEParamSolution)
  r0 = get_at_time(sol.r,:initial)
  cache = allocate_odecache(sol.solver,sol.odeop,r0,sol.us0)

  state0,cache = ode_start(sol.solver,sol.odeop,r0,sol.us0,cache)

  statef = copy.(state0)
  rf,statef,cache = ode_march!(statef,sol.solver,sol.odeop,r0,state0,cache)

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

  rf,statef,cache = ode_march!(statef,sol.solver,sol.odeop,r0,state0,cache)

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
  return free_values
end

function Algebra.solve(
  solver::ODESolver,
  odeop::ODEParamOperator,
  r::TransientParamRealization,
  u0::T) where T

  GenericODEParamSolution(solver,odeop,r,u0)
end

# for testing purposes

function ODEs.test_ode_solution(sol::ODEParamSolution)
  for (r_n,u_n) in sol
    @test isa(r_n,TransientParamRealization)
    @test isa(u_n,ParamVector)
  end
  true
end
