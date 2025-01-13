"""
    struct TransientParamFESolution{V} <: TransientFESolution
      odesol::ODEParamSolution{V}
      trial
    end

Wrapper around a `TransientParamFEOperator` and `ODESolver` that represents the
parametric solution at a set of time steps. It is an iterator that computes the solution
at each time step in a lazy fashion when accessing the solution.

"""
struct TransientParamFESolution{V} <: TransientFESolution
  odesol::ODEParamSolution{V}
  trial
end

function TransientParamFESolution(
  solver::ODESolver,
  op::TransientParamFEOperator,
  r::TransientRealization,
  u0::Tuple{Vararg{AbstractVector}})

  odeop = get_algebraic_operator(op)
  odesol = solve(solver,odeop,r,u0)
  trial = get_trial(op)
  TransientParamFESolution(odesol,trial)
end

function TransientParamFESolution(
  solver::ODESolver,
  op::TransientParamFEOperator,
  r::TransientRealization,
  uh0::Tuple{Vararg{Function}})

  params = get_params(r)
  u0 = get_free_dof_values.(map(uh0->uh0(params),uh0))
  TransientParamFESolution(solver,op,r,u0)
end

function TransientParamFESolution(
  solver::ODESolver,
  op::TransientParamFEOperator,
  r::TransientRealization,
  uh0::Union{Function,AbstractVector})

  TransientParamFESolution(solver,op,r,(uh0,))
end

function Base.iterate(sol::TransientParamFESolution)
  ode_it = iterate(sol.odesol)
  if isnothing(ode_it)
    return nothing
  end
  (rf,uf),ode_it_state = ode_it

  Uh = allocate_space(sol.trial,rf)
  Uh = evaluate!(Uh,sol.trial,rf)
  uhf = FEFunction(Uh,uf)

  state = Uh,ode_it_state
  (rf,uhf),state
end

function Base.iterate(sol::TransientParamFESolution,state)
  Uh,ode_it_state = state
  ode_it = iterate(sol.odesol,ode_it_state)
  if isnothing(ode_it)
    return nothing
  end
  (rf,uf),ode_it_state = ode_it

  Uh = evaluate!(Uh,sol.trial,rf)
  uhf = FEFunction(Uh,uf)

  state = Uh,ode_it_state
  (rf,uhf),state
end

function Base.collect(sol::TransientParamFESolution{V}) where V
  odesol = sol.odesol
  ntimes = num_times(odesol.r)

  free_values = Vector{V}(undef,ntimes)
  for (k,(rt,uht)) in enumerate(sol)
    ut = get_free_dof_values(uht)
    free_values[k] = copy(ut)
  end

  return free_values,odesol.tracker
end

function collect_initial_values(sol::TransientParamFESolution)
  collect_initial_values(sol.odesol)
end

function Algebra.solve(
  solver::ODESolver,
  op::TransientParamFEOperator,
  r::TransientRealization,
  uh0)

  TransientParamFESolution(solver,op,r,uh0)
end

function Algebra.solve(
  solver::ODESolver,
  op::SplitTransientParamFEOperator,
  r::TransientRealization,
  uh0)

  solve(solver,set_domains(op),r,uh0)
end

function Algebra.solve(
  solver::ODESolver,
  op::LinearNonlinearTransientParamFEOperator,
  r::TransientRealization,
  uh0)

  TransientParamFESolution(solver,join_operators(op),r,uh0)
end
