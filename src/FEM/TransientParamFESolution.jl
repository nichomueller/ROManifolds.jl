struct TransientParamFESolution
  odesol::ODEParamSolution
  trial
end

function TransientParamFESolution(
  solver::ODESolver,
  op::TransientParamFEOperator,
  uh0::Function,
  r::TransientParamRealization)

  params = get_params(r)
  ode_op = get_algebraic_operator(op)
  u0 = get_free_dof_values(uh0(params))
  ode_sol = solve(solver,ode_op,u0,r)
  trial = get_trial(op)

  TransientParamFESolution(ode_sol,trial)
end

function Base.iterate(sol::TransientParamFESolution)
  odesolnext = iterate(sol.odesol)
  if odesolnext === nothing
    return nothing
  end
  (uf,rf),odesolstate = odesolnext

  Uh = allocate_trial_space(sol.trial,rf)
  Uh = evaluate!(Uh,sol.trial,rf)
  uh = FEFunction(Uh,uf)

  state = Uh,odesolstate
  (uh,rf),state
end

function Base.iterate(sol::TransientParamFESolution,state)
  Uh,odesolstate = state
  odesolnext = iterate(sol.odesol,odesolstate)
  if odesolnext === nothing
    return nothing
  end
  (uf,rf),odesolstate = odesolnext

  Uh = evaluate!(Uh,sol.trial,rf)
  uh = FEFunction(Uh,uf)

  state = Uh,odesolstate
  (uh,rf),state
end

function Base.collect(sol::TransientParamFESolution)
  odesol = sol.odesol
  ntimes = num_times(odesol.r)

  initial_values = odesol.u0
  V = typeof(initial_values)
  free_values = Vector{V}(undef,ntimes)
  for (k,(uht,rt)) in enumerate(sol)
    ut = get_free_dof_values(uht)
    free_values[k] = copy(ut)
  end

  return free_values
end

function Algebra.solve(
  solver::ODESolver,
  op::TransientParamFEOperator,
  uh0::Function,
  r::TransientParamRealization)

  TransientParamFESolution(solver,op,uh0,r)
end

function Algebra.solve(
  solver::ODESolver,
  op::TransientParamFEOperatorWithTrian,
  uh0::Function,
  r::TransientParamRealization)

  TransientParamFESolution(solver,op.op,uh0,r)
end

function Algebra.solve(
  solver::ODESolver,
  op::LinearNonlinearTransientParamFEOperator,
  uh0::Function,
  r::TransientParamRealization)

  TransientParamFESolution(solver,join_operators(op),uh0,r)
end

function Algebra.solve(
  solver::ODESolver,
  op::TransientParamFEOperator,
  uh0::Function;
  kwargs...)

  r = realization(op;kwargs...)
  solve(solver,op,uh0,r)
end

function TransientFETools.test_transient_fe_solver(
  solver::ODESolver,
  op::TransientParamFEOperator,
  u0,
  r)

  solution = solve(solver,op,u0,r)
  for (uhn,rn) in solution
    @test isa(uhn,FEFunction)
    @test isa(rn,TransientParamRealization)
  end
  true
end
