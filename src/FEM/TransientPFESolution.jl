struct TransientPFESolution
  sol::PODESolution
  trial::TransientTrialPFESpace
end

function TransientPFESolution(
  solver::ODESolver,
  op::TransientFEOperator,
  uh0;
  kwargs...)

  r = realization(op.ptspace;kwargs...)
  TransientPFESolution(solver,op,uh0,r)
end

function TransientPFESolution(
  solver::ODESolver,
  op::TransientFEOperator,
  uh0,
  r::Realization)

  @unpack params,times = r
  ode_op = get_algebraic_operator(op)
  u0 = get_free_dof_values(uh0)(params)
  ode_sol = solve(solver,ode_op,u0,params,times)
  trial = get_trial(op)

  TransientPFESolution(ode_sol,trial)
end

function Base.iterate(sol::TransientPFESolution)
  odesolnext = iterate(sol.odesol)
  if odesolnext === nothing
    return nothing
  end
  (uf,tf),odesolstate = odesolnext

  Uh = allocate_trial_space(sol.trial)
  Uh = evaluate!(Uh,sol.trial,tf)
  uh = FEFunction(Uh,uf)

  state = Uh,odesolstate
  (uh,tf),state
end

function Base.iterate(sol::TransientPFESolution,state)
  Uh,odesolstate = state
  odesolnext = iterate(sol.odesol,odesolstate)
  if odesolnext === nothing
    return nothing
  end
  (uf,tf),odesolstate = odesolnext

  Uh = evaluate!(Uh,sol.trial,tf)
  uh = FEFunction(Uh,uf)

  state = Uh,odesolstate
  (uh,tf),state
end

function Algebra.solve(
  solver::ODESolver,
  op::TransientFEOperator,
  uh0;
  kwargs...)
  TransientFESolution(solver,op,uh0)
end
