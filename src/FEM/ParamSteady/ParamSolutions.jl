function Algebra.solve!(u,solver::NonlinearFESolver,feop::ParamFEOperator,r::Realization)
  x = get_free_dof_values(u)
  op = get_algebraic_operator(feop,r)
  t = @timed solve!(x,solver.nls,op)
  stats = CostTracker(t,name="FEM")
  trial = get_trial(feop)(r)
  uh = FEFunction(trial,x)
  uh,stats
end

function Algebra.solve!(u,solver::LinearFESolver,feop::ParamFEOperator,r::Realization)
  x = get_free_dof_values(u)
  op = get_algebraic_operator(feop,r)
  t = @timed solve!(x,solver.ls,op,r)
  stats = CostTracker(t,name="FEM")
  trial = get_trial(feop)(r)
  uh = FEFunction(trial,x)
  uh,stats
end

function Algebra.solve(solver::FESolver,op::FEOperator,r::Realization)
  U = get_trial(op)(r)
  uh = zero(U)
  vh,stats = solve!(uh,solver,op,r)
  vh,stats
end

function Algebra.solve(solver::FESolver,op::ParamFEOperatorWithTrian,r::Realization)
  solve(solver,op.op,r)
end

# linear - nonlinear interface

function Algebra.solve(solver::FESolver,op::LinearNonlinearParamFEOperator,r::Realization)
  solve(solver,join_operators(op),r)
end
