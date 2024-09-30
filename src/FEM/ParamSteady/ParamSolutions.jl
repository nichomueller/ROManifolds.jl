function Algebra.solve!(
  x::AbstractParamVector,
  ls::LinearSolver,
  op::ParamOpFromFEOp,
  r::Realization)

  paramcache = allocate_paramcache(op,r,x)
  A = paramcache.A
  b = paramcache.b

  t = @timed begin
    jacobian!(A,op,r,x,paramcache)
    residual!(b,op,r,x,paramcache)
    solve!(x,ls,A,b)
  end
  stats = CostTracker(t,name="FEM")

  stats,paramcache
end

function Algebra.solve!(
  x::AbstractParamVector,
  nls::NewtonRaphsonSolver,
  op::ParamOpFromFEOp,
  r::Realization)

  paramcache = allocate_paramcache(op,r,x)
  A = paramcache.A
  b = paramcache.b

  t = @timed begin
    jacobian!(A,op,r,x,paramcache)
    residual!(b,op,r,x,paramcache)
    dx = similar(b)
    ss = symbolic_setup(nls.ls,A)
    ns = numerical_setup(ss,A)
    Algebra._solve_nr!(x,A,b,dx,ns,nls,op)
  end
  stats = CostTracker(t,name="FEM")

  stats,paramcache
end

function Algebra.solve!(u,solver::NonlinearFESolver,feop::ParamFEOperator,r::Realization)
  x = get_free_dof_values(u)
  op = get_algebraic_operator(feop)
  stats,paramcache = solve!(x,solver.nls,op,r)
  trial = paramcache.trial
  uh = FEFunction(trial,x)
  uh,stats
end

function Algebra.solve!(u,solver::LinearFESolver,feop::ParamFEOperator,r::Realization)
  x = get_free_dof_values(u)
  op = get_algebraic_operator(feop)
  stats,paramcache = solve!(x,solver.ls,op,r)
  trial = paramcache.trial
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

function Algebra.solve(solver::FESolver,op::LinearNonlinearParamFEOperator,r::Realization)
  solve(solver,join_operators(op),r)
end
