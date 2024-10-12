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
  nls::NonlinearSolver,
  op::ParamOpFromFEOp,
  r::Realization)

  paramcache = allocate_paramcache(op,r,x)
  A = paramcache.A
  b = paramcache.b

  t = @timed begin
    jacobian!(A,op,r,x,paramcache)
    residual!(b,op,r,x,paramcache)
    solve_param_nr!(x,A,b,r,op,paramcache)
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

# utils

function solve_param_nr!(x,A,b,dx,r,ns,nls::NewtonRaphsonSolver,op,paramcache)
  @notimplemented "Use NewtonSolver instead"
end

function solve_param_nr!(x,A,b,r,nls::GridapSolvers.NewtonSolver,op,paramcache)
  dx = allocate_in_domain(A)
  ss = symbolic_setup(nls.ls,A)
  ns = numerical_setup(ss,A,x)
  log = nls.log

  res = norm(b)
  done = LinearSolvers.init!(log,res)
  while !done
    rmul!(b,-1)
    solve!(dx,ns,b)
    x .+= dx

    residual!(b,op,r,x,paramcache)
    res  = norm(b)
    done = LinearSolvers.update!(log,res)

    if !done
      jacobian!(A,op,r,x,paramcache)
      numerical_setup!(ns,A,x)
    end
  end

  LinearSolvers.finalize!(log,res)
  return x
end
