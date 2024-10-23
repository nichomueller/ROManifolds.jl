function Algebra.solve!(u,solver::NonlinearFESolver,feop::ParamFEOperator,r::Realization)
  x = get_free_dof_values(u)
  op = get_algebraic_operator(feop)
  nlop = ParamNonlinearOperator(op,r)
  t = @timed solve!(x,solver.nls,nlop)
  stats = CostTracker(t,name="FEM")
  trial = get_trial(feop)(r)
  uh = FEFunction(trial,x)
  uh,stats
end

function Algebra.solve!(u,solver::LinearFESolver,feop::ParamFEOperator,r::Realization)
  x = get_free_dof_values(u)
  op = get_algebraic_operator(feop)
  nlop = ParamNonlinearOperator(op,r)
  t = @timed solve!(x,solver.ls,nlop)
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

# function Algebra.solve(solver::FESolver,op::LinearNonlinearParamFEOperator,r::Realization)
#   lop = get_linear_operator(op)
#   nlop = get_nonlinear_operator(op)
#   if isa(lop,ParamFEOperatorWithTrian) || isa(nlop,ParamFEOperatorWithTrian)
#     @assert isa(lop,ParamFEOperatorWithTrian) && isa(nlop,ParamFEOperatorWithTrian)
#     op′ = LinearNonlinearParamFEOperator(lop.op,nlop.op)
#   else
#     op′ = op
#   end
#   U = get_trial(op′)(r)
#   uh = zero(U)
#   vh,stats = solve!(uh,solver,op′,r)
#   vh,stats
# end
function Algebra.solve(solver::FESolver,op::LinearNonlinearParamFEOperator,r::Realization)
  solve(solver,join_operators(op),r)
end
