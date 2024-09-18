function RBSteady.RBSolver(fesolver::ODESolver,state_reduction::AbstractReduction;kwargs...)
  @notimplemented
end

function RBSteady.RBSolver(
  fesolver::ThetaMethod,
  state_reduction::AbstractReduction;
  nparams_res=20,
  nparams_jac=20,
  nparams_djac=nparams_jac,
  nparams_test=10)

  θ = fesolver.θ
  combine_res = (x) -> x
  combine_jac = (x,y) -> θ*x+(1-θ)*y
  combine_djac = (x,y) -> θ*(x-y)

  red_style = ReductionStyle(state_reduction)

  residual_reduction = TransientMDEIMReduction(combine_res,red_style;nparams=nparams_res,nparams_test)
  jac_reduction = TransientMDEIMReduction(combine_jac,red_style;nparams=nparams_jac,nparams_test)
  djac_reduction = TransientMDEIMReduction(combine_djac,red_style;nparams=nparams_djac,nparams_test)
  jacobian_reduction = (jac_reduction,djac_reduction)

  RBSolver(fesolver,state_reduction,residual_reduction,jacobian_reduction)
end

RBSteady.num_jac_params(s::RBSolver{<:ODESolver}) = num_params(first(s.jacobian_reduction))

function RBSteady.fe_snapshots(
  solver::RBSolver,
  op::TransientParamFEOperator,
  uh0::Function;
  nparams=num_params(solver),
  r=realization(op;nparams))

  fesolver = get_fe_solver(solver)

  sol = solve(fesolver,op,uh0;r)
  odesol = sol.odesol
  r = odesol.r

  values,stats = collect(sol)

  i = get_vector_index_map(op)
  snaps = Snapshots(values,i,r)
  return snaps,stats
end
