function RBSteady.RBSolver(fesolver::ODESolver,state_reduction::AbstractReduction;kwargs...)
  @notimplemented
end

function RBSteady.RBSolver(
  fesolver::ThetaMethod,
  state_reduction::AbstractReduction;
  nparams_res=20,
  nparams_jac=20,
  nparams_djac=nparams_jac)

  θ = fesolver.θ
  combine_res = (x) -> x
  combine_jac = (x,y) -> θ*x+(1-θ)*y
  combine_djac = (x,y) -> θ*(x-y)

  red_style = ReductionStyle(state_reduction)

  residual_reduction = TransientMDEIMReduction(combine_res,red_style;nparams=nparams_res)
  jac_reduction = TransientMDEIMReduction(combine_jac,red_style;nparams=nparams_jac)
  djac_reduction = TransientMDEIMReduction(combine_djac,red_style;nparams=nparams_djac)
  jacobian_reduction = (jac_reduction,djac_reduction)

  RBSolver(fesolver,state_reduction,residual_reduction,jacobian_reduction)
end

RBSteady.num_jac_params(s::RBSolver{<:ODESolver}) = num_params(first(s.jacobian_reduction))

function RBSteady.solution_snapshots(
  solver::RBSolver,
  feop::TransientParamFEOperator,
  args...;
  nparams=RBSteady.num_offline_params(solver),
  r=realization(feop;nparams))

  solution_snapshots(solver,feop,r,args...)
end

function RBSteady.solution_snapshots(
  solver::RBSolver,
  feop::TransientParamFEOperator,
  r::TransientRealization,
  args...)

  fesolver = get_fe_solver(solver)
  sol = solve(fesolver,feop,r,args...)
  odesol = sol.odesol
  values,stats = collect(sol)
  i = get_vector_index_map(feop)
  snaps = Snapshots(values,i,r)
  return snaps,stats
end

function RBSteady.residual_snapshots(
  solver::RBSolver,
  odeop::ODEParamOperator,
  snaps)

  fesolver = get_fe_solver(solver)
  sres = select_snapshots(snaps,RBSteady.res_params(solver))
  us_res = (get_values(sres),)
  r_res = get_realization(sres)
  b = residual(fesolver,odeop,r_res,us_res)
  ib = get_vector_index_map(odeop)
  return Snapshots(b,ib,r_res)
end

function RBSteady.jacobian_snapshots(
  solver::RBSolver,
  odeop::ODEParamOperator,
  snaps)

  fesolver = get_fe_solver(solver)
  sjac = select_snapshots(snaps,RBSteady.jac_params(solver))
  us_jac = (get_values(sjac),)
  r_jac = get_realization(sjac)
  A = jacobian(fesolver,odeop,r_jac,us_jac)
  iA = get_matrix_index_map(odeop)
  return Snapshots(A,iA,r_jac)
end
