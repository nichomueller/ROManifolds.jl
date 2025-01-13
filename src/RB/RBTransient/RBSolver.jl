# check TransientMDEIMReduction for more details
time_combinations(fesolver::ODESolver) = @notimplemented

function time_combinations(fesolver::GeneralizedAlpha1)
  combine_res(x) = nothing
  combine_jac(x,y) = nothing
  combine_djac(x,y) = nothing
  return combine_res,combine_jac,combine_djac
end

function time_combinations(fesolver::GeneralizedAlpha2)
  combine_res(x) = nothing
  combine_jac(x,y) = nothing
  combine_djac(x,y) = nothing
  combine_ddjac(x,y) = nothing
  return combine_res,combine_jac,combine_djac,combine_ddjac
end

function time_combinations(fesolver::ThetaMethod)
  dt,θ = fesolver.dt,fesolver.θ
  combine_res(x) = x
  combine_jac(x,y) = θ*x+(1-θ)*y
  combine_djac(x,y) = (x-y)/dt
  return combine_res,combine_jac,combine_djac
end

function RBSteady.RBSolver(
  fesolver::ODESolver,
  state_reduction::Reduction;
  nparams_res=20,
  nparams_jac=20,
  nparams_djac=nparams_jac)

  red_style = ReductionStyle(state_reduction)
  cres,cjac,cdjac = time_combinations(fesolver)

  residual_reduction = TransientMDEIMReduction(cres,red_style;nparams=nparams_res)
  jac_reduction = TransientMDEIMReduction(cjac,red_style;nparams=nparams_jac)
  djac_reduction = TransientMDEIMReduction(cdjac,red_style;nparams=nparams_djac)
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
  values,stats = collect(sol.odesol)
  initial_values = collect_initial_values(sol.odesol)
  i = get_dof_map(feop)
  snaps = Snapshots(values,initial_values,i,r)
  return snaps,stats
end

function RBSteady.residual_snapshots(
  solver::RBSolver,
  odeop::ODEParamOperator,
  s::AbstractSnapshots;
  nparams=RBSteady.res_params(solver))

  fesolver = get_fe_solver(solver)
  sres = select_snapshots(s,nparams)
  us_res = (get_values(sres),)
  us0_res = get_initial_values(sres)
  r_res = get_realization(sres)
  b = residual(fesolver,odeop,r_res,us_res,us0_res)
  ib = get_dof_map_at_domains(odeop)
  return Snapshots(b,ib,r_res)
end

function RBSteady.jacobian_snapshots(
  solver::RBSolver,
  odeop::ODEParamOperator,
  s::AbstractSnapshots;
  nparams=RBSteady.jac_params(solver))

  fesolver = get_fe_solver(solver)
  sjac = select_snapshots(s,nparams)
  us_jac = (get_values(sjac),)
  us0_jac = get_initial_values(sjac)
  r_jac = get_realization(sjac)
  A = jacobian(fesolver,odeop,r_jac,us_jac,us0_jac)
  iA = get_sparse_dof_map_at_domains(odeop)
  jac_reduction = RBSteady.get_jacobian_reduction(solver)
  sA = ()
  for (reda,a,ia) in zip(jac_reduction,A,iA)
    sa = Snapshots(a,ia,r_jac)
    sA = (sA...,select_snapshots(sa,1:num_params(reda)))
  end
  return sA
end
