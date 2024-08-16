struct SpaceOnlyMDEIM <: MDEIMStyle end
struct SpaceTimeMDEIM <: MDEIMStyle end

RBSteady.get_mdeim_style_filename(::SpaceOnlyMDEIM) = "space_only_mdeim"
RBSteady.get_mdeim_style_filename(::SpaceTimeMDEIM) = "space_time_mdeim"

function RBSteady.RBSolver(
  fesolver::ODESolver,
  ϵ::Float64;
  mdeim_style=SpaceTimeMDEIM(),
  nsnaps_state=50,
  nsnaps_res=20,
  nsnaps_jac=20,
  nsnaps_test=10,
  fe_stats=CostTracker(),
  rb_offline_stats=CostTracker(),
  rb_online_stats=CostTracker())

  RBSolver(fesolver,ϵ,mdeim_style,nsnaps_state,nsnaps_res,nsnaps_jac,nsnaps_test,
    fe_stats,rb_offline_stats,rb_online_stats)
end

const ThetaMethodRBSolver = RBSolver{ThetaMethod}

function RBSteady.fe_solutions(
  solver::RBSolver,
  op::TransientParamFEOperator,
  uh0::Function;
  nparams=num_params(solver),
  r=realization(op;nparams))

  fesolver = get_fe_solver(solver)
  fe_stats = get_fe_stats(solver)
  reset_tracker!(fe_stats)

  sol = solve(fesolver,op,uh0,fe_stats;r)
  odesol = sol.odesol
  r = odesol.r

  values = collect(sol)

  i = get_vector_index_map(op)
  snaps = Snapshots(values,i,r)
  return snaps
end
