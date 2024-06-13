struct SpaceOnlyMDEIM <: MDEIMStyle end
struct SpaceTimeMDEIM <: MDEIMStyle end

RBSteady.get_mdeim_style_filename(::SpaceOnlyMDEIM) = "space_only_mdeim"
RBSteady.get_mdeim_style_filename(::SpaceTimeMDEIM) = "space_time_mdeim"

function RBSteady.RBSolver(
  fesolver::ODESolver,
  ϵ::Float64;
  nsnaps_state=50,
  nsnaps_mdeim=20,
  nsnaps_test=10)

  mdeim_style = SpaceTimeMDEIM()
  RBSolver(fesolver,ϵ,mdeim_style,nsnaps_state,nsnaps_mdeim,nsnaps_test)
end

const ThetaMethodRBSolver = RBSolver{ThetaMethod}

function RBSteady.fe_solutions(
  solver::RBSolver,
  op::TransientParamFEOperator,
  uh0::Function;
  kwargs...)

  fesolver = get_fe_solver(solver)
  nparams = num_params(solver)
  sol = solve(fesolver,op,uh0;nparams,kwargs...)
  odesol = sol.odesol
  r = odesol.r

  stats = @timed begin
    values = collect(odesol)
  end

  snaps = Snapshots(values,r)
  cs = ComputationalStats(stats,nparams)
  return snaps,cs
end
