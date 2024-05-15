"""Recursive creation of a directory"""
function create_dir(dir::String)
  if !isdir(dir)
    parent_dir, = splitdir(dir)
    create_dir(parent_dir)
    mkdir(dir)
  end
  return
end

abstract type MDEIMStyle end
struct SpaceOnlyMDEIM <: MDEIMStyle end
struct SpaceTimeMDEIM <: MDEIMStyle end

struct RBSolver{S,M}
  fesolver::S
  ϵ::Float64
  mdeim_style::M
  nsnaps_state::Int
  nsnaps_mdeim::Int
  nsnaps_test::Int
  function RBSolver(
    fesolver::S,
    ϵ::Float64,
    mdeim_style::M=SpaceTimeMDEIM();
    nsnaps_state=50,
    nsnaps_mdeim=20,
    nsnaps_test=10) where {S,M}
    new{S,M}(fesolver,ϵ,mdeim_style,nsnaps_state,nsnaps_mdeim,nsnaps_test)
  end
end

const ThetaMethodRBSolver = RBSolver{ThetaMethod}

get_fe_solver(s::RBSolver) = s.fesolver
num_offline_params(solver::RBSolver) = solver.nsnaps_state
offline_params(solver::RBSolver) = 1:num_offline_params(solver)
num_online_params(solver::RBSolver) = solver.nsnaps_test
online_params(solver::RBSolver) = 1+num_offline_params(solver):num_online_params(solver)+num_offline_params(solver)
FEM.num_params(solver::RBSolver) = num_offline_params(solver) + num_online_params(solver)
num_mdeim_params(solver::RBSolver) = solver.nsnaps_mdeim
mdeim_params(solver::RBSolver) = 1:num_mdeim_params(solver)
get_tol(solver::RBSolver) = solver.ϵ

function get_test_directory(solver::RBSolver;dir=datadir())
  keyword = solver.mdeim_style == SpaceOnlyMDEIM() ? "space_only_mdeim" : "space_time_mdeim"
  test_dir = joinpath(dir,keyword * "_$(solver.ϵ)")
  create_dir(test_dir)
  test_dir
end

function fe_solutions(
  solver::RBSolver,
  op::TransientParamFEOperator,
  uh0::Function;
  kwargs...)

  fesolver = get_fe_solver(solver)
  nparams = num_params(solver)
  sol = solve(fesolver,op,uh0;nparams,kwargs...)
  odesol = sol.odesol
  realization = odesol.r

  stats = @timed begin
    values = collect(sol)
  end

  snaps = Snapshots(values,realization)
  cs = ComputationalStats(stats,nparams)
  return snaps,cs
end

function ode_solutions(
  solver::RBSolver,
  op::TransientParamFEOperator,
  uh0::Function;
  kwargs...)

  fesolver = get_fe_solver(solver)
  nparams = num_params(solver)
  sol = solve(fesolver,op,uh0;nparams,kwargs...)
  odesol = sol.odesol
  realization = odesol.r

  stats = @timed begin
    values = collect(odesol)
  end

  snaps = Snapshots(values,realization)
  cs = ComputationalStats(stats,nparams)
  return snaps,cs
end

function Algebra.solve(
  solver::RBSolver,
  feop::TransientParamFEOperator,
  uh0::Function;
  kwargs...)

  fesnaps,festats = ode_solutions(rbsolver,feop,uh0μ)
  rbop = reduced_operator(rbsolver,feop,fesnaps)
  rbsnaps,rbstats = solve(rbsolver,rbop,fesnaps)
  results = rb_results(rbsolver,rbop,fesnaps,rbsnaps,festats,rbstats)
  return results
end
