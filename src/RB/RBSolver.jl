"""Recursive creation of a directory"""
function create_dir(dir::String)
  if !isdir(dir)
    parent_dir, = splitdir(dir)
    create_dir(parent_dir)
    mkdir(dir)
  end
  return
end

abstract type RBSolver{S} end
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

struct SpaceOnlyMDEIM end
struct SpaceTimeMDEIM end

struct PODMDEIMSolver{S,M} <: RBSolver{S}
  fesolver::S
  ϵ::Float64
  mdeim_style::M
  nsnaps_state::Int
  nsnaps_mdeim::Int
  nsnaps_test::Int
  function PODMDEIMSolver(
    fesolver::S,
    ϵ::Float64,
    mdeim_style::M;
    nsnaps_state=50,
    nsnaps_mdeim=20,
    nsnaps_test=10) where {S,M}
    new{S,M}(fesolver,ϵ,mdeim_style,nsnaps_state,nsnaps_mdeim,nsnaps_test)
  end
end

function RBSolver(fesolver,ϵ=1e-4,st_mdeim=SpaceTimeMDEIM();kwargs...)
  PODMDEIMSolver(fesolver,ϵ,st_mdeim;kwargs...)
end

function get_test_directory(solver::PODMDEIMSolver;dir=datadir())
  keyword = solver.mdeim_style == SpaceOnlyMDEIM() ? "space_only_mdeim" : "space_time_mdeim"
  test_dir = joinpath(dir,keyword * "_$(solver.ϵ)")
  create_dir(test_dir)
  test_dir
end

struct TTPODMDEIMSolver{S,M} <: RBSolver{S}
  fesolver::S
  ϵ::Float64
  mdeim_style::M
  nsnaps_state::Int
  nsnaps_mdeim::Int
  nsnaps_test::Int
  function TTPODMDEIMSolver(
    fesolver::S,
    ϵ::Float64;
    nsnaps_state=50,
    nsnaps_mdeim=20,
    nsnaps_test=10) where S
    mdeim_style = SpaceTimeMDEIM()
    M = typeof(mdeim_style)
    new{S,M}(fesolver,ϵ,mdeim_style,nsnaps_state,nsnaps_mdeim,nsnaps_test)
  end
end

function TTRBSolver(fesolver,ϵ=1e-4;kwargs...)
  TTPODMDEIMSolver(fesolver,ϵ;kwargs...)
end

function get_test_directory(solver::TTPODMDEIMSolver;dir=datadir())
  keyword = solver.mdeim_style == SpaceOnlyMDEIM() ? "TT_space_only_mdeim" : "TT_space_time_mdeim"
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
  sol = solve(fesolver,op,uh0;kwargs...)
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
  sol = solve(fesolver,op,uh0;kwargs...)
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

  snaps,fem_stats = fe_solutions(solver,feop,uh0)
  rbop = reduced_operator(solver,feop,snaps)
  rb_sol,rb_stats = solve(solver,rbop,snaps)
  results = rb_results(solver,rbop,snaps,rb_sol,fem_stats,rb_stats;kwargs...)
  return results
end
