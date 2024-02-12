function get_test_dir(path::String,ϵ;st_mdeim=false)
  keyword = st_mdeim ? "st" : "standard"
  outer_path = joinpath(path,keyword)
  dir = joinpath(outer_path,"$ϵ")
  dir
end

struct SpaceOnlyMDEIM end
struct SpaceTimeMDEIM end

struct RBInfo{M}
  ϵ::Float64
  mdeim_style::M
  norm_style::Symbol
  dir::String
  nsnaps_state::Int
  nsnaps_mdeim::Int
  nsnaps_test::Int
  save_structures::Bool
end

function RBInfo(
  test_path::String;
  ϵ=1e-4,
  st_mdeim=false,
  norm_style=:l2,
  nsnaps_state=50,
  nsnaps_mdeim=20,
  nsnaps_test=10,
  save_structures=true)

  mdeim_style = st_mdeim == true ? SpaceTimeMDEIM() : SpaceOnlyMDEIM()
  dir = get_rb_path(test_path,ϵ;st_mdeim)
  RBInfo(ϵ,mdeim_style,norm_style,dir,nsnaps_state,
    nsnaps_mdeim,nsnaps_test,save_structures)
end

num_offline_params(info::RBInfo) = info.nsnaps_state
offline_params(info::RBInfo) = 1:num_offline_params(info)
num_online_params(info::RBInfo) = info.nsnaps_test
online_params(info::RBInfo) = 1+num_offline_params(info):num_online_params(info)+num_offline_params(info)
FEM.num_params(info::RBInfo) = num_offline_params(info) + num_online_params(info)
num_mdeim_params(info::RBInfo) = info.nsnaps_mdeim
mdeim_params(info::RBInfo) = 1:num_offline_params(info)
get_tol(info::RBInfo) = info.ϵ

function get_norm_matrix(info::RBInfo,feop::TransientParamFEOperator)
  norm_style = info.norm_style
  try
    T = get_vector_type(feop.test)
    load(info,SparseMatrixCSC{eltype(T),Int};norm_style)
  catch
    if norm_style == :l2
      nothing
    elseif norm_style == :L2
      get_L2_norm_matrix(feop)
    elseif norm_style == :H1
      get_H1_norm_matrix(feop)
    else
      @unreachable
    end
  end
end

function DrWatson.save(info::RBInfo,args::Tuple)
  if info.save_structures
    map(a->save(info,a),args)
  end
end

struct RBSolver{S}
  info::RBInfo
  fesolver::S
end

const RBThetaMethod = RBSolver{ThetaMethod}

get_fe_solver(s::RBSolver) = s.fesolver
get_info(s::RBSolver) = s.info

function RBSolver(fesolver,dir;kwargs...)
  info = RBInfo(dir;kwargs...)
  RBSolver(info,fesolver)
end

function DrWatson.save(s::RBSolver,args...)
  save(get_info(s),args...)
end

function load_solve(
  solver::RBSolver,
  feop::TransientParamFEOperator,
  args...;
  kwargs...)

  snaps = wload(get_snapshots_dir(solver))
  fem_stats = wload(get_stats_dir(solver))
  rbop = wload(get_reduced_operator_dir(solver))
  rb_sol,rb_stats = solve(solver,rbop,snaps)
  results = rb_results(solver,rbop,snaps,rb_sol,fem_stats,rb_stats;kwargs...)
  return results
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

# for visualization/testing purposes
struct ComputationalStats
  avg_time::Float64
  avg_nallocs::Float64
  function ComputationalStats(stats::NamedTuple,nruns::Int)
    avg_time = stats[:time] / nruns
    avg_nallocs = stats[:bytes] / (1e6*nruns)
    new(avg_time,avg_nallocs)
  end
end

get_avg_time(c::ComputationalStats) = c.avg_time
get_avg_nallocs(c::ComputationalStats) = c.avg_nallocs

get_stats_dir(info::RBInfo) = info.dir * "stats"

function DrWatson.save(info::RBInfo,c::ComputationalStats)
  wsave(get_stats_dir(info),c)
end

struct RBResults
  name::Symbol
  sol::TransientSnapshotsSwappedColumns
  sol_approx::TransientSnapshotsSwappedColumns
  fem_stats::ComputationalStats
  rb_stats::ComputationalStats
  norm_matrix
end

function rb_results(
  solver::RBSolver,
  feop::TransientParamFEOperator,
  s::AbstractTransientSnapshots,
  son_approx::TransientSnapshotsSwappedColumns,
  fem_stats::ComputationalStats,
  rb_stats::ComputationalStats,
  name=:vel)

  info = get_info(solver)
  X = get_norm_matrix(info,feop)
  son = select_snapshots(s,online_params(info))
  son_rev = reverse_snapshots(son)
  results = RBResults(name,son_rev,son_approx,fem_stats,rb_stats,X)
  save(results)
  return results
end

get_results_dir(info::RBInfo) = info.dir * "results"

function DrWatson.save(info::RBInfo,r::RBResults)
  wsave(get_results_dir(info),r)
end

function speedup(fem_stats::ComputationalStats,rb_stats::ComputationalStats)
  speedup_time = get_avg_time(fem_stats) / get_avg_time(rb_stats)
  speedup_memory = get_avg_nallocs(fem_stats) / get_avg_nallocs(rb_stats)
  return speedup_time,speedup_memory
end

function speedup(r::RBResults)
  speedup(r.fem_stats,r.rb_stats)
end

function space_time_error(sol,sol_approx,norm_matrix)
  _norm(v::AbstractVector,::Nothing) = norm(v)
  _norm(v::AbstractVector,X::AbstractMatrix) = sqrt(v'*X*v)
  err_norm = []
  sol_norm = []
  space_time_norm = []
  for i = axes(sol,2)
    push!(err_norm,_norm(sol[:,i]-sol_approx[:,i],norm_matrix))
    push!(sol_norm,_norm(sol[:,i],norm_matrix))
    if mod(i,num_params(sol)) == 0
      push!(space_time_norm,norm(err_norm)/norm(sol_norm))
      err_norm = []
      sol_norm = []
    end
  end
  avg_error = sum(space_time_norm) / length(space_time_norm)
  return avg_error
end

function space_time_error(r::RBResults)
  space_time_error(r.sol,r.sol_approx,r.norm_matrix)
end

function _plot(solver::RBSolver,feop::TransientParamFEOperator,r::RBResults)
  sol,sol_approx = r.sol,r.sol_approx
  trial = get_trial(feop)
  info = get_info(solver)
  plt_dir = joinpath(info.dir,"plots")
  fe_plt_dir = joinpath(plt_dir,"fe_solution")
  _plot(trial,sol;dir=fe_plt_dir,varname=r.name)
  rb_plt_dir = joinpath(plt_dir,"rb_solution")
  _plot(trial,sol_approx;dir=rb_plt_dir,varname=r.name)
end
