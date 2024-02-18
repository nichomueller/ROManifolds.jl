function load_solve(solver::RBSolver,args...;kwargs...)
  info = get_info(solver)
  snaps = deserialize(get_snapshots_filename(info))
  fem_stats = deserialize(get_stats_filename(info))
  rbop = deserialize(get_op_filename(info))
  rb_sol,rb_stats = solve(solver,rbop,snaps)
  results = rb_results(solver,rbop,snaps,rb_sol,fem_stats,rb_stats;kwargs...)
  return results
end

function DrWatson.save(s::RBSolver,args...)
  save(get_info(s),args...)
end

function DrWatson.save(info::RBInfo,args::Tuple)
  if info.save_structures
    map(a->save(info,a),args)
  end
end

get_snapshots_filename(info::RBInfo) = info.dir * "/snapshots.jld"

function DrWatson.save(info::RBInfo,s::AbstractTransientSnapshots)
  serialize(get_snapshots_filename(info),s)
end

get_op_filename(info::RBInfo) = info.dir * "/operator.jld"

function DrWatson.save(info::RBInfo,op::ReducedOperator)
  serialize(get_op_filename(info),op)
end

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

get_stats_filename(info::RBInfo) = info.dir * "/stats.jld"

function DrWatson.save(info::RBInfo,c::ComputationalStats)
  serialize(get_stats_filename(info),c)
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
  save(solver,results)
  return results
end

function rb_results(solver,op::ReducedOperator,args...;kwargs...)
  feop = FEM.get_fe_operator(op)
  rb_results(solver,feop,args...;kwargs...)
end

get_results_filename(info::RBInfo) = info.dir * "/results.jld"

function DrWatson.save(info::RBInfo,r::RBResults)
  serialize(get_results_filename(info),r)
end

function speedup(fem_stats::ComputationalStats,rb_stats::ComputationalStats)
  speedup_time = get_avg_time(fem_stats) / get_avg_time(rb_stats)
  speedup_memory = get_avg_nallocs(fem_stats) / get_avg_nallocs(rb_stats)
  return speedup_time,speedup_memory
end

function speedup(r::RBResults)
  speedup(r.fem_stats,r.rb_stats)
end

function space_time_error(sol,sol_approx,norm_matrix=nothing)
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

# plots

function FESpaces.FEFunction(
  fs::SingleFieldParamFESpace,s::AbstractTransientSnapshots{Mode1Axis})
  r = get_realization(s)
  @assert FEM.length_free_values(fs) == length(r)
  free_values = _to_param_array(s.values)
  diri_values = get_dirichlet_dof_values(fs)
  FEFunction(fs,free_values,diri_values)
end

function FESpaces.FEFunction(
  fs::SingleFieldParamFESpace,s2::AbstractTransientSnapshots{Mode2Axis})
  @warn "This snapshot has a mode-2 representation, the resulting FEFunction(s) might be incorrect"
  s = change_mode(s2)
  FEFunction(fs,s)
end

function _plot(
  trial::TransientTrialParamFESpace,
  s::AbstractTransientSnapshots;
  dir=pwd(),
  varname="u")

  r = get_realization(s)
  r0 = FEM.get_at_time(r,:initial)
  times = get_times(r)
  createpvd(r0,dir) do pvd
    for (it,t) = enumerate(times)
      rt = FEM.get_at_time(r,t)
      free_values = s.values[it]
      sht = FEFunction(trial(rt),free_values)
      files = ParamString(dir,rt)
      trian = get_triangulation(sht)
      vtk = createvtk(trian,files,cellfields=[varname=>sht])
      pvd[rt] = vtk
    end
  end
end

function generate_plots(
  solver::RBSolver,
  feop::TransientParamFEOperator,
  r::RBResults)

  sol,sol_approx = r.sol,r.sol_approx
  trial = get_trial(feop)
  info = get_info(solver)
  plt_dir = joinpath(info.dir,"plots")
  fe_plt_dir = joinpath(plt_dir,"fe_solution")
  _plot(trial,sol;dir=fe_plt_dir,varname=r.name)
  rb_plt_dir = joinpath(plt_dir,"rb_solution")
  _plot(trial,sol_approx;dir=rb_plt_dir,varname=r.name)
end
