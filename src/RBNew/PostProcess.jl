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

function DrWatson.save(info::RBInfo,s::AbstractSnapshots)
  serialize(get_snapshots_filename(info),s)
end

get_op_filename(info::RBInfo) = info.dir * "/operator.jld"

function DrWatson.save(info::RBInfo,op::RBNonlinearOperator)
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

struct RBResults{A,B,BA,C,D}
  name::A
  sol::B
  sol_approx::BA
  fem_stats::C
  rb_stats::C
  norm_matrix::D
end

function rb_results(
  info::RBInfo,
  feop::TransientParamFEOperator,
  s,
  son_approx,
  fem_stats,
  rb_stats;
  name=info.variable_name)

  X = get_norm_matrix(info,feop)
  son = select_snapshots(s,online_params(info)) |> reverse_snapshots
  results = RBResults(name,son,son_approx,fem_stats,rb_stats,X)
  save(info,results)
  return results
end

function rb_results(solver::RBSolver,op::RBNonlinearOperator,args...;kwargs...)
  info = get_info(solver)
  feop = FEM.get_fe_operator(op)
  rb_results(info,feop,args...;kwargs...)
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

function space_time_error(sol::BlockSnapshots,sol_approx::BlockSnapshots,norm_matrix)
  @check get_touched_blocks(sol) == get_touched_blocks(sol_approx)
  space_time_error.(sol,sol_approx,norm_matrix)
end

function space_time_error(sol::BlockSnapshots,sol_approx::BlockSnapshots)
  @check get_touched_blocks(sol) == get_touched_blocks(sol_approx)
  norm_matrix = fill(nothing,size(sol))
  space_time_error(sol,sol_approx,norm_matrix)
end

function space_time_error(r::RBResults)
  space_time_error(r.sol,r.sol_approx,r.norm_matrix)
end

# plots

function FESpaces.FEFunction(
  fs::SingleFieldParamFESpace,s::AbstractSnapshots{Mode1Axis})
  r = get_realization(s)
  @assert FEM.length_free_values(fs) == length(r)
  free_values = _to_param_array(s.values)
  diri_values = get_dirichlet_dof_values(fs)
  FEFunction(fs,free_values,diri_values)
end

function FESpaces.FEFunction(
  fs::SingleFieldParamFESpace,s2::AbstractSnapshots{Mode2Axis})
  @warn "This snapshot has a mode-2 representation, the resulting FEFunction(s) might be incorrect"
  s = change_mode(s2)
  FEFunction(fs,s)
end

function _plot(solh::SingleFieldParamFEFunction,r::TransientParamRealization;dir=pwd(),varname="vel")
  trian = get_triangulation(solh)
  create_dir(dir)
  createpvd(dir) do pvd
    for (i,t) in enumerate(get_times(r))
      solh_t = FEM._getindex(solh,i)
      vtk = createvtk(trian,dir,cellfields=[varname=>solh_t])
      pvd[t] = vtk
    end
  end
end

# for the time being, plot only first param
function _get_at_first_param(trial,s)
  r = get_realization(s)
  r1 = FEM.get_at_param(r)
  free_values = get_values(s)
  free_values1 = free_values[1:num_times(r)]
  r1,FEFunction(trial(r1),free_values1)
end

function _plot(trial,s;kwargs...)
  r,sh = _get_at_first_param(trial,s)
  _plot(sh,r;kwargs...)
end

function _plot(trial::TransientMultiFieldTrialParamFESpace,s::BlockSnapshots;varname=("vel","press"),kwargs...)
  free_values = get_values(s)
  r = get_realization(s)
  trials = trial(r)
  sh = FEFunction(trials,free_values)
  nfields = length(trials.spaces)
  for n in 1:nfields
    _plot(sh[n],r,varname=varname[n];kwargs...)
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
