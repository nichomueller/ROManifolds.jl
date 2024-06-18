function load_solve(solver;dir=pwd(),kwargs...)
  snaps = deserialize(get_snapshots_filename(dir))
  rbop = deserialize(get_op_filename(dir))
  rb_sol,rb_stats = solve(solver,rbop,snaps)
  old_results = deserialize(get_results_filename(dir))
  old_fem_stats = old_results.fem_stats
  results = rb_results(solver,rbop,snaps,rb_sol,old_fem_stats,rb_stats;kwargs...)
  return results
end

function DrWatson.save(dir,args::Tuple)
  map(a->save(dir,a),args)
end

function get_snapshots_filename(dir)
  parent_dir, = splitdir(dir)
  parent_dir * "/snapshots.jld"
end

function DrWatson.save(dir,s::Union{AbstractSnapshots,BlockSnapshots})
  serialize(get_snapshots_filename(dir),s)
end

function get_op_filename(dir)
  dir * "/operator.jld"
end

function DrWatson.save(dir,op::RBOperator)
  serialize(dir * "/operator.jld",op)
end

struct ComputationalStats
  avg_time::Float64
  avg_nallocs::Float64
  function ComputationalStats(stats::NamedTuple,nruns::Integer)
    avg_time = stats[:time] / nruns
    avg_nallocs = stats[:bytes] / (1e6*nruns)
    new(avg_time,avg_nallocs)
  end
end

get_avg_time(c::ComputationalStats) = c.avg_time
get_avg_nallocs(c::ComputationalStats) = c.avg_nallocs

struct RBResults{A,B,BA,C,D}
  name::A
  sol::B
  sol_approx::BA
  fem_stats::C
  rb_stats::C
  norm_matrix::D
end

function rb_results(
  solver::RBSolver,
  feop,
  s,
  son_approx,
  fem_stats,
  rb_stats;
  name="vel")

  X = assemble_norm_matrix(feop)
  son = select_snapshots(s,online_params(solver))
  RBResults(name,son,son_approx,fem_stats,rb_stats,X)
end

function rb_results(solver::RBSolver,op::RBOperator,args...;kwargs...)
  feop = ParamODEs.get_fe_operator(op)
  rb_results(solver,feop,args...;kwargs...)
end

function get_results_filename(dir)
  dir * "/results.jld"
end

function DrWatson.save(dir,r::RBResults)
  serialize(get_results_filename(dir),r)
end

function compute_speedup(fem_stats::ComputationalStats,rb_stats::ComputationalStats)
  speedup_time = get_avg_time(fem_stats) / get_avg_time(rb_stats)
  speedup_memory = get_avg_nallocs(fem_stats) / get_avg_nallocs(rb_stats)
  return speedup_time,speedup_memory
end

function compute_speedup(r::RBResults)
  compute_speedup(r.fem_stats,r.rb_stats)
end

function compute_error(_sol::AbstractSnapshots,_sol_approx::AbstractSnapshots,args...)
  sol = flatten_snapshots(_sol)
  sol_approx = flatten_snapshots(_sol_approx)
  compute_error(sol,sol_approx,args...)
end

function compute_error(sol::UnfoldingSteadySnapshots,sol_approx::UnfoldingSteadySnapshots,norm_matrix=nothing)
  @check size(sol) == size(sol_approx)
  space_norm = zeros(num_params(sol))
  @inbounds for i = axes(sol,2)
    err_norm = _norm(sol[:,i]-sol_approx[:,i],norm_matrix)
    sol_norm = _norm(sol[:,i],norm_matrix)
    space_norm[i] = err_norm / sol_norm
  end
  avg_error = sum(space_norm) / length(space_norm)
  return avg_error
end

function compute_error(sol::BlockSnapshots,sol_approx::BlockSnapshots,norm_matrix::BlockMatrix)
  @check get_touched_blocks(sol) == get_touched_blocks(sol_approx)
  active_block_ids = get_touched_blocks(sol)
  block_map = BlockMap(size(sol),active_block_ids)
  errors = [compute_error(sol[i],sol_approx[i],norm_matrix[Block(i,i)]) for i in active_block_ids]
  return_cache(block_map,errors...)
end

function compute_error(r::RBResults)
  compute_error(r.sol,r.sol_approx,r.norm_matrix)
end

# # plots

# function FESpaces.FEFunction(
#   fs::SingleFieldParamFESpace,s::AbstractSnapshots{Mode1Axis})
#   r = get_realization(s)
#   @assert param_length(fs) == length(r)
#   free_values = _to_param_array(s.values)
#   diri_values = get_dirichlet_dof_values(fs)
#   FEFunction(fs,free_values,diri_values)
# end

# function _plot(solh::SingleFieldParamFEFunction,r::TransientParamRealization;dir=pwd(),varname="vel")
#   trian = get_triangulation(solh)
#   create_dir(dir)
#   createpvd(dir) do pvd
#     for (i,t) in enumerate(get_times(r))
#       solh_t = param_getindex(solh,i)
#       vtk = createvtk(trian,dir,cellfields=[varname=>solh_t])
#       pvd[t] = vtk
#     end
#   end
# end

# function _plot(trial,s;kwargs...)
#   r,sh = _get_at_first_param(trial,s)
#   _plot(sh,r;kwargs...)
# end

# function _plot(trial::TransientMultiFieldParamFESpace,s::BlockSnapshots;varname=("vel","press"),kwargs...)
#   free_values = get_values(s)
#   r = get_realization(s)
#   trial = trial(r)
#   sh = FEFunction(trial,free_values)
#   nfields = length(trial.spaces)
#   for n in 1:nfields
#     _plot(sh[n],r,varname=varname[n];kwargs...)
#   end
# end

# function generate_plots(feop::TransientParamFEOperator,r::RBResults;dir=pwd())
#   sol,sol_approx = r.sol,r.sol_approx
#   trial = get_trial(feop)
#   plt_dir = joinpath(dir,"plots")
#   fe_plt_dir = joinpath(plt_dir,"fe_solution")
#   _plot(trial,sol;dir=fe_plt_dir,varname=r.name)
#   rb_plt_dir = joinpath(plt_dir,"rb_solution")
#   _plot(trial,sol_approx;dir=rb_plt_dir,varname=r.name)
# end
