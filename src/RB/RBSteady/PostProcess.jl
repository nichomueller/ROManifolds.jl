function load_solve(solver,feop;dir=pwd(),kwargs...)
  snaps = deserialize(get_snapshots_filename(dir))
  old_rbop = deserialize(get_op_filename(dir))
  rbop = deserialize_operator(feop,old_rbop)
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
  dir * "/snapshots.jld"
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

function deserialize_operator(feop,op)
  @abstractmethod
end

function deserialize_operator(feop::ParamFEOperatorWithTrian,rbop::PGMDEIMOperator)
  trian_res = feop.trian_res
  trian_jac = feop.trian_jac
  rhs_old,lhs_old = rbop.rhs,rbop.lhs

  iperm_jac,trian_jac_new = map(t->find_closest_view(trian_jac,t),lhs_old.trians) |> tuple_of_arrays
  value_jac_new = map(i -> getindex(lhs_old.values,i...),iperm_jac)
  lhs_new = Contribution(value_jac_new,trian_jac_new)

  iperm_res,trian_res_new = map(t->find_closest_view(trian_res,t),rhs_old.trians) |> tuple_of_arrays
  value_res_new = map(i -> getindex(rhs_old.values,i...),iperm_res)
  rhs_new = Contribution(value_res_new,trian_res_new)

  return PGMDEIMOperator(rbop.op,lhs_new,rhs_new)
end

function deserialize_operator(
  feop::LinearNonlinearParamFEOperatorWithTrian,
  rbop::LinearNonlinearPGMDEIMOperator)

  rbop_lin = deserialize_operator(get_linear_operator(feop),get_linear_operator(rbop))
  rbop_nlin = deserialize_operator(get_nonlinear_operator(feop),get_nonlinear_operator(rbop))
  return LinearNonlinearPGMDEIMOperator(rbop_lin,rbop_nlin)
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
  println("Speedup in time: $(speedup_time)")
  println("Speedup in memory: $(speedup_memory)")
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

function average_plot(
  trial::TrialParamFESpace,
  v::AbstractVector;
  name=:vel,
  dir=joinpath(pwd(),"plots"))

  create_dir(dir)
  trian = get_triangulation(trial)
  vh = FEFunction(param_getindex(trial,1),v)
  vtk = createvtk(trian,dir,cellfields=[name=>vh])
end

function average_plot(trial::FESpace,r::RBResults;kwargs...)
  r̄ = mean(get_realization(r.sol))
  r₀ = zero(get_realization(r.sol))
  average_plot(trial(r̄),mean(r.sol);name=r.name,kwargs...)
  average_plot(trial(r₀),mean(r.sol - r.sol_approx);name=r.name,kwargs...)
end

function average_plot(trial::MultiFieldFESpace,r::RBResults;kwargs...)
  r̄ = mean(get_realization(r.sol))
  r₀ = zero(get_realization(r.sol))
  for (i,Ui) in trial
    average_plot(Ui(r̄),mean(r.sol[i]);name=r.name[i],kwargs...)
    average_plot(Ui(r₀),mean(r.sol[i] - r.sol_approx[i]);name=r.name[i],kwargs...)
  end
end
