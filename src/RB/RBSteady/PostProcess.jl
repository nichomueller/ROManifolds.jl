"""
    load_solve(solver::RBSolver,feop::ParamFEOperator,dir::String;kwargs...) -> RBResults
    load_solve(solver::RBSolver,feop::TransientParamFEOperator,dir::String;kwargs...) -> RBResults

Loads the snapshots previously saved to file, loads the reduced operator
previously saved to file, and returns the results. This function allows to entirely
skip the RB's offline phase. The field `dir` should be the directory at which the
saved quantities can be found. Note that this function must be used after the test
case has been run at least once!

"""
function load_solve(solver,feop,dir;kwargs...)
  snaps = deserialize(get_snapshots_filename(dir))
  rbop = deserialize_operator(feop,dir)
  rb_sol,rb_stats = solve(solver,rbop,snaps)
  old_results = deserialize(get_results_filename(dir))
  old_fem_stats = old_results.fem_stats
  results = rb_results(solver,rbop,snaps,rb_sol,old_fem_stats,rb_stats;kwargs...)
  return results
end

function deserialize_operator(feop::ParamFEOperatorWithTrian,dir)
  trian_res = feop.trian_res
  trian_jac = feop.trian_jac

  op = deserialize_pg_operator(feop,dir)
  red_rhs = deserialize_contribution(dir,trian_res,get_test(op);label="res")
  red_lhs = deserialize_contribution(dir,trian_jac,get_trial(op),get_test(op);label="jac")
  trians_rhs = get_domains(red_rhs)
  trians_lhs = get_domains(red_lhs)
  new_op = change_triangulation(op,trians_rhs,trians_lhs)
  rbop = PGMDEIMOperator(new_op,red_lhs,red_rhs)
  return rbop
end

function deserialize_operator(
  feop::LinearNonlinearParamFEOperatorWithTrian,
  rbop::LinearNonlinearPGMDEIMOperator)

  rbop_lin = deserialize_operator(get_linear_operator(feop),get_linear_operator(rbop))
  rbop_nlin = deserialize_operator(get_nonlinear_operator(feop),get_nonlinear_operator(rbop))
  return LinearNonlinearPGMDEIMOperator(rbop_lin,rbop_nlin)
end

function deserialize_pg_operator(feop::ParamFEOperatorWithTrian,dir)
  op = get_algebraic_operator(feop)
  fe_test = get_test(feop)
  fe_trial = get_trial(feop)
  basis_test = deserialize(get_projection_filename(dir;label="test"))
  basis_trial = deserialize(get_projection_filename(dir;label="trial"))
  test = fe_subspace(fe_test,basis_test)
  trial = fe_subspace(fe_trial,basis_trial)
  return PGOperator(op,trial,test)
end

function deserialize_contribution(dir,trian,args...;label="res")
  ad,redt = (),()
  for (i,t) in enumerate(trian)
    adi = deserialize(get_decomposition_filename(dir;label=label*"_$i"))
    redti = reduce_triangulation(t,get_integration_domain(adi),args...)
    if isa(redti,AbstractArray)
      redti = ParamDataStructures.merge_triangulations(redti)
    end
    ad = (ad...,adi)
    redt = (redt...,redti)
  end
  return Contribution(ad,redt)
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

function get_projection_filename(dir;label="test")
  dir * "/basis_$(label).jld"
end

function DrWatson.save(dir,b::Projection;kwargs...)
  serialize(get_projection_filename(dir;kwargs...),b)
end

function get_decomposition_filename(dir;label="res")
  dir * "/affdec_$(label).jld"
end

function DrWatson.save(dir,ad::AffineDecomposition;kwargs...)
  serialize(get_decomposition_filename(dir;kwargs...),ad)
end

function DrWatson.save(dir,op::PGMDEIMOperator;kwargs...)
  save(dir,op.op;kwargs...)
  for (i,ad_res) in enumerate(op.rhs.values)
    save(dir,ad_res;label="res_$i")
  end
  for (i,ad_jac) in enumerate(op.lhs.values)
    save(dir,ad_jac;label="jac_$i")
  end
end

function DrWatson.save(dir,op::PGOperator;kwargs...)
  btest = get_basis(get_test(op))
  btrial = get_basis(get_trial(op))
  save(dir,btest;label="test")
  save(dir,btest;label="trial")
end

"""
    struct ComputationalStats
      avg_time::Float64
      avg_nallocs::Float64
    end

"""
struct ComputationalStats
  avg_time::Float64
  avg_nallocs::Float64
end

function ComputationalStats(stats::NamedTuple,nruns::Integer)
  avg_time = stats[:time] / nruns
  avg_nallocs = stats[:bytes] / (1e6*nruns)
  ComputationalStats(avg_time,avg_nallocs)
end

get_avg_time(c::ComputationalStats) = c.avg_time
get_avg_nallocs(c::ComputationalStats) = c.avg_nallocs

function get_stats(t::IterativeCostTracker)
  avg_time = t.time / t.nruns
  avg_nallocs = t.nallocs / t.nruns
  ComputationalStats(avg_time,avg_nallocs)
end

"""
    struct RBResults{A,B,BA,C,D}
      name::A
      sol::B
      sol_approx::BA
      fem_stats::C
      rb_stats::C
      norm_matrix::D
    end

Allows to compute errors and computational speedups to compare the properties of
the algorithm with the FE performance. In particular:

- `sol`, `sol_approx` are the online FE solution and their RB approximations
- `fem_stats`, `rb_stats` are the ComputationalStats relative to the FE and RB
  algorithms
- `norm_matrix` is the norm matrix with respect to which the errors are computed
  (can also be of type Nothing, in which case a simple ℓ² error measure is used)

"""
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

"""
    compute_speedup(r::RBResults) -> (Number, Number)

Computes the speedup in time and memory, where `speedup` = RB estimate / FE estimate

"""
function compute_speedup(r::RBResults)
  compute_speedup(r.fem_stats,r.rb_stats)
end

function compute_error(
  sol::AbstractSteadySnapshots{T,N},
  sol_approx::AbstractSteadySnapshots{T,N},
  norm_matrix) where {T,N}

  @check size(sol) == size(sol_approx)
  space_norm = zeros(num_params(sol))
  @inbounds for i = num_params(sol)
    soli = selectdim(sol,N,i)
    soli_approx = selectdim(sol_approx,N,i)
    err_norm = _norm(soli-soli_approx,norm_matrix)
    sol_norm = _norm(soli,norm_matrix)
    space_norm[i] = err_norm / sol_norm
  end
  avg_error = sum(space_norm) / length(space_norm)
  return avg_error
end

function compute_error(
  sol::AbstractSteadySnapshots,
  sol_approx::AbstractSteadySnapshots,
  norm_matrix::AbstractTProductArray)

  compute_error(sol,sol_approx,tp_decomposition(norm_matrix))
end

function compute_error(sol::BlockSnapshots,sol_approx::BlockSnapshots,norm_matrix)
  @check get_touched_blocks(sol) == get_touched_blocks(sol_approx)
  active_block_ids = get_touched_blocks(sol)
  block_map = BlockMap(size(sol),active_block_ids)
  errors = [compute_error(sol[i],sol_approx[i],norm_matrix[Block(i,i)]) for i in active_block_ids]
  return_cache(block_map,errors...)
end

"""
    compute_error(r::RBResults) -> Number

Computes the RB/FE error; in transient applications, the measure is a space-time
norm. First, a spatial norm (computed according to the field `norm_matrix`) is
computed at every time step; then, a ℓ² norm of the spatial norms is performed in
time

"""
function compute_error(r::RBResults)
  compute_error(r.sol,r.sol_approx,r.norm_matrix)
end

function average_plot(
  trial::TrialParamFESpace,
  v::AbstractVector;
  name="vel",
  dir=joinpath(pwd(),"plots"))

  trian = get_triangulation(trial)
  vh = FEFunction(param_getindex(trial,1),v)
  vtk = createvtk(trian,dir,cellfields=[name=>vh])
end

function average_plot(
  trial::FESpace,
  sol::AbstractSnapshots,
  sol_approx::AbstractSnapshots;
  dir=joinpath(pwd(),"plots"),
  kwargs...)

  @check size(sol) == size(sol_approx)
  param_mean(a::AbstractArray{T,N}) where {T,N} = dropdims(mean(a;dims=N);dims=N)

  create_dir(dir)

  r = get_realization(sol)
  r̄ = mean(r)
  r₀ = zero(r)

  dir_average = joinpath(dir,"average")
  average_plot(trial(r̄),param_mean(sol);dir=dir_average,kwargs...)

  dir_error = joinpath(dir,"error")
  average_plot(trial(r₀),param_mean(sol - sol_approx);dir=dir_error,kwargs...)
end

function average_plot(trial::FESpace,r::RBResults;kwargs...)
  average_plot(trial,r.sol,r.sol_approx;name=r.name,kwargs...)
end

function average_plot(trial::MultiFieldFESpace,r::RBResults;kwargs...)
  for (i,Ui) in trial
    average_plot(Ui,r.sol[i],r.sol_approx[i];name=r.name[i],kwargs...)
  end
end

"""
    average_plot(op::RBOperator,r::RBResults;kwargs...)
    average_plot(op::TransientRBOperator,r::RBResults;kwargs...)

Computes the plot of the mean snapshot and the plof of the mean error. The mean
is computed along the axis of parameters

"""
function average_plot(op::RBOperator,r::RBResults;kwargs...)
  average_plot(get_fe_trial(op),r;kwargs...)
end
