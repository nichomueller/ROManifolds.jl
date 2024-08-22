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
  fe_sol = deserialize(get_snapshots_filename(dir))
  rbop = deserialize_operator(feop,dir)
  rb_sol,_ = solve(solver,rbop,fe_sol)
  old_results = deserialize(get_results_filename(dir))
  timer = get_timer(solver)
  merge!(timer,old_results.timer)
  results = rb_results(solver,rbop,fe_sol,rb_sol;kwargs...)
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
    struct RBResults
      name::Union{String,Vector{String}}
      timer::TimerOutput
      error::Vector{Float64}
    end

Allows to compute errors and computational speedups to compare the properties of
the algorithm with the FE performance.

"""
struct RBResults
  name::Union{String,Vector{String}}
  timer::TimerOutput
  error::Vector{Float64}
end

TimerOutputs.get_timer(r::RBResults) = r.timer

function rb_results(solver::RBSolver,feop,s,son_approx;name="vel")
  timer = get_timer(solver)
  X = assemble_norm_matrix(feop)
  son = select_snapshots(s,online_params(solver))
  error = compute_error(son,son_approx,X)
  RBResults(name,timer,error)
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

function Utils.compute_error(sol::BlockSnapshots,sol_approx::BlockSnapshots,norm_matrix)
  @check get_touched_blocks(sol) == get_touched_blocks(sol_approx)
  active_block_ids = get_touched_blocks(sol)
  block_map = BlockMap(size(sol),active_block_ids)
  errors = [compute_error(sol[i],sol_approx[i],norm_matrix[Block(i,i)]) for i in active_block_ids]
  return_cache(block_map,errors...)
end

function Utils.compute_error(r::RBResults)
  mean(r.error)
end
