"""
    create_dir(dir::String) -> Nothing

Recursive creation of a directory `dir`

"""
function create_dir(dir::String)
  if !isdir(dir)
    parent_dir, = splitdir(dir)
    create_dir(parent_dir)
    mkdir(dir)
  end
  return
end

"""
    load_solve(solver::RBSolver,feop::ParamFEOperator,dir::String;kwargs...) -> RBResults
    load_solve(solver::RBSolver,feop::TransientParamFEOperator,dir::String;kwargs...) -> RBResults

Loads the snapshots previously saved to file, loads the reduced operator
previously saved to file, and returns the results. This function allows to entirely
skip the RB's offline phase. The field `dir` should be the directory at which the
saved quantities can be found. Note that this function must be used after the test
case has been run at least once!

"""
function load_solve(solver,feop,dir)
  fe_sol = load_snapshots(dir)
  fe_stats = load_stats(dir)
  rbop = load_operator(dir,feop)
  rb_sol,rb_stats,_ = solve(solver,rbop,fe_sol)
  old_results = deserialize(get_results_filename(dir))
  results = rb_results(solver,rbop,fe_sol,rb_sol,rb_stats,fe_stats)
  return results
end

function DrWatson.save(dir,args::Tuple)
  map(a->save(dir,a),args)
end

_get_label(name::String,label) = @abstractmethod
_get_label(name::String,label::Union{Number,Symbol}) = _get_label(name,string(label))
_get_label(name::String,label::String) = name * "_" * label

function _get_label(name,labels...)
  first_lab,last_labs... = labels
  _get_label(name,_get_label(first_lab,last_labs...))
end

function get_filename(dir::String,name::String,labels...;extension=".jld")
  joinpath(dir,_get_label(name,labels...)*extension)
end

function DrWatson.save(dir,s::Union{AbstractSnapshots,BlockSnapshots};label="")
  snaps_dir = get_filename(dir,"snapshots",label)
  serialize(snaps_dir,s)
end

function load_snapshots(dir;label="")
  snaps_dir = get_filename(dir,"snapshots",label)
  deserialize(snaps_dir)
end

function DrWatson.save(dir,stats::NamedTuple;label="")
  stats_dir = get_filename(dir,"stats",label)
  serialize(stats_dir,s)
end

function load_stats(dir;label="")
  stats_dir = get_filename(dir,"stats",label)
  deserialize(stats_dir)
end

function DrWatson.save(dir,b::Projection;label="")
  proj_dir = get_filename(dir,"basis",label)
  serialize(proj_dir,b)
end

function load_projection(dir;label="")
  proj_dir = get_filename(dir,"basis",label)
  deserialize(proj_dir)
end

function DrWatson.save(dir,r::FESubspace;label="")
  save(dir,get_basis(r);label)
end

function load_fe_subspace(dir,f::FESpace;label="")
  basis = load_projection(dir;label)
  fe_subspace(f,basis)
end

function DrWatson.save(dir,hp::HyperReduction;label="")
  ad_dir = get_filename(dir,"hypred",label)
  serialize(ad_dir,hp)
end

function load_decomposition(dir;label="")
  ad_dir = get_filename(dir,"hypred",label)
  deserialize(ad_dir)
end

function DrWatson.save(dir,contrib::AffineContribution;label::String="")
  for (i,c) in enumerate(get_values(contrib))
    save(dir,c;label=_get_label(label,i))
  end
end

function load_contribution(dir,trian,args...;label::String="")
  dec,redt = (),()
  for (i,t) in enumerate(trian)
    deci = load_decomposition(dir;label=_get_label(label,i))
    redti = reduce_triangulation(t,get_integration_domain(adi),args...)
    if isa(redti,AbstractArray)
      redti = ParamDataStructures.merge_triangulations(redti)
    end
    dec = (dec...,deci)
    redt = (redt...,redti)
  end
  return Contribution(dec,redt)
end

function DrWatson.save(dir,op::PGOperator;label="")
  save(dir,get_test(op);label=_get_label(label,"test"))
  save(dir,get_trial(op);label=_get_label(label,"trial"))
end

function load_pg_operator(dir,feop::ParamFEOperatorWithTrian;label="")
  op = get_algebraic_operator(feop)
  test = load_fe_subspace(dir,get_test(feop);label=_get_label(label,"test"))
  trial = load_fe_subspace(dir,get_trial(feop);label=_get_label(label,"trial"))
  return PGOperator(feop,trial,test)
end

function DrWatson.save(dir,op::PGMDEIMOperator;label="")
  save(dir,op.op;label)
  save(dir,op.rhs;label=_get_label(label,"rhs"))
  save(dir,op.lhs;label=_get_label(label,"lhs"))
end

function load_operator(dir,feop::ParamFEOperatorWithTrian;label="")
  trian_res = feop.trian_res
  trian_jac = feop.trian_jac
  pop = load_pg_operator(dir,feop)

  red_rhs = load_contribution(dir,trian_res,get_test(op);label=_get_label(label,"rhs"))
  red_lhs = load_contribution(dir,trian_jac,get_trial(op),get_test(op);label=_get_label(label,"lhs"))
  trians_rhs = get_domains(red_rhs)
  trians_lhs = get_domains(red_lhs)
  new_op = change_triangulation(op,trians_rhs,trians_lhs)
  op = PGMDEIMOperator(new_op,red_lhs,red_rhs)
  return op
end

function DrWatson.save(dir,op::LinearNonlinearPGMDEIMOperator;label="")
  save(dir,get_linear_operator(op);label=_get_label(label,"linear"))
  save(dir,get_nonlinear_operator(op);label=_get_label(label,"nonlinear"))
end

function load_operator(dir,feop::LinearNonlinearParamFEOperatorWithTrian;label="")
  op_lin = load_operator(dir,get_linear_operator(feop);label=_get_label(label,"linear"))
  op_nlin = load_operator(dir,get_nonlinear_operator(feop);label=_get_label(label,"nonlinear"))
  LinearNonlinearParamFEOperatorWithTrian(op_lin,op_nlin)
end

"""
    struct RBResults{E}
      error::E
      speedup::SU
    end

Allows to compute errors and computational speedups to compare the properties of
the algorithm with the FE performance.

"""
struct RBResults{E}
  error::E
  speedup::SU
end

function Base.show(io::IO,k::MIME"text/plain",r::RBResults)
  println(io," ----------------------- RBResults ----------------------------")
  println(io," > error: $(r.error)")
  println(io," > speedup in time: $(r.speedup.speedup_time)")
  println(io," > speedup in memory: $(r.speedup.speedup_memory)")
  println(io," -------------------------------------------------------------")
end

function Base.show(io::IO,r::RBResults)
  show(io,MIME"text/plain"(),r)
end

function rb_results(solver::RBSolver,feop,s,son_approx,fe_stats,rb_stats)
  state_red = get_state_reduction(solver)
  norm_style = NormStyle(state_red)
  son = select_snapshots(s,online_params(solver))
  error = compute_relative_error(norm_style,feop,son,son_approx)
  speedup = compute_speedup(fe_stats,rb_stats)
  RBResults(error,speedup)
end

function rb_results(solver::RBSolver,op::RBOperator,args...)
  feop = ParamSteady.get_fe_operator(op)
  rb_results(solver,feop,args...)
end

function DrWatson.save(dir,r::RBResults;label="")
  results_dir = get_filename(dir,"results",label)
  serialize(results_dir,r)
end

function load_results(dir;label="")
  results_dir = get_filename(dir,"results",label)
  deserialize(results_dir,r)
end

function Utils.compute_relative_error(norm_style::EnergyNorm,feop,son,son_approx)
  X = assemble_matrix(feop,get_norm(norm_style))
  compute_relative_error(son,son_approx,X)
end

function Utils.compute_relative_error(norm_style::EuclideanNorm,feop,son,son_approxs)
  compute_relative_error(son,son_approx)
end

function Utils.compute_relative_error(sol::BlockSnapshots,sol_approx::BlockSnapshots)
  @check get_touched_blocks(sol) == get_touched_blocks(sol_approx)
  active_block_ids = get_touched_blocks(sol)
  block_map = BlockMap(size(sol),active_block_ids)
  errors = [compute_relative_error(sol[i],sol_approx[i]) for i in active_block_ids]
  return_cache(block_map,errors...)
end

function Utils.compute_relative_error(sol::BlockSnapshots,sol_approx::BlockSnapshots,norm_matrix)
  @check get_touched_blocks(sol) == get_touched_blocks(sol_approx)
  active_block_ids = get_touched_blocks(sol)
  block_map = BlockMap(size(sol),active_block_ids)
  errors = [compute_relative_error(sol[i],sol_approx[i],norm_matrix[Block(i,i)]) for i in active_block_ids]
  return_cache(block_map,errors...)
end
