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
    load_solve(solver::RBSolver,feop::ParamFEOperator,dir::String;kwargs...) -> RBPerformance
    load_solve(solver::RBSolver,feop::TransientParamFEOperator,dir::String;kwargs...) -> RBPerformance

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
  results = rb_performance(solver,rbop,fe_sol,rb_sol,rb_stats,fe_stats)
  return results
end

function DrWatson.save(dir,args::Tuple)
  map(a->save(dir,a),args)
end

_get_label(name::String,label) = @abstractmethod
_get_label(name::String,label::Union{Number,Symbol}) = _get_label(name,string(label))
function _get_label(name::String,label::String)
  if label==""
    name
  else
    if name==""
      label
    else
      name * "_" * label
    end
  end
end

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

function DrWatson.save(dir,r::RBSpace;label="")
  save(dir,get_reduced_subspace(r);label)
end

function load_fe_subspace(dir,f::FESpace;label="")
  basis = load_projection(dir;label)
  fe_subspace(f,basis)
end

function DrWatson.save(dir,hp::HyperReduction;label="")
  hr_dir = get_filename(dir,"hypred",label)
  serialize(hr_dir,hp)
end

function load_decomposition(dir;label="")
  hr_dir = get_filename(dir,"hypred",label)
  deserialize(hr_dir)
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
    redti = reduced_triangulation(t,deci,args...)
    if isa(redti,AbstractArray)
      redti = ParamDataStructures.merge_triangulations(redti)
    end
    dec = (dec...,deci)
    redt = (redt...,redti)
  end
  return Contribution(dec,redt)
end

function _save_fixed_operator_parts(dir,op;label="")
  save(dir,get_test(op);label=_get_label(label,"test"))
  save(dir,get_trial(op);label=_get_label(label,"trial"))
end

function _save_trian_operator_parts(dir,op::GenericRBOperator;label="")
  save(dir,op.rhs;label=_get_label(label,"rhs"))
  save(dir,op.lhs;label=_get_label(label,"lhs"))
end

function DrWatson.save(dir,op::GenericRBOperator;kwargs...)
  _save_fixed_operator_parts(dir,op;kwargs...)
  _save_trian_operator_parts(dir,op;kwargs...)
end

function _load_fixed_operator_parts(dir,feop;label="")
  test = load_fe_subspace(dir,get_test(feop);label=_get_label(label,"test"))
  trial = load_fe_subspace(dir,get_trial(feop);label=_get_label(label,"trial"))
  return trial,test
end

function _load_trian_operator_parts(dir,feop::ParamFEOperatorWithTrian,trial,test;label="")
  trian_res = feop.trian_res
  trian_jac = feop.trian_jac
  pop = get_algebraic_operator(feop)
  red_rhs = load_contribution(dir,trian_res,test;label=_get_label(label,"rhs"))
  red_lhs = load_contribution(dir,trian_jac,trial,test;label=_get_label(label,"lhs"))
  trians_rhs = get_domains(red_rhs)
  trians_lhs = get_domains(red_lhs)
  new_pop = change_triangulation(pop,trians_rhs,trians_lhs)
  return new_pop,red_lhs,red_rhs
end

function load_operator(dir,feop::ParamFEOperatorWithTrian;kwargs...)
  trial,test = _load_fixed_operator_parts(dir,feop;kwargs...)
  pop,red_lhs,red_rhs = _load_trian_operator_parts(dir,feop,trial,test;kwargs...)
  op = GenericRBOperator(pop,trial,test,red_lhs,red_rhs)
  return op
end

function DrWatson.save(dir,op::LinearNonlinearRBOperator;label="")
  _save_fixed_operator_parts(dir,op.op_linear;label)
  _save_trian_operator_parts(dir,op.op_linear;label=_get_label(label,"linear"))
  _save_trian_operator_parts(dir,op.op_nonlinear;label=_get_label(label,"nonlinear"))
end

function load_operator(dir,feop::LinearNonlinearParamFEOperator;label="")
  @assert isa(feop.op_linear,ParamFEOperatorWithTrian)
  @assert isa(feop.op_nonlinear,ParamFEOperatorWithTrian)

  trial,test = _fixed_operator_parts(dir,feop.op_linear;label)
  pop_lin,red_lhs_lin,red_rhs_lin = _load_trian_operator_parts(
    dir,feop.op_linear,trial,test;label=_get_label("linear",label))
  pop_nlin,red_lhs_nlin,red_rhs_nlin = _load_trian_operator_parts(
    dir,feop.op_nonlinear,trial,test;label=_get_label("nonlinear",label))
  op_lin = GenericRBOperator(pop_lin,trial,test,red_lhs_lin,red_rhs_lin)
  op_nlin = GenericRBOperator(pop_nlin,trial,test,red_lhs_nlin,red_rhs_nlin)
  return LinearNonlinearRBOperator(op_lin,op_nlin)
end

"""
    struct RBPerformance{E}
      error::E
      speedup::SU
    end

Allows to compute errors and computational speedups to compare the properties of
the algorithm with the FE performance.

"""
struct RBPerformance
  error
  speedup
end

function Base.show(io::IO,k::MIME"text/plain",perf::RBPerformance)
  println(io," ----------------------- RBPerformance ----------------------------")
  println(io," > error: $(perf.error)")
  println(io," > speedup in time: $(perf.speedup.speedup_time)")
  println(io," > speedup in memory: $(perf.speedup.speedup_memory)")
  println(io," -------------------------------------------------------------")
end

function Base.show(io::IO,perf::RBPerformance)
  show(io,MIME"text/plain"(),perf)
end

function rb_performance(
  solver::RBSolver,
  feop,
  fesnaps::AbstractArray,
  rbsnaps::AbstractArray,
  festats::CostTracker,
  rbstats::CostTracker)

  state_red = get_state_reduction(solver)
  norm_style = NormStyle(state_red)
  error = compute_relative_error(norm_style,feop,fesnaps,rbsnaps)
  speedup = compute_speedup(festats,rbstats)
  RBPerformance(error,speedup)
end

function rb_performance(
  solver::RBSolver,
  feop,
  rbop,
  fesnaps::AbstractArray,
  x̂::AbstractParamVector,
  festats::CostTracker,
  rbstats::CostTracker,
  r::AbstractRealization)

  x = inv_project(get_trial(rbop)(r),x̂)
  rbsnaps = Snapshots(x,get_vector_index_map(feop),r)
  rb_performance(solver,feop,fesnaps,rbsnaps,festats,rbstats)
end

function DrWatson.save(dir,perf::RBPerformance;label="")
  results_dir = get_filename(dir,"results",label)
  serialize(results_dir,perf)
end

function load_results(dir;label="")
  results_dir = get_filename(dir,"results",label)
  deserialize(results_dir,perf)
end

function Utils.compute_relative_error(norm_style::EnergyNorm,feop,sol,sol_approx)
  X = assemble_matrix(feop,get_norm(norm_style))
  compute_relative_error(sol,sol_approx,X)
end

function Utils.compute_relative_error(norm_style::EuclideanNorm,feop,sol,sol_approx)
  compute_relative_error(sol,sol_approx)
end

function Utils.compute_relative_error(
  sol::AbstractSteadySnapshots{T,N},
  sol_approx::AbstractSteadySnapshots{T,N},
  args...) where {T,N}

  @check size(sol) == size(sol_approx)
  errors = zeros(num_params(sol))
  @inbounds for ip = 1:num_params(sol)
    solip = selectdim(sol,N,ip)
    solip_approx = selectdim(sol_approx,N,ip)
    err_norm = induced_norm(solip-solip_approx,args...)
    sol_norm = induced_norm(solip,args...)
    errors[ip] = err_norm / sol_norm
  end
  return mean(errors)
end

function Utils.compute_relative_error(sol::BlockSnapshots,sol_approx::BlockSnapshots)
  @check sol.touched == sol_approx.touched
  error = Array{Float64,ndims(sol)}(undef,size(sol))
  for i in eachindex(sol)
    if sol.touched[i]
      error[i] = compute_relative_error(sol[i],sol_approx[i])
    end
  end
  error
end

function Utils.compute_relative_error(sol::BlockSnapshots,sol_approx::BlockSnapshots,norm_matrix)
  @check sol.touched == sol_approx.touched
  error = Array{Float64,ndims(sol)}(undef,size(sol))
  for i in eachindex(sol)
    if sol.touched[i]
      error[i] = compute_relative_error(sol[i],sol_approx[i],norm_matrix[Block(i,i)])
    end
  end
  error
end
