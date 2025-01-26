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

function DrWatson.save(dir,s::AbstractSnapshots;label="")
  snaps_dir = get_filename(dir,"snaps",label)
  serialize(snaps_dir,s)
end

"""
    load_snapshots(dir;label="") -> AbstractSnapshots

Load the snapshots at the directory `dir`. Throws an error if the snapshots
have not been previously saved to file
"""
function load_snapshots(dir;label="")
  snaps_dir = get_filename(dir,"snaps",label)
  deserialize(snaps_dir)
end

function DrWatson.save(dir,stats::PerformanceTracker;label="")
  stats_dir = get_filename(dir,"stats",label)
  serialize(stats_dir,stats)
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

function load_reduced_subspace(dir,f::FESpace;label="")
  basis = load_projection(dir;label)
  reduced_subspace(f,basis)
end

for T in (:HyperReduction,:BlockHyperReduction)
  @eval begin
    function DrWatson.save(dir,hp::$T;label="")
      hr_dir = get_filename(dir,"hypred",label)
      serialize(hr_dir,hp)
    end
  end
end

function load_decomposition(dir;label="")
  hr_dir = get_filename(dir,"hypred",label)
  deserialize(hr_dir)
end

function DrWatson.save(dir,contrib::Contribution;label="")
  for (i,c) in enumerate(get_values(contrib))
    save(dir,c;label=_get_label(label,i))
  end
end

function load_contribution(
  dir,
  trian::Tuple{Vararg{Triangulation}};
  f::Function=load_decomposition,
  label="")

  dec = ()
  for (i,t) in enumerate(trian)
    deci = f(dir;label=_get_label(label,i))
    dec = (dec...,deci)
  end
  return Contribution(dec,trian)
end

function load_contribution(
  dir,
  trian::Tuple{Vararg{Triangulation}},
  args::RBSpace...;
  f::Function=load_decomposition,
  label="")

  dec,redt = (),()
  for (i,t) in enumerate(trian)
    deci = f(dir;label=_get_label(label,i))
    redti = reduced_triangulation(t,deci,args...)
    if isa(redti,AbstractArray)
      redti = Utils.merge_triangulations(redti)
    end
    dec = (dec...,deci)
    redt = (redt...,redti)
  end
  return Contribution(dec,redt)
end

function DrWatson.save(dir,contrib::ArrayContribution,::SplitParamFEOperator;label="res")
  save(dir,contrib;label)
end

function DrWatson.save(dir,contrib::TupOfArrayContribution,feop::LinearNonlinearParamFEOperator;label="res")
  @check length(contrib) == 2
  save(dir,first(contrib),get_linear_operator(feop);label=_get_label(label,"lin"))
  save(dir,last(contrib),get_nonlinear_operator(feop);label=_get_label(label,"nlin"))
end

function load_residuals(dir,feop::SplitParamFEOperator;label="res")
  load_contribution(dir,get_domains_res(feop);load_snapshots,label)
end

function load_jacobians(dir,feop::SplitParamFEOperator;label="jac")
  load_contribution(dir,get_domains_jac(feop);load_snapshots,label)
end

function load_residuals(dir,feop::LinearNonlinearParamFEOperator;label="res")
  res_lin = load_residuals(dir,get_linear_operator(feop);label=_get_label(label,"lin"))
  res_nlin = load_residuals(dir,get_nonlinear_operator(feop);label=_get_label(label,"nlin"))
  return (res_lin,res_nlin)
end

function load_jacobians(dir,feop::LinearNonlinearParamFEOperator;label="jac")
  jac_lin = load_jacobians(dir,get_linear_operator(feop);label=_get_label(label,"lin"))
  jac_nlin = load_jacobians(dir,get_nonlinear_operator(feop);label=_get_label(label,"nlin"))
  return (jac_lin,jac_nlin)
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
  test = load_reduced_subspace(dir,get_test(feop);label=_get_label(label,"test"))
  trial = load_reduced_subspace(dir,get_trial(feop);label=_get_label(label,"trial"))
  return trial,test
end

function _load_trian_operator_parts(dir,feop::SplitParamFEOperator,trial,test;label="")
  trian_res = get_domains_res(feop)
  trian_jac = get_domains_jac(feop)
  pop = get_algebraic_operator(feop)
  red_rhs = load_contribution(dir,trian_res,test;label=_get_label(label,"rhs"))
  red_lhs = load_contribution(dir,trian_jac,trial,test;label=_get_label(label,"lhs"))
  trians_rhs = get_domains(red_rhs)
  trians_lhs = get_domains(red_lhs)
  new_pop = change_domains(pop,trians_rhs,trians_lhs)
  return new_pop,red_lhs,red_rhs
end

"""
    load_operator(dir,feop::SplitParamFEOperator;kwargs...) -> RBOperator
    load_operator(dir,feop::SplitTransientParamFEOperator;kwargs...) -> TransientRBOperator

Given a FE operator `feop`, load its reduced counterpart stored in the
directory `dir`. Throws an error if the reduced operator has not been previously
saved to file
"""
function load_operator(dir,feop::SplitParamFEOperator;kwargs...)
  trial,test = _load_fixed_operator_parts(dir,feop;kwargs...)
  pop,red_lhs,red_rhs = _load_trian_operator_parts(dir,feop,trial,test;kwargs...)
  op = GenericRBOperator(pop,trial,test,red_lhs,red_rhs)
  return op
end

function DrWatson.save(dir,op::LinearNonlinearRBOperator;label="")
  _save_fixed_operator_parts(dir,op.op_linear;label)
  _save_trian_operator_parts(dir,op.op_linear;label=_get_label(label,"lin"))
  _save_trian_operator_parts(dir,op.op_nonlinear;label=_get_label(label,"nlin"))
end

function load_operator(dir,feop::LinearNonlinearParamFEOperator{SplitDomains};label="")
  trial,test = _fixed_operator_parts(dir,feop.op_linear;label)
  pop_lin,red_lhs_lin,red_rhs_lin = _load_trian_operator_parts(
    dir,feop.op_linear,trial,test;label=_get_label("lin",label))
  pop_nlin,red_lhs_nlin,red_rhs_nlin = _load_trian_operator_parts(
    dir,feop.op_nonlinear,trial,test;label=_get_label("nlin",label))
  op_lin = GenericRBOperator(pop_lin,trial,test,red_lhs_lin,red_rhs_lin)
  op_nlin = GenericRBOperator(pop_nlin,trial,test,red_lhs_nlin,red_rhs_nlin)
  return LinearNonlinearRBOperator(op_lin,op_nlin)
end

"""
    struct ROMPerformance
      error
      speedup
    end

Allows to compute errors and computational speedups to compare the properties of
the algorithm with the FE performance.
"""
struct ROMPerformance
  error
  speedup
end

function Base.show(io::IO,k::MIME"text/plain",perf::ROMPerformance)
  println(io," ----------------------- ROMPerformance ----------------------------")
  println(io," > error: $(perf.error)")
  println(io," > speedup in time: $(perf.speedup.speedup_time)")
  println(io," > speedup in memory: $(perf.speedup.speedup_memory)")
  println(io," -------------------------------------------------------------")
end

function Base.show(io::IO,perf::ROMPerformance)
  show(io,MIME"text/plain"(),perf)
end

"""
    eval_performance(
      solver::RBSolver,
      feop::ParamFEOperator,
      fesnaps::AbstractSnapshots,
      rbsnaps::AbstractSnapshots,
      festats::CostTracker,
      rbstats::CostTracker,
      args...
      ) -> ROMPerformance

Arguments:
  - `solver`: solver for the reduced problem
  - `feop`: FE operator representing the PDE
  - `fesnaps`: online snapshots of the FE solution
  - `rbsnaps`: reduced approximation of `fesnaps`
  - `festats`: time and memory consumption needed to compute `fesnaps`
  - `rbstats`: time and memory consumption needed to compute `rbsnaps`

Returns the performance of the reduced algorithm, in terms of the (relative) error
between `rbsnaps` and `fesnaps`, and the computational speedup between `rbstats`
and `festats`
"""
function eval_performance(
  solver::RBSolver,
  feop::ParamFEOperator,
  fesnaps::AbstractSnapshots,
  rbsnaps::AbstractSnapshots,
  festats::CostTracker,
  rbstats::CostTracker,
  args...)

  state_red = get_state_reduction(solver)
  norm_style = NormStyle(state_red)
  error = compute_relative_error(norm_style,feop,fesnaps,rbsnaps,args...)
  speedup = compute_speedup(festats,rbstats)
  ROMPerformance(error,speedup)
end

function eval_performance(
  solver::RBSolver,
  feop::ParamFEOperator,
  rbop::ParamOperator,
  fesnaps::AbstractSnapshots,
  x̂::AbstractParamVector,
  festats::CostTracker,
  rbstats::CostTracker,
  r::AbstractRealization,
  args...)

  rbsnaps = to_snapshots(get_trial(rbop),x̂,r)
  eval_performance(solver,feop,fesnaps,rbsnaps,festats,rbstats,args...)
end

function DrWatson.save(dir,perf::ROMPerformance;label="")
  results_dir = get_filename(dir,"results",label)
  serialize(results_dir,perf)
end

"""
    load_results(dir;label="") -> ROMPerformance

Load the results at the directory `dir`. Throws an error if the results
have not been previously saved to file
"""
function load_results(dir;label="")
  results_dir = get_filename(dir,"results",label)
  deserialize(results_dir,perf)
end

function Utils.compute_relative_error(norm_style::NormStyle,feop,sol,sol_approx,trian::Triangulation)
  sol_trian = change_domain(sol,trian)
  sol_approx_trian = change_domain(sol_approx,trian)
  compute_relative_error(norm_style,feop,sol_trian,sol_approx_trian)
end

function Utils.compute_relative_error(norm_style::EnergyNorm,feop,sol,sol_approx)
  X = assemble_matrix(feop,get_norm(norm_style))
  compute_relative_error(sol,sol_approx,X)
end

function Utils.compute_relative_error(norm_style::EuclideanNorm,feop,sol,sol_approx)
  compute_relative_error(sol,sol_approx)
end

function Utils.compute_relative_error(
  sol::SteadySnapshots{T,N},
  sol_approx::SteadySnapshots{T,N},
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
  T = eltype2(sol)
  error = Array{T,ndims(sol)}(undef,size(sol))
  for i in eachindex(sol)
    if sol.touched[i]
      error[i] = compute_relative_error(sol[i],sol_approx[i])
    end
  end
  error
end

function Utils.compute_relative_error(
  sol::BlockSnapshots,
  sol_approx::BlockSnapshots,
  X::MatrixOrTensor)

  @check sol.touched == sol_approx.touched
  error = zeros(size(sol))
  for i in eachindex(sol)
    if sol.touched[i]
      error[i] = compute_relative_error(sol[i],sol_approx[i],X[Block(i,i)])
    end
  end
  error
end

function plot_a_solution(dir,Ω,uh,ûh,r::Realization)
  uh1 = param_getindex(uh,1)
  ûh1 = param_getindex(ûh,1)
  eh1 = uh1 - ûh1
  writevtk(Ω,dir*".vtu",cellfields=["uh"=>uh1,"ûh"=>ûh1,"eh"=>eh1])
end

"""
    plot_a_solution(
      dir,
      feop::ParamFEOperator,
      sol::Snapshots,
      sol_approx::Snapshots,
      args...
      ) -> Nothing

Plots a single FE solution, RB solution, and the point-wise error between the two,
by selecting the first FE snapshot in `sol` and the first reduced snapshot in `sol_approx`
"""
function plot_a_solution(
  dir,
  feop::ParamFEOperator,
  sol::Snapshots,
  sol_approx::Snapshots,
  field=1)

  r = get_realization(sol)

  trial = get_trial(feop)(r)
  U = isa(trial,MultiFieldFESpace) ? trial[field] : trial
  Ω = get_triangulation(U)

  uh = FEFunction(U,get_values(sol))
  ûh = FEFunction(U,get_values(sol_approx))
  dirfield = joinpath(dir,"var$field")

  plot_a_solution(dirfield,Ω,uh,ûh,r)
end

function plot_a_solution(
  dir,
  feop::ParamFEOperator,
  sol::BlockSnapshots,
  sol_approx::BlockSnapshots,
  args...)

  @check sol.touched == sol_approx.touched
  for i in eachindex(sol)
    if sol.touched[i]
      plot_a_solution(dir,feop,sol[i],sol_approx[i],i)
    end
  end
end

function plot_a_solution(
  dir,
  feop::ParamFEOperator,
  rbop::ParamOperator,
  fesnaps::AbstractSnapshots,
  x̂::AbstractParamVector,
  r::AbstractRealization)

  rbsnaps = to_snapshots(get_trial(rbop),x̂,r)
  plot_a_solution(dir,feop,fesnaps,rbsnaps)
end
