function DrWatson.save(dir,op::TransientRBOperator)
  serialize(dir * "/operator.jld",op)
end

function RBSteady.rb_results(solver::RBSolver,op::TransientRBOperator,args...;kwargs...)
  feop = ParamSteady.get_fe_operator(op)
  rb_results(solver,feop,args...;kwargs...)
end

function RBSteady.compute_error(sol::ModeTransientSnapshots,sol_approx::ModeTransientSnapshots,norm_matrix=nothing)
  err_norm = zeros(num_times(sol))
  sol_norm = zeros(num_times(sol))
  space_time_norm = zeros(num_params(sol))
  @inbounds for i = axes(sol,2)
    it = fast_index(i,num_times(sol))
    ip = slow_index(i,num_times(sol))
    err_norm[it] = RBSteady._norm(sol[:,i]-sol_approx[:,i],norm_matrix)
    sol_norm[it] = RBSteady._norm(sol[:,i],norm_matrix)
    if mod(i,num_params(sol)) == 0
      space_time_norm[ip] = norm(err_norm) / norm(sol_norm)
    end
  end
  avg_error = sum(space_time_norm) / length(space_time_norm)
  return avg_error
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
