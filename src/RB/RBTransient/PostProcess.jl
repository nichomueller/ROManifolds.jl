function DrWatson.save(dir,op::TransientRBOperator)
  serialize(dir * "/operator.jld",op)
end

function RBSteady.deserialize_operator(feop::TransientParamFEOperatorWithTrian,rbop::TransientPGMDEIMOperator)
  trian_res = feop.trian_res
  trian_jacs = feop.trian_jacs
  rhs_old,lhs_old = rbop.rhs,rbop.lhs

  trian_lhs_new = ()
  lhs_new = ()
  for i = eachindex(lhs_old)
    lhs_old_i = lhs_old[i]
    trian_jac_i = trian_jacs[i]
    iperm_jac,trian_jac_new = map(t->find_closest_view(trian_jac_i,t),lhs_old_i.trians) |> tuple_of_arrays
    value_jac_new = map(i -> getindex(lhs_old_i.values,i...),iperm_jac)
    trian_lhs_new = (trian_lhs_new...,trian_jac_new)
    lhs_new = (lhs_new...,Contribution(value_jac_new,trian_jac_new))
  end

  iperm_res,trian_res_new = map(t->find_closest_view(trian_res,t),rhs_old.trians) |> tuple_of_arrays
  value_res_new = map(i -> getindex(rhs_old.values,i...),iperm_res)
  rhs_new = Contribution(value_res_new,trian_res_new)

  pop_new =  RBSteady.change_operator(feop,rbop.op)
  op_new = change_triangulation(pop_new,trian_res_new,trian_lhs_new;approximate=true)

  return TransientPGMDEIMOperator(op_new,lhs_new,rhs_new)
end

function RBSteady.deserialize_operator(
  feop::LinearNonlinearTransientParamFEOperatorWithTrian,
  rbop::LinearNonlinearTransientPGMDEIMOperator)

  rbop_lin = deserialize_operator(get_linear_operator(feop),get_linear_operator(rbop))
  rbop_nlin = deserialize_operator(get_nonlinear_operator(feop),get_nonlinear_operator(rbop))
  return LinearNonlinearTransientPGMDEIMOperator(rbop_lin,rbop_nlin)
end

function RBSteady.change_operator(feop::TransientParamFEOperatorWithTrian,pop::TransientPGOperator)
  op = get_algebraic_operator(feop)
  test = get_test(feop)
  trial = get_trial(feop)
  basis_test = RBSteady.get_basis(get_test(pop))
  basis_trial = RBSteady.get_basis(get_trial(pop))
  test′ = fe_subspace(test,basis_test)
  trial′ = fe_subspace(trial,basis_trial)
  return TransientPGOperator(op,trial′,test′)
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

function RBSteady.average_plot(
  trial::TrialParamFESpace,
  mat::AbstractMatrix;
  name="vel",
  dir=joinpath(pwd(),"plots"))

  RBSteady.create_dir(dir)
  trian = get_triangulation(trial)
  createpvd(dir) do pvd
    for i in axes(mat,2)
      solh_i = FEFunction(param_getindex(trial,i),mat[:,i])
      vtk = createvtk(trian,dir,cellfields=[name=>solh_i])
      pvd[i] = vtk
    end
  end
end

function RBSteady.average_plot(op::TransientRBOperator,r::RBResults;kwargs...)
  average_plot(get_fe_trial(op),r;kwargs...)
end
