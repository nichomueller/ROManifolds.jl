function RBSteady.deserialize_operator(feop::TransientParamFEOperatorWithTrian,dir;linearity="lin")
  trian_res = feop.trian_res
  trian_jacs = feop.trian_jacs

  op = RBSteady.deserialize_pg_operator(feop,dir)
  red_rhs = RBSteady.deserialize_contribution(dir,trian_res,get_test(op);label=linearity*"res")
  red_lhs = ()
  for (i,trian_jac) in enumerate(trian_jacs)
    rlhsi = RBSteady.deserialize_contribution(dir,trian_jac,get_trial(op),get_test(op);label=linearity*"jac_$i")
    red_lhs = (red_lhs...,rlhsi)
  end

  trians_rhs = get_domains(red_rhs)
  trians_lhs = map(get_domains,red_lhs)
  new_op = change_triangulation(op,trians_rhs,trians_lhs)
  rbop = TransientPGMDEIMOperator(new_op,red_lhs,red_rhs)
  return rbop
end

function DrWatson.save(dir,op::TransientPGMDEIMOperator;linearity="lin",kwargs...)
  save(dir,op.op;kwargs...)
  for (i,ad_res) in enumerate(op.rhs.values)
    save(dir,ad_res;label=linearity*"res_$i")
  end
  for (ilhs,lhs) in enumerate(op.lhs)
    for (i,ad_jac) in enumerate(lhs.values)
      save(dir,ad_jac;label=linearity*"jac_$(ilhs)_$i")
    end
  end
end

function RBSteady.deserialize_operator(feop::LinearNonlinearTransientParamFEOperatorWithTrian,dir)
  op_lin = deserialize_operator(get_linear_operator(feop),dir;linearity="lin")
  op_nlin = deserialize_operator(get_nonlinear_operator(feop),dir;linearity="nlin")
  rbop = LinearNonlinearTransientPGMDEIMOperator(new_op,red_lhs,red_rhs)
  return rbop
end

function DrWatson.save(dir,op::TransientPGOperator;kwargs...)
  btest = RBSteady.get_basis(get_test(op))
  btrial = RBSteady.get_basis(get_trial(op))
  save(dir,btest;label="test")
  save(dir,btrial;label="trial")
end

function DrWatson.save(dir,op::LinearNonlinearTransientPGMDEIMOperator)
  save(dir,get_linear_operator(op);linearity="lin")
  save(dir,get_nonlinear_operator(op);linearity="nlin")
end

function RBSteady.deserialize_operator(
  feop::LinearNonlinearTransientParamFEOperatorWithTrian,
  rbop::LinearNonlinearTransientPGMDEIMOperator)

  rbop_lin = deserialize_operator(get_linear_operator(feop),get_linear_operator(rbop))
  rbop_nlin = deserialize_operator(get_nonlinear_operator(feop),get_nonlinear_operator(rbop))
  return LinearNonlinearTransientPGMDEIMOperator(rbop_lin,rbop_nlin)
end

function RBSteady.deserialize_pg_operator(feop::TransientParamFEOperatorWithTrian,dir)
  op = get_algebraic_operator(feop)
  fe_test = get_test(feop)
  fe_trial = get_trial(feop)
  basis_test = deserialize(RBSteady.get_projection_filename(dir;label="test"))
  basis_trial = deserialize(RBSteady.get_projection_filename(dir;label="trial"))
  test = fe_subspace(fe_test,basis_test)
  trial = fe_subspace(fe_trial,basis_trial)
  return TransientPGOperator(op,trial,test)
end

function RBSteady.rb_results(solver::RBSolver,op::TransientRBOperator,args...)
  feop = ParamSteady.get_fe_operator(op)
  rb_results(solver,feop,args...)
end

function Utils.compute_relative_error(
  sol::AbstractTransientSnapshots{T,N},
  sol_approx::AbstractTransientSnapshots{T,N},
  args...) where {T,N}

  @check size(sol) == size(sol_approx)
  err_norm = zeros(num_times(sol))
  sol_norm = zeros(num_times(sol))
  errors = zeros(num_params(sol))
  @inbounds for ip = 1:num_params(sol)
    solip = selectdim(sol,N,ip)
    solip_approx = selectdim(sol_approx,N,ip)
    for it in 1:num_times(sol)
      solitp = selectdim(solip,N-1,it)
      solitp_approx = selectdim(solip_approx,N-1,it)
      err_norm[it] = induced_norm(solitp-solitp_approx,args...)
      sol_norm[it] = induced_norm(solitp,args...)
    end
    errors[ip] = norm(err_norm) / norm(sol_norm)
  end
  return mean(errors)
end

function Utils.compute_relative_error(
  sol::TransientMultiValueSnapshots{T,N},
  sol_approx::TransientMultiValueSnapshots{T,N},
  args...) where {T,N}

  @check size(sol) == size(sol_approx)
  err_norm = zeros(num_times(sol))
  sol_norm = zeros(num_times(sol))
  errors = zeros(num_params(sol))
  @inbounds for ip = 1:num_params(sol)
    solip = selectdim(sol,N,ip)
    solip_approx = selectdim(sol_approx,N,ip)
    for it in 1:num_times(sol)
      solitp = selectdim(solip,N-1,it)
      solitp_approx = selectdim(solip_approx,N-1,it)
      fill!(err_norm,0.0)
      fill!(sol_norm,0.0)
      for ic in 1:num_components(sol)
        soliptc = selectdim(solitp,N-2,ic)
        soliptc_approx = selectdim(solitp_approx,N-2,ic)
        err_norm[it] += induced_norm(soliptc-soliptc_approx,args...)^2
        sol_norm[it] += induced_norm(soliptc,args...)^2
      end
      err_norm[it] = sqrt(err_norm[it])
      sol_norm[it] = sqrt(sol_norm[it])
    end
    errors[ip] = norm(err_norm) / norm(sol_norm)
  end
  return mean(errors)
end
