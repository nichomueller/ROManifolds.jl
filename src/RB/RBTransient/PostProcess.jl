function DrWatson.save(dir,op::GenericTransientRBOperator;label="")
  save(dir,get_test(op);label=RBSteady._get_label(label,"test"))
  save(dir,get_trial(op);label=RBSteady._get_label(label,"trial"))
  save(dir,op.rhs;label=RBSteady._get_label(label,"rhs"))
  save(dir,op.lhs;label=RBSteady._get_label(label,"lhs"))
end

function RBSteady.load_operator(dir,feop::TransientParamFEOperatorWithTrian;label="")
  trian_res = feop.trian_res
  trian_jacs = feop.trian_jacs
  odeop = get_algebraic_operator(feop)

  test = RBSteady.load_fe_subspace(dir,get_test(feop);label=RBSteady._get_label(label,"test"))
  trial = RBSteady.load_fe_subspace(dir,get_trial(feop);label=RBSteady._get_label(label,"trial"))
  red_rhs = RBSteady.load_contribution(dir,trian_res,get_test(op);label=_get_label(label,"rhs"))
  red_lhs = ()
  for (i,trian_jac) in enumerate(trian_jacs)
    rlhsi = RBSteady.load_contribution(dir,trian_jac,get_trial(op),get_test(op);label=_get_label(label,"lhs",i))
    red_lhs = (red_lhs...,rlhsi)
  end
  trians_rhs = get_domains(red_rhs)
  trians_lhs = map(get_domains,red_lhs)
  new_odeop = change_triangulation(odeop,trians_rhs,trians_lhs)
  op = GenericTransientRBOperator(new_odeop,trial,test,red_lhs,red_rhs)
  return op
end

function DrWatson.save(dir,op::LinearNonlinearTransientRBOperator;label="")
  save(dir,get_linear_operator(op);label=RBSteady._get_label(label,"linear"))
  save(dir,get_nonlinear_operator(op);label=RBSteady._get_label(label,"nonlinear"))
end

function RBSteady.load_operator(dir,feop::LinearNonlinearTransientParamFEOperatorWithTrian;label="")
  op_lin = load_operator(dir,get_linear_operator(feop);label=RBSteady._get_label(label,"linear"))
  op_nlin = load_operator(dir,get_nonlinear_operator(feop);label=RBSteady._get_label(label,"nonlinear"))
  LinearNonlinearParamFEOperatorWithTrian(op_lin,op_nlin)
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
