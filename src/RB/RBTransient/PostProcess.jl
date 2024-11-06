function RBSteady._save_trian_operator_parts(dir,op::GenericTransientRBOperator;label="")
  save(dir,op.rhs;label=RBSteady._get_label(label,"rhs"))
  for (i,lhsi) in enumerate(op.lhs)
    save(dir,lhsi;label=RBSteady._get_label(label,"lhs_$i"))
  end
end

function DrWatson.save(dir,op::GenericTransientRBOperator;kwargs...)
  RBSteady._save_fixed_operator_parts(dir,op;kwargs...)
  RBSteady._save_trian_operator_parts(dir,op;kwargs...)
end

function RBSteady._load_trian_operator_parts(dir,feop::TransientParamFEOperatorWithTrian,trial,test;label="")
  trian_res = feop.trian_res
  trian_jacs = feop.trian_jacs
  odeop = get_algebraic_operator(feop)
  red_rhs = RBSteady.load_contribution(dir,trian_res,test;label=RBSteady._get_label(label,"rhs"))
  red_lhs = ()
  for (i,trian_jac) in enumerate(trian_jacs)
    rlhsi = RBSteady.load_contribution(dir,trian_jac,trial,test;label=RBSteady._get_label(label,"lhs",i))
    red_lhs = (red_lhs...,rlhsi)
  end
  trians_rhs = get_domains(red_rhs)
  trians_lhs = map(get_domains,red_lhs)
  new_odeop = change_domains(odeop,trians_rhs,trians_lhs)
  return new_odeop,red_lhs,red_rhs
end

function RBSteady.load_operator(dir,feop::TransientParamFEOperatorWithTrian;kwargs...)
  trial,test = RBSteady._load_fixed_operator_parts(dir,feop;kwargs...)
  odeop,red_lhs,red_rhs = RBSteady._load_trian_operator_parts(dir,feop,trial,test;kwargs...)
  op = GenericTransientRBOperator(odeop,trial,test,red_lhs,red_rhs)
  return op
end

function DrWatson.save(dir,op::LinearNonlinearTransientRBOperator;label="")
  RBSteady._save_fixed_operator_parts(dir,op.op_linear;label)
  RBSteady._save_trian_operator_parts(dir,op.op_linear;label=RBSteady._get_label(label,"linear"))
  RBSteady._save_trian_operator_parts(dir,op.op_nonlinear;label=RBSteady._get_label(label,"nonlinear"))
end

function RBSteady.load_operator(dir,feop::LinearNonlinearTransientParamFEOperator;label="")
  @assert isa(feop.op_linear,TransientParamFEOperatorWithTrian)
  @assert isa(feop.op_nonlinear,TransientParamFEOperatorWithTrian)

  trial,test = RBSteady._load_fixed_operator_parts(dir,feop.op_linear;label)
  odeop_lin,red_lhs_lin,red_rhs_lin = RBSteady._load_trian_operator_parts(
    dir,feop.op_linear,trial,test;label=RBSteady._get_label("linear",label))
  odeop_nlin,red_lhs_nlin,red_rhs_nlin = RBSteady._load_trian_operator_parts(
    dir,feop.op_nonlinear,trial,test;label=RBSteady._get_label("nonlinear",label))
  op_lin = GenericTransientRBOperator(odeop_lin,trial,test,red_lhs_lin,red_rhs_lin)
  op_nlin = GenericTransientRBOperator(odeop_nlin,trial,test,red_lhs_nlin,red_rhs_nlin)
  return LinearNonlinearTransientRBOperator(op_lin,op_nlin)
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
