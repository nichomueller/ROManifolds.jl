function compose_functionals(p::ParamFEProblem)
  compose_bilinear_functionals(p),compose_linear_functionals(p)
end

function compose_linear_functionals(p::ParamFEProblem{A,D,false,S}) where {A,D,S}
  linear_form_idx = findall(x->x==0,is_bilinear(p))
  Broadcasting(compose_functionals)(p.param_fe_functional[linear_form_idx])
end

function compose_linear_functionals(p::ParamFEProblem{A,D,true,S}) where {A,D,S}
  all_test = get_all_test(problem)
  test = get_test(problem)
  test_v_idx = findall(x->x==test[1],all_test)
  test_q_idx = setdiff(eachindex(all_test),test_v_idx)
  @assert length(test) == 2

  (Broadcasting(compose_functionals)(p.param_fe_functional[test_v_idx]),
   Broadcasting(compose_functionals)(p.param_fe_functional[test_q_idx]))
end

function compose_linear_functionals(::ParamFEProblem)
  error("Not implemented")
end

function compose_bilinear_functionals(p::ParamFEProblem{A,D,false,true}) where {A,D}
  bilinear_form_idx = findall(x->x!=0,is_bilinear(p))
  Broadcasting(compose_functionals)(p.param_fe_functional[bilinear_form_idx])
end

function compose_bilinear_functionals(p::ParamFEProblem{A,D,true,true}) where {A,D}
  all_trial = get_all_trial(problem)
  trial = get_trial(problem)
  @assert length(trial) == 2

  trial_uv_idx = findall(x->x==trial[1],all_trial)
  trial_uq_idx = setdiff(eachindex(all_trial),trial_uv_idx)

  (Broadcasting(compose_functionals)(p.param_fe_functional[trial_uv_idx]),
   Broadcasting(compose_functionals)(p.param_fe_functional[trial_uq_idx]))
end

function compose_bilinear_functionals(p::ParamFEProblem{Nonlinear,D,true,true}) where D
  all_trial = get_all_trial(problem)
  trial = get_trial(problem)
  @assert length(trial) == 2

  nonlinear_idx = findall(x->x==false,Broadcasting(islinear)(p.param_fe_functional))

  trial_uv_idx = findall(x->x==trial[1],all_trial)
  trial_uq_idx = setdiff(eachindex(all_trial),trial_uv_idx)

  trial_uv_idx_lin = setdiff(trial_uv_idx,nonlinear_idx)
  trial_uq_idx_lin = setdiff(trial_uq_idx,nonlinear_idx)
  trial_uv_idx_nonlin = intersect(trial_uv_idx,nonlinear_idx)
  trial_uq_idx_nonlin = intersect(trial_uq_idx,nonlinear_idx)
  @assert isempty(trial_uq_idx_nonlin) "Not implemented"

  (Broadcasting(compose_functionals)(p.param_fe_functional[trial_uv_idx_nonlin]),
   Broadcasting(compose_functionals)(p.param_fe_functional[trial_uv_idx_lin]),
   Broadcasting(compose_functionals)(p.param_fe_functional[trial_uq_idx_lin]))
end

function compose_bilinear_functionals(p::ParamFEProblem{A,D,false,false}) where {A,D}
  bilinear_form_idx = findall(x->x!=0,is_bilinear(p))
  time_dependent_form_idx = first(findall(x->x==:M,get_id(p))) #ugly

  (compose_functionals(p.param_fe_functional[time_dependent_form_idx]),
   Broadcasting(compose_functionals)(p.param_fe_functional[bilinear_form_idx]))
end

function compose_bilinear_functionals(p::ParamFEProblem{A,D,false,false}) where {A,D}
  time_dependent_form_idx = first(findall(x->x==:M,get_id(p))) #ugly

  all_trial = get_all_trial(problem)
  trial = get_trial(problem)
  @assert length(trial) == 2 "Not implemented"

  trial_uv_idx = setdiff(findall(x->x==trial[1],all_trial),time_dependent_form_idx)
  trial_uq_idx = setdiff(eachindex(all_trial),trial_uv_idx)

  (compose_functionals(p.param_fe_functional[time_dependent_form_idx]),
   Broadcasting(compose_functionals)(p.param_fe_functional[trial_uv_idx]),
   Broadcasting(compose_functionals)(p.param_fe_functional[trial_uq_idx]))
end

function compose_bilinear_functionals(p::ParamFEProblem{Nonlinear,D,true,true}) where D
  time_dependent_form_idx = first(findall(x->x==:M,get_id(p))) #ugly

  all_trial = get_all_trial(problem)
  trial = get_trial(problem)
  @assert length(trial) == 2

  nonlinear_idx = findall(x->x==false,Broadcasting(islinear)(p.param_fe_functional))
  @assert length(nonlinear_idx) == 2 "Not implemented"
  @assert isempty(intersect(time_dependent_form_idx,nonlinear_idx)) "Not implemented"

  trial_uv_idx = setdiff(findall(x->x==trial[1],all_trial),time_dependent_form_idx)
  trial_uq_idx = setdiff(eachindex(all_trial),trial_uv_idx)

  trial_uv_idx_lin = setdiff(trial_uv_idx,nonlinear_idx)
  trial_uq_idx_lin = setdiff(trial_uq_idx,nonlinear_idx)
  trial_uv_idx_nonlin = intersect(trial_uv_idx,nonlinear_idx)
  trial_uq_idx_nonlin = intersect(trial_uq_idx,nonlinear_idx)
  @assert isempty(trial_uq_idx_nonlin) "Not implemented"

  (compose_functionals(p.param_fe_functional[time_dependent_form_idx]),
   Broadcasting(compose_functionals)(p.param_fe_functional[trial_uv_idx_nonlin]),
   Broadcasting(compose_functionals)(p.param_fe_functional[trial_uv_idx_lin]),
   Broadcasting(compose_functionals)(p.param_fe_functional[trial_uq_idx_lin]))
end

function compose_bilinear_functionals(::ParamFEProblem)
  error("Not implemented")
end

function param_operator(problem::ParamFEProblem{A,D,false,true}) where {A,D}
  param_form2,param_form1 = compose_functionals(problem)
  lhs(μ,u,v) = sum([param_form2[i](μ,u,v) for i=eachindex(param_form2)])
  rhs(μ,v) = sum([param_form1[i](μ,v) for i=eachindex(param_form1)])
  param_space = get_param_space(problem)
  trial = get_trial(problem)
  test = get_test(problem)

  ParamAffineFEOperator(lhs,rhs,param_space,trial...,test...)
end

function param_operator(problem::ParamFEProblem{A,D,true,true}) where {A,D}
  all_param_form2,all_param_form1 = compose_functionals(problem)
  param_form2_uv,param_form2_uq = all_param_form2
  param_form1_v,param_form1_q = all_param_form1

  function lhs(μ,(u,p),(v,q))
    (sum([param_form2_uv[i](μ,u,v) for i=eachindex(param_form2_uv)]) -
     sum([param_form2_uq[i](μ,u,q) for i=eachindex(param_form2_uq)]) -
     sum([param_form2_uq[i](μ,p,v) for i=eachindex(param_form2_uq)]))
  end

  function rhs(μ,(v,q))
    (sum([param_form1_v[i](μ,v) for i=eachindex(param_form1_v)]) -
     sum([param_form1_q[i](μ,q) for i=eachindex(param_form1_q)]))
  end

  param_space = get_param_space(problem)
  trial = get_trial(problem)
  test = get_test(problem)
  X,Y = ParamMultiFieldTrialFESpace(trial),MultiFieldFESpace(test)

  ParamAffineFEOperator(lhs,rhs,param_space,X,Y)
end

function param_operator(problem::ParamFEProblem{Nonlinear,D,true,true}) where D
  all_param_form2,all_param_form1 = compose_functionals(problem)
  param_form2_nonlin,param_form2_uv,param_form2_uq = all_param_form2
  param_form1_v,param_form1_q = all_param_form1

  function lhs_lin(μ,(u,p),(v,q))
    (sum([param_form2_uv[i](μ,u,v) for i=eachindex(param_form2_uv)]) -
     sum([param_form2_uq[i](μ,u,q) for i=eachindex(param_form2_uq)]) -
     sum([param_form2_uq[i](μ,p,v) for i=eachindex(param_form2_uq)]))
  end

  lhs_nonlin(u,v) = param_form2_nonlin[1](u,u,v)
  dlhs_nonlin(u,du,v) = param_form2_nonlin[2](du,u,v)

  function rhs(μ,(v,q))
    (sum([param_form1_v[i](μ,v) for i=eachindex(param_form1_v)]) -
     sum([param_form1_q[i](μ,q) for i=eachindex(param_form1_q)]))
  end

  res(μ,(u,p),(v,q)) = lhs_lin(μ,(u,p),(v,q)) + lhs_nonlin(u,v) - rhs(μ,(v,q))
  jac(μ,(u,p),(du,dp),(v,q)) = lhs_lin(μ,(du,dp),(v,q)) + dlhs_nonlin(u,du,v)

  param_space = get_param_space(problem)
  trial = get_trial(problem)
  test = get_test(problem)
  X,Y = ParamMultiFieldTrialFESpace(trial),MultiFieldFESpace(test)

  ParamAffineFEOperator(lhs,rhs,param_space,X,Y)
end

function param_operator(problem::ParamFEProblem{A,D,false,false}) where {A,D}
  all_param_form2,param_form1 = compose_functionals(problem)
  m,param_form2 = all_param_form2
  lhs(μ,t,u,v) = sum([param_form2[i](μ,t,u,v) for i=eachindex(param_form2)])
  rhs(μ,t,v) = sum([param_form1[i](μ,t,v) for i=eachindex(param_form1)])
  ParamTransientAffineFEOperator(m,lhs,rhs,trial...,test...)
end

function param_operator(problem::ParamFEProblem{A,D,true,false}) where {A,D}
  all_param_form2,all_param_form1 = compose_functionals(problem)
  m,param_form2_uv,param_form2_uq = all_param_form2
  param_form1_v,param_form1_q = all_param_form1

  function lhs(μ,(u,p),(v,q))
    (sum([param_form2_uv[i](μ,u,v) for i=eachindex(param_form2_uv)]) -
     sum([param_form2_uq[i](μ,u,q) for i=eachindex(param_form2_uq)]) -
     sum([param_form2_uq[i](μ,p,v) for i=eachindex(param_form2_uq)]))
  end

  function rhs(μ,(v,q))
    (sum([param_form1_v[i](μ,v) for i=eachindex(param_form1_v)]) -
     sum([param_form1_q[i](μ,q) for i=eachindex(param_form1_q)]))
  end

  param_space = get_param_space(problem)
  trial = get_trial(problem)
  test = get_test(problem)
  X,Y = ParamTransientMultiFieldFESpace(trial),MultiFieldFESpace(test)

  ParamTransientAffineFEOperator(m,lhs,rhs,param_space,X,Y)
end

function param_operator(problem::ParamFEProblem{Nonlinear,D,true,false}) where D
  all_param_form2,all_param_form1 = compose_functionals(problem)
  param_form2_time,param_form2_nonlin,param_form2_uv,param_form2_uq = all_param_form2
  param_form1_v,param_form1_q = all_param_form1

  function m(μ,t,(u,p),(v,q))
    param_form2_time(μ,t,(∂t(u),∂t(p)),(v,q))
  end

  function lhs_lin(μ,t,(u,p),(v,q))
    (sum([param_form2_uv[i](μ,t,u,v) for i=eachindex(param_form2_uv)]) -
     sum([param_form2_uq[i](μ,t,u,q) for i=eachindex(param_form2_uq)]) -
     sum([param_form2_uq[i](μ,t,p,v) for i=eachindex(param_form2_uq)]))
  end

  lhs_nonlin(u,v) = param_form2_nonlin[1](u,u,v)
  dlhs_nonlin(u,du,v) = param_form2_nonlin[2](du,u,v)

  function rhs(μ,t,(v,q))
    (sum([param_form1_v[i](μ,t,v) for i=eachindex(param_form1_v)]) -
     sum([param_form1_q[i](μ,t,q) for i=eachindex(param_form1_q)]))
  end

  res(μ,t,(u,p),(v,q)) = (m(μ,t,(∂t(u),∂t(p)),(v,q)) + lhs_lin(μ,t,(u,p),(v,q)) +
    lhs_nonlin(u,v) - rhs(μ,t,(v,q)))
  jac(μ,t,(u,p),(du,dp),(v,q)) = lhs_lin(μ,t,(du,dp),(v,q)) + dlhs_nonlin(u,du,v)
  jac_t(μ,t,(u,p),(dut,dpt),(v,q)) = m(μ,t,(dut,dpt),(v,q))

  trial = get_trial(problem)
  test = get_test(problem)
  X,Y = ParamTransientMultiFieldFESpace(trial),MultiFieldFESpace(test)

  ParamTransientFEOperator(res,jac,jac_t,X,Y)
end
