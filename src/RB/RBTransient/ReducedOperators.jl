function RBSteady.reduced_operator(
  solver::RBSolver,
  odeop::ODEParamOperator,
  red_trial::RBSpace,
  red_test::RBSpace,
  s::AbstractSnapshots)

  red_lhs,red_rhs = reduced_weak_form(solver,odeop,red_trial,red_test,s)
  trians_rhs = get_domains(red_rhs)
  trians_lhs = map(get_domains,red_lhs)
  odeop′ = change_domains(odeop,trians_rhs,trians_lhs)
  GenericRBOperator(odeop′,red_trial,red_test,red_lhs,red_rhs)
end

function RBSteady.reduced_operator(
  solver::RBSolver,
  odeop::ODEParamOperator{LinearNonlinearParamODE},
  red_trial::RBSpace,
  red_test::RBSpace,
  s::AbstractSnapshots)

  red_op_lin = reduced_operator(solver,get_linear_operator(odeop),red_trial,red_test,s)
  red_op_nlin = reduced_operator(solver,get_nonlinear_operator(odeop),red_trial,red_test,s)
  LinearNonlinearRBOperator(red_op_lin,red_op_nlin)
end

const TransientRBOperator{O} = GenericRBOperator{O,TupOfAffineContribution}

function Algebra.allocate_residual(
  op::TransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  paramcache)

  allocate_hypred_cache(op.rhs,r)
end

function Algebra.allocate_jacobian(
  op::TransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  paramcache)

  allocate_hypred_cache(op.lhs,r)
end

function Algebra.residual!(
  b::HRParamArray,
  op::TransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  paramcache
  )

  np = num_params(r)
  hr_time_ids = get_common_time_domain(op.rhs)
  hr_param_time_ids = range_1d(1:np,hr_time_ids,np)
  hr_uh = _make_hr_uh_from_us(op.op,us,paramcache.trial,hr_param_time_ids)

  test = get_test(op.op)
  v = get_fe_basis(test)

  trian_res = get_domains_res(op.op)
  μ = get_params(r)
  hr_t = view(get_times(r),hr_time_ids)
  res = get_res(op.op)
  dc = res(μ,hr_t,hr_uh,v)

  map(trian_res) do strian
    b_strian = b.fecache[strian]
    rhs_strian = op.rhs[strian]
    vecdata = collect_cell_hr_vector(test,dc,strian,rhs_strian,hr_param_time_ids)
    assemble_hr_vector_add!(b_strian,vecdata...)
  end

  inv_project!(b,op.rhs)
end

function Algebra.jacobian!(
  A::HRParamArray,
  op::TransientRBOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache
  )

  np = num_params(r)
  hr_time_ids = get_common_time_domain(op.rhs)
  hr_param_time_ids = range_1d(1:np,hr_time_ids,np)
  hr_uh = _make_hr_uh_from_us(op.op,us,paramcache.trial,hr_param_time_ids)

  trial = get_trial(op.op)
  du = get_trial_fe_basis(trial)
  test = get_test(op.op)
  v = get_fe_basis(test)

  trian_jac = get_domains_jac(op.op)
  μ = get_params(r)
  hr_t = view(get_times(r),hr_time_ids)
  jac = get_jac(op.op)
  dc = jac(μ,t,uh,du,v)

  map(trian_jac) do strian
    A_strian = A.fecache[strian]
    lhs_strian = op.lhs[strian]
    matdata = collect_cell_hr_matrix(trial,test,dc,strian,lhs_strian,hr_param_time_ids)
    assemble_hr_matrix_add!(A_strian,matdata...)
  end

  inv_project!(A,op.lhs)
end

# utils

function _make_hr_uh_from_us(
  odeop::ODEParamOperator,
  us::Tuple{Vararg{AbstractVector}},
  trial::FESpace,
  hr_param_time_ids)

  hr_us,hr_trial = _reduce_arguments(us,trial,hr_param_time_ids)
  ODEs._make_uh_from_us(odeop,hr_us,hr_trial)
end

function _reduce_arguments(
  fv::ConsecutiveParamVector,
  trial::TrialParamFESpace,
  hr_ids::AbstractVector)

  dv = trial.dirichlet_values
  fv′ = ConsecutiveParamArray(view(fv.data,:,hr_ids))
  dv′ = ConsecutiveParamArray(view(dv.data,:,hr_ids))
  trial′ = TrialParamFESpace(dv′,trial.space)
  return fv′,trial′
end

function _reduce_arguments(
  fv::ConsecutiveParamVector,
  trial::TrivialParamFESpace,
  hr_ids::AbstractVector)

  fv′ = ConsecutiveParamArray(view(fv.data,:,hr_ids))
  trial′ = TrialParamFESpace(trial.space,length(hr_ids))
  return fv′,trial′
end

function _reduce_arguments(
  fv::BlockConsecutiveParamVector,
  trial::MultiFieldFESpace,
  hr_ids::AbstractVector)

  vec_fv′,vec_trial′ = map(blocks(fv),trial.spaces) do fv,trial
    _reduce_arguments(fv,trial,hr_ids)
  end |> tuple_of_arrays
  fv′ = mortar(vec_fv′)
  trial′ = MultiFieldFESpace(trial.vector_type,vec_trial′,trial.style)
  return fv′,trial′
end

function _reduce_arguments(
  us::Tuple{Vararg{AbstractVector}},
  trial::Tuple{Vararg{FESpace}},
  hr_ids::AbstractVector)

  @check length(us) == length(trial)
  us′ = ()
  trial′ = ()
  for i = eachindex(us)
    usi′,triali′ = _reduce_arguments(us[i],trial[i],hr_ids)
    us′ = (us′...,usi′)
    trial′ = (trial′...,triali′)
  end
  return us′,triali′
end
