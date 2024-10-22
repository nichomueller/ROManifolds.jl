"""
"""
struct ODEParamOpFromTFEOp{T} <: ODEParamOperator{T}
  op::TransientParamFEOperator{T}
  r::TransientRealization
end

ParamSteady.get_realization(op::ODEParamOpFromTFEOp) = op.r
ParamSteady.get_fe_operator(op::ODEParamOpFromTFEOp) = op.op

function ParamSteady.allocate_paramcache(
  odeop::ODEParamOpFromTFEOp,
  us::Tuple{Vararg{AbstractVector}})

  r = get_realization(odeop)

  order = get_order(odeop)
  pttrial = get_trial(odeop.op)
  trial = allocate_space(pttrial,r)
  pttrials = (pttrial,)
  trials = (trial,)
  for k in 1:order
    pttrials = (pttrials...,∂t(pttrials[k]))
    trials = (trials...,allocate_space(pttrials[k+1],r))
  end

  tfeopcache = allocate_tfeopcache(odeop.op,r,us)

  uh = ODEs._make_uh_from_us(odeop,us,trials)
  test = get_test(odeop.op)
  v = get_fe_basis(test)
  trial = evaluate(get_trial(odeop.op),nothing)
  du = get_trial_fe_basis(trial)
  assem = get_param_assembler(odeop.op,r)

  const_forms = ()
  num_forms = get_num_forms(odeop.op)
  jacs = get_jacs(odeop.op)

  μ,t = get_params(r),get_times(r)

  dc = DomainContribution()
  for k in 1:order+1
    jac = jacs[k]
    dc = dc + jac(μ,t,uh,du,v)
  end
  matdata = collect_cell_matrix(trial,test,dc)
  A_full = allocate_matrix(assem,matdata)

  for k in 1:num_forms
    const_form = nothing
    if is_form_constant(odeop,k)
      jac = jacs[k]
      dc = jac(μ,t,uh,du,v)
      matdata = collect_cell_matrix(trial,test,dc)
      const_form = copy(A_full)
      fillstored!(const_form,zero(eltype(const_form)))
      assemble_matrix_add!(const_form,assem,matdata)
    end
    const_forms = (const_forms...,const_form)
  end

  ParamCache(trials,pttrials,tfeopcache,const_forms)
end

function ParamSteady.update_paramcache!(paramcache,odeop::ODEParamOpFromTFEOp)
  r = get_realization(odeop)

  trials = ()
  for k in 1:get_order(odeop)+1
    trials = (trials...,evaluate!(paramcache.trial[k],paramcache.ptrial[k],r))
  end
  paramcache.trial = trials

  tfeopcache,op = paramcache.feop_cache,odeop.op
  paramcache.feop_cache = update_tfeopcache!(tfeopcache,op,r)

  paramcache
end

function Algebra.allocate_residual(
  odeop::ODEParamOpFromTFEOp,
  us::Tuple{Vararg{AbstractVector}},
  paramcache)

  r = get_realization(odeop)

  uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
  test = get_test(odeop.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(odeop.op,r)

  μ,t = get_params(r),get_times(r)

  res = get_res(odeop.op)
  vecdata = collect_cell_vector(test,res(μ,t,uh,v))
  allocate_vector(assem,vecdata)
end

function Algebra.residual!(
  b::AbstractVector,
  odeop::ODEParamOpFromTFEOp,
  us::Tuple{Vararg{AbstractVector}},
  paramcache;
  add::Bool=false)

  r = get_realization(odeop)

  uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
  test = get_test(odeop.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(odeop.op,r)

  !add && fill!(b,zero(eltype(b)))

  μ,t = get_params(r),get_times(r)

  res = get_res(odeop.op)
  dc = res(μ,t,uh,v)
  vecdata = collect_cell_vector(test,dc)
  assemble_vector_add!(b,assem,vecdata)

  b
end

function Algebra.residual!(
  b::AbstractVector,
  odeop::ODEParamOpFromTFEOp{SemilinearParamODE},
  us::Tuple{Vararg{AbstractVector}},
  paramcache;
  add::Bool=false)

  r = get_realization(odeop)

  uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
  test = get_test(odeop.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(odeop.op,r)

  !add && fill!(b,zero(eltype(b)))

  μ,t = get_params(r),get_times(r)

  res = get_res(odeop.op)
  dc = res(μ,t,uh,v)

  vecdata = collect_cell_vector(test,dc)
  assemble_vector_add!(b,assem,vecdata)

  b
end

function Algebra.residual!(
  b::AbstractVector,
  odeop::ODEParamOpFromTFEOp{LinearParamODE},
  us::Tuple{Vararg{AbstractVector}},
  paramcache;
  add::Bool=false)

  r = get_realization(odeop)

  uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
  test = get_test(odeop.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(odeop.op,r)

  !add && fill!(b,zero(eltype(b)))

  μ,t = get_params(r),get_times(r)

  # Residual
  res = get_res(odeop.op)
  dc = res(μ,t,uh,v)

  vecdata = collect_cell_vector(test,dc)
  assemble_vector_add!(b,assem,vecdata)

  b
end

function Algebra.allocate_jacobian(
  odeop::ODEParamOpFromTFEOp,
  us::Tuple{Vararg{AbstractVector}},
  paramcache)

  r = get_realization(odeop)

  uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
  trial = evaluate(get_trial(odeop.op),nothing)
  du = get_trial_fe_basis(trial)
  test = get_test(odeop.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(odeop.op,r)

  μ,t = get_params(r),get_times(r)

  jacs = get_jacs(odeop.op)
  dc = DomainContribution()
  for k in 1:get_order(odeop.op)+1
    jac = jacs[k]
    dc = dc + jac(μ,t,uh,du,v)
  end
  matdata = collect_cell_matrix(trial,test,dc)
  allocate_matrix(assem,matdata)
end

function ODEs.jacobian_add!(
  A::AbstractMatrix,
  odeop::ODEParamOpFromTFEOp,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache)

  r = get_realization(odeop)

  uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
  trial = evaluate(get_trial(odeop.op),nothing)
  du = get_trial_fe_basis(trial)
  test = get_test(odeop.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(odeop.op,r)

  μ,t = get_params(r),get_times(r)

  jacs = get_jacs(odeop.op)
  dc = DomainContribution()
  for k in 1:get_order(odeop)+1
    w = ws[k]
    iszero(w) && continue
    jac = jacs[k]
    dc = dc + w * jac(μ,t,uh,du,v)
  end

  if num_domains(dc) > 0
    matdata = collect_cell_matrix(trial,test,dc)
    assemble_matrix_add!(A,assem,matdata)
  end

  A
end

function ODEs.jacobian_add!(
  A::AbstractMatrix,
  odeop::ODEParamOpFromTFEOp{LinearParamODE},
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache)

  r = get_realization(odeop)

  uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
  trial = evaluate(get_trial(odeop.op),nothing)
  du = get_trial_fe_basis(trial)
  test = get_test(odeop.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(odeop.op,r)

  μ,t = get_params(r),get_times(r)

  jacs = get_jacs(odeop.op)
  dc = DomainContribution()
  for k in 1:get_order(odeop)+1
    w = ws[k]
    iszero(w) && continue
    if is_form_constant(odeop,k)
      axpy_entries!(w,paramcache.const_forms[k],A)
    else
      jac = jacs[k]
      dc = dc + w * jac(μ,t,uh,du,v)
    end
  end

  if num_domains(dc) > 0
    matdata = collect_cell_matrix(trial,test,dc)
    assemble_matrix_add!(A,assem,matdata)
  end

  A
end

"""
"""
struct ODEParamOpFromTFEOpWithTrian{T} <: ODEParamOperatorWithTrian{T}
  op::TransientParamFEOperatorWithTrian{T}
  r::TransientRealization
end

ParamSteady.get_realization(op::ODEParamOpFromTFEOpWithTrian) = op.r
ParamSteady.get_fe_operator(op::ODEParamOpFromTFEOpWithTrian) = op.op

function ParamSteady.set_triangulation(odeop::ODEParamOpFromTFEOpWithTrian,trians_rhs,trians_lhs)
  ODEParamOpFromTFEOpWithTrian(set_triangulation(odeop.op,trians_rhs,trians_lhs))
end

function ParamSteady.change_triangulation(odeop::ODEParamOpFromTFEOpWithTrian,trians_rhs,trians_lhs)
  ODEParamOpFromTFEOpWithTrian(change_triangulation(odeop.op,trians_rhs,trians_lhs))
end

function ParamSteady.allocate_paramcache(
  odeop::ODEParamOpFromTFEOpWithTrian,
  us::Tuple{Vararg{AbstractVector}})

  r = get_realization(odeop)
  paramcache = _define_odeopcache(odeop,us)

  uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
  test = get_test(odeop.op)
  v = get_fe_basis(test)
  trial = evaluate(get_trial(odeop.op),nothing)
  du = get_trial_fe_basis(trial)
  assem = get_param_assembler(odeop.op,r)

  const_forms = ()
  num_forms = get_num_forms(odeop.op)
  jacs = get_jacs(odeop.op)

  μ,t = get_params(r),get_times(r)

  dc = DomainContribution()
  for k in 1:get_order(odeop)+1
    jac = jacs[k]
    dc = dc + jac(μ,t,uh,du,v)
  end
  matdata = collect_cell_matrix(trial,test,dc)
  A_full = allocate_matrix(assem,matdata)

  for k in 1:num_forms
    const_form = nothing
    if is_form_constant(odeop,k)
      jac = jacs[k]
      dc = jac(μ,t,uh,du,v)
      matdata = collect_cell_matrix(trial,test,dc)
      const_form = copy(A_full)
      fillstored!(const_form,zero(eltype(const_form)))
      assemble_matrix_add!(const_form,assem,matdata)
    end
    const_forms = (const_forms...,const_form)
  end

  paramcache.const_forms = const_forms
  return paramcache
end

function ParamSteady.update_paramcache!(
  paramcache,
  odeop::ODEParamOpFromTFEOpWithTrian)

  r = get_realization(odeop)

  trials = ()
  for k in 1:get_order(odeop)+1
    trials = (trials...,evaluate!(paramcache.trial[k],paramcache.ptrial[k],r))
  end
  paramcache.trial = trials

  tfeopcache,op = paramcache.feop_cache,odeop.op
  paramcache.feop_cache = update_tfeopcache!(tfeopcache,op,r)

  paramcache
end

function _define_odeopcache(
  odeop::ODEParamOpFromTFEOpWithTrian,
  us::Tuple{Vararg{AbstractVector}})

  r = get_realization(odeop)
  order = get_order(odeop)
  pttrial = get_trial(odeop.op)
  trial = evaluate(pttrial,r)
  pttrials = (pttrial,)
  trials = (trial,)
  for k in 1:order
    pttrials = (pttrials...,∂t(pttrials[k]))
    trials = (trials...,evaluate(pttrials[k+1],r))
  end

  tfeopcache = allocate_tfeopcache(odeop.op,r,us)

  ParamCache(trials,pttrials,tfeopcache,nothing)
end

function Algebra.allocate_residual(
  odeop::ODEParamOpFromTFEOpWithTrian,
  us::Tuple{Vararg{AbstractVector}},
  paramcache)

  r = get_realization(odeop)

  uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
  test = get_test(odeop.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(odeop.op,r)

  μ,t = get_params(r),get_times(r)
  res = get_res(odeop.op)
  dc = res(μ,t,uh,v)

  b = contribution(odeop.op.trian_res) do trian
    vecdata = collect_cell_vector_for_trian(test,dc,trian)
    allocate_vector(assem,vecdata)
  end
  b
end

function Algebra.residual!(
  b::Contribution,
  odeop::ODEParamOpFromTFEOpWithTrian,
  us::Tuple{Vararg{AbstractVector}},
  paramcache;
  add::Bool=false)

  r = get_realization(odeop)

  uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
  test = get_test(odeop.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(odeop.op,r)

  !add && fill!(b,zero(eltype(b)))

  μ,t = get_params(r),get_times(r)

  res = get_res(odeop.op)
  dc = res(μ,t,uh,v)

  map(b.values,odeop.op.trian_res) do values,trian
    vecdata = collect_cell_vector_for_trian(test,dc,trian)
    assemble_vector_add!(values,assem,vecdata)
  end
  b
end

function Algebra.residual(
  odeop::ODEParamOpFromTFEOpWithTrian,
  us::Tuple{Vararg{AbstractVector}},
  paramcache=_define_odeopcache(odeop,us))

  r = get_realization(odeop)

  uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
  test = get_test(odeop.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(odeop.op,r)

  μ,t = get_params(r),get_times(r)

  # Residual
  res = get_res(odeop.op)
  dc = res(μ,t,uh,v)

  contribution(odeop.op.trian_res) do trian
    vecdata = collect_cell_vector_for_trian(test,dc,trian)
    assemble_vector(assem,vecdata)
  end
end

function Algebra.allocate_jacobian(
  odeop::ODEParamOpFromTFEOpWithTrian,
  us::Tuple{Vararg{AbstractVector}},
  paramcache)

  r = get_realization(odeop)

  uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
  trial = evaluate(get_trial(odeop.op),nothing)
  du = get_trial_fe_basis(trial)
  test = get_test(odeop.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(odeop.op,r)

  μ,t = get_params(r),get_times(r)

  jacs = get_jacs(odeop.op)
  As = ()
  for k in 1:get_order(odeop.op)+1
    jac = jacs[k]
    trian_jac = odeop.op.trian_jacs[k]
    dc = jac(μ,t,uh,du,v)
    A = contribution(trian_jac) do trian
      matdata = collect_cell_matrix_for_trian(trial,test,dc,trian)
      allocate_matrix(assem,matdata)
    end
    As = (As...,A)
  end
  As
end

function ODEs.jacobian_add!(
  As::TupOfArrayContribution,
  odeop::ODEParamOpFromTFEOpWithTrian,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache)

  r = get_realization(odeop)

  uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
  trial = evaluate(get_trial(odeop.op),nothing)
  du = get_trial_fe_basis(trial)
  test = get_test(odeop.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(odeop.op,r)

  μ,t = get_params(r),get_times(r)

  jacs = get_jacs(odeop.op)
  for k in 1:get_order(odeop)+1
    A = As[k]
    w = ws[k]
    iszero(w) && continue
    jac = jacs[k]
    trian_jac = odeop.op.trian_jacs[k]
    dc = w * jac(μ,t,uh,du,v)
    if num_domains(dc) > 0
      map(A.values,trian_jac) do values,trian
        matdata = collect_cell_matrix_for_trian(trial,test,dc,trian)
        assemble_matrix_add!(values,assem,matdata)
      end
    end
  end

  As
end

function Algebra.jacobian(
  odeop::ODEParamOpFromTFEOpWithTrian,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache=_define_odeopcache(odeop,us))

  r = get_realization(odeop)

  uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
  trial = evaluate(get_trial(odeop.op),nothing)
  du = get_trial_fe_basis(trial)
  test = get_test(odeop.op)
  v = get_fe_basis(test)

  assem = get_param_assembler(odeop.op,r)
  μ,t = get_params(r),get_times(r)

  jacs = get_jacs(odeop.op)
  As = ()
  for k in 1:get_order(odeop.op)+1
    w = ws[k]
    iszero(w) && continue
    jac = jacs[k]
    trian_jac = odeop.op.trian_jacs[k]
    dc = jac(μ,t,uh,du,v)
    A = contribution(trian_jac) do trian
      matdata = collect_cell_matrix_for_trian(trial,test,dc,trian)
      assemble_matrix(assem,matdata)
    end
    As = (As...,A)
  end

  As
end

function ODEs.jacobian_add!(
  As::TupOfArrayContribution,
  odeop::ODEParamOpFromTFEOpWithTrian{LinearParamODE},
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache)

  r = get_realization(odeop)

  uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
  trial = evaluate(get_trial(odeop.op),nothing)
  du = get_trial_fe_basis(trial)
  test = get_test(odeop.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(odeop.op,r)

  μ,t = get_params(r),get_times(r)

  jacs = get_jacs(odeop.op)
  for k in 1:get_order(odeop)+1
    A = As[k]
    w = ws[k]
    iszero(w) && continue
    jac = jacs[k]
    if is_form_constant(odeop,k)
      axpy_entries!(w,paramcache.const_forms[k],A)
    else
      trian_jac = odeop.op.trian_jacs[k]
      dc = w * jac(μ,t,uh,du,v)
      if num_domains(dc) > 0
        map(A.values,trian_jac) do values,trian
          matdata = collect_cell_matrix_for_trian(trial,test,dc,trian)
          assemble_matrix_add!(values,assem,matdata)
        end
      end
    end
  end

  As
end

function Algebra.jacobian(
  odeop::ODEParamOpFromTFEOpWithTrian{LinearParamODE},
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache=_define_odeopcache(odeop,r,us))

  r = get_realization(odeop)

  uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
  trial = evaluate(get_trial(odeop.op),nothing)
  du = get_trial_fe_basis(trial)
  test = get_test(odeop.op)
  v = get_fe_basis(test)

  assem = get_param_assembler(odeop.op,r)
  μ,t = get_params(r),get_times(r)

  jacs = get_jacs(odeop.op)
  As = ()
  for k in 1:get_order(odeop.op)+1
    w = ws[k]
    iszero(w) && continue
    jac = jacs[k]
    trian_jac = odeop.op.trian_jacs[k]
    dc = jac(μ,t,uh,du,v)
    A = contribution(trian_jac) do trian
      matdata = collect_cell_matrix_for_trian(trial,test,dc,trian)
      assemble_matrix(assem,matdata)
    end
    As = (As...,A)
  end

  As
end
