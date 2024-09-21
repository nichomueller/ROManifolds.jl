"""
"""
struct ODEParamOpFromTFEOp{T} <: ODEParamOperator{T}
  op::TransientParamFEOperator{T}
end

ReferenceFEs.get_order(odeop::ODEParamOpFromTFEOp) = get_order(odeop.op)
FESpaces.get_test(odeop::ODEParamOpFromTFEOp) = get_test(odeop.op)
FESpaces.get_trial(odeop::ODEParamOpFromTFEOp) = get_trial(odeop.op)
ParamDataStructures.realization(odeop::ODEParamOpFromTFEOp;kwargs...) = realization(odeop.op;kwargs...)
ParamSteady.get_fe_operator(odeop::ODEParamOpFromTFEOp) = odeop.op
ODEs.get_num_forms(odeop::ODEParamOpFromTFEOp) = get_num_forms(odeop.op)
ODEs.is_form_constant(odeop::ODEParamOpFromTFEOp,k::Integer) = is_form_constant(odeop.op,k)
ParamSteady.get_vector_index_map(odeop::ODEParamOpFromTFEOp) = get_vector_index_map(odeop.op)
ParamSteady.get_matrix_index_map(odeop::ODEParamOpFromTFEOp) = get_matrix_index_map(odeop.op)

function ParamSteady.get_linear_operator(odeop::ODEParamOpFromTFEOp)
  ODEParamOpFromTFEOp(get_linear_operator(odeop.op))
end

function ParamSteady.get_nonlinear_operator(odeop::ODEParamOpFromTFEOp)
  ODEParamOpFromTFEOp(get_nonlinear_operator(odeop.op))
end

function ODEs.allocate_odeopcache(
  odeop::ODEParamOpFromTFEOp,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}})

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

  ODEOpFromTFEOpCache(trials,pttrials,tfeopcache,const_forms)
end

function ODEs.update_odeopcache!(odeopcache,odeop::ODEParamOpFromTFEOp,r::TransientRealization)
  trials = ()
  for k in 1:get_order(odeop)+1
    trials = (trials...,evaluate!(odeopcache.Us[k],odeopcache.Uts[k],r))
  end
  odeopcache.Us = trials

  tfeopcache,op = odeopcache.tfeopcache,odeop.op
  odeopcache.tfeopcache = update_tfeopcache!(tfeopcache,op,r)

  odeopcache
end

function Algebra.allocate_residual(
  odeop::ODEParamOpFromTFEOp,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache)

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
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
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache;
  add::Bool=false)

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
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
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache;
  add::Bool=false)

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
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
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache;
  add::Bool=false)

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
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
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache)

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
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
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
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
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
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
      axpy_entries!(w,odeopcache.const_forms[k],A)
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
end

ReferenceFEs.get_order(odeop::ODEParamOpFromTFEOpWithTrian) = get_order(odeop.op)
FESpaces.get_test(odeop::ODEParamOpFromTFEOpWithTrian) = get_test(odeop.op)
FESpaces.get_trial(odeop::ODEParamOpFromTFEOpWithTrian) = get_trial(odeop.op)
ParamDataStructures.realization(odeop::ODEParamOpFromTFEOpWithTrian;kwargs...) = realization(odeop.op;kwargs...)
ParamSteady.get_fe_operator(odeop::ODEParamOpFromTFEOpWithTrian) = odeop.op
ODEs.get_num_forms(odeop::ODEParamOpFromTFEOpWithTrian) = get_num_forms(odeop.op)
ODEs.is_form_constant(odeop::ODEParamOpFromTFEOpWithTrian,k::Integer) = is_form_constant(odeop.op,k)
ParamSteady.get_vector_index_map(odeop::ODEParamOpFromTFEOpWithTrian) = get_vector_index_map(odeop.op)
ParamSteady.get_matrix_index_map(odeop::ODEParamOpFromTFEOpWithTrian) = get_matrix_index_map(odeop.op)

function ParamSteady.get_linear_operator(odeop::ODEParamOpFromTFEOpWithTrian)
  ODEParamOpFromTFEOpWithTrian(get_linear_operator(odeop.op))
end

function ParamSteady.get_nonlinear_operator(odeop::ODEParamOpFromTFEOpWithTrian)
  ODEParamOpFromTFEOpWithTrian(get_nonlinear_operator(odeop.op))
end

function ParamSteady.set_triangulation(odeop::ODEParamOpFromTFEOpWithTrian,trians_rhs,trians_lhs)
  ODEParamOpFromTFEOpWithTrian(set_triangulation(odeop.op,trians_rhs,trians_lhs))
end

function ParamSteady.change_triangulation(odeop::ODEParamOpFromTFEOpWithTrian,trians_rhs,trians_lhs)
  ODEParamOpFromTFEOpWithTrian(change_triangulation(odeop.op,trians_rhs,trians_lhs))
end

function ODEs.allocate_odeopcache(
  odeop::ODEParamOpFromTFEOpWithTrian,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}})

  odeopcache = _define_odeopcache(odeop,r,us)

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
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

  odeopcache.const_forms = const_forms
  return odeopcache
end

function ODEs.update_odeopcache!(
  odeopcache,
  odeop::ODEParamOpFromTFEOpWithTrian,
  r::TransientRealization)

  trials = ()
  for k in 1:get_order(odeop)+1
    trials = (trials...,evaluate!(odeopcache.Us[k],odeopcache.Uts[k],r))
  end
  odeopcache.Us = trials

  tfeopcache,op = odeopcache.tfeopcache,odeop.op
  odeopcache.tfeopcache = update_tfeopcache!(tfeopcache,op,r)

  odeopcache
end

function _define_odeopcache(
  odeop::ODEParamOpFromTFEOpWithTrian,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}})

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

  ODEOpFromTFEOpCache(trials,pttrials,tfeopcache,nothing)
end

function Algebra.allocate_residual(
  odeop::ODEParamOpFromTFEOpWithTrian,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache)

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
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
  odeop::ODEParamOpFromTFEOp,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache;
  kwargs...)

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
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
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache=_define_odeopcache(odeop,r,us))

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
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
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache)

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
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
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
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
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache=_define_odeopcache(odeop,r,us))

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
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
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
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
      axpy_entries!(w,odeopcache.const_forms[k],A)
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
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache=_define_odeopcache(odeop,r,us))

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
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
