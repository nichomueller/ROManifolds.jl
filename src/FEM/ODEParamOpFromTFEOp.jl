struct ODEParamOpFromTFEOp{T} <: ODEParamOperator{T}
  op::TransientParamFEOperator{T}
end

Polynomials.get_order(odeop::ODEParamOpFromTFEOp) = get_order(odeop.op)
FESpaces.get_test(odeop::ODEParamOpFromTFEOp) = get_test(odeop.op)
FESpaces.get_trial(odeop::ODEParamOpFromTFEOp) = get_trial(odeop.op)
realization(odeop::ODEParamOpFromTFEOp;kwargs...) = realization(odeop.op;kwargs...)
get_fe_operator(odeop::ODEParamOpFromTFEOp) = odeop.op
ODEs.get_num_forms(odeop::ODEParamOpFromTFEOp) = get_num_forms(odeop.op)
ODEs.get_forms(odeop::ODEParamOpFromTFEOp) = get_forms(odeop.op)
ODEs.is_form_constant(odeop::ODEParamOpFromTFEOp,k::Integer) = is_form_constant(odeop.op,k)

function get_linear_operator(odeop::ODEParamOpFromTFEOp)
  ODEParamOpFromTFEOp(get_linear_operator(odeop.op))
end

function get_nonlinear_operator(odeop::ODEParamOpFromTFEOp)
  ODEParamOpFromTFEOp(get_nonlinear_operator(odeop.op))
end

function ODEs.allocate_odeopcache(
  odeop::ODEParamOpFromTFEOp,
  r::TransientParamRealization,
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
  assem = get_assembler(odeop.op,r)

  const_forms = ()
  num_forms = get_num_forms(odeop.op)
  jacs = get_jacs(odeop.op)

  μ,t = get_params(r),get_times(r)

  dc = DomainContribution()
  for k in 0:order
    jac = jacs[k+1]
    dc = dc + jac(μ,t,uh,du,v)
  end
  matdata = collect_cell_matrix(trial,test,dc)
  A_full = allocate_matrix(assem,matdata)

  for k in 0:num_forms-1
    const_form = nothing
    if is_form_constant(odeop,k)
      jac = jacs[k+1]
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

function ODEs.update_odeopcache!(odeopcache,odeop::ODEParamOpFromTFEOp,r::TransientParamRealization)
  trials = ()
  for k in 0:get_order(odeop)
    trials = (trials...,evaluate!(odeopcache.Us[k+1],odeopcache.Uts[k+1],r))
  end
  odeopcache.Us = trials

  tfeopcache,op = odeopcache.tfeopcache,odeop.op
  odeopcache.tfeopcache = update_tfeopcache!(tfeopcache,op,r)

  odeopcache
end

function Algebra.allocate_residual(
  odeop::ODEParamOpFromTFEOp,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache)

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
  test = get_test(odeop.op)
  v = get_fe_basis(test)
  assem = get_assembler(odeop.op,r)

  μ,t = get_params(r),get_times(r)

  res = get_res(odeop.op)
  vecdata = collect_cell_vector(test,res(μ,t,uh,v))
  allocate_vector(assem,vecdata)
end

function Algebra.residual!(
  b::AbstractVector,
  odeop::ODEParamOpFromTFEOp,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache;
  add::Bool=false)

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
  test = get_test(odeop.op)
  v = get_fe_basis(test)
  assem = get_assembler(odeop.op,r)

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
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache;
  add::Bool=false)

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
  test = get_test(odeop.op)
  v = get_fe_basis(test)
  assem = get_assembler(odeop.op,r)

  !add && fill!(b,zero(eltype(b)))

  μ,t = get_params(r),get_times(r)

  res = get_res(odeop.op)
  dc = res(μ,t,uh,v)

  order = get_order(odeop)
  mass = get_forms(odeop.op)[1]
  ∂tNuh = ∂t(uh,Val(order))
  dc = dc + mass(μ,t,∂tNuh,v)

  vecdata = collect_cell_vector(test,dc)
  assemble_vector_add!(b,assem,vecdata)

  r
end

function Algebra.residual!(
  b::AbstractVector,
  odeop::ODEParamOpFromTFEOp{LinearParamODE},
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache;
  add::Bool=false)

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
  test = get_test(odeop.op)
  v = get_fe_basis(test)
  assem = get_assembler(odeop.op,r)

  !add && fill!(b,zero(eltype(b)))

  μ,t = get_params(r),get_times(r)

  # Residual
  res = get_res(odeop.op)
  dc = res(μ,t,uh,v)

  # Forms
  order = get_order(odeop)
  forms = get_forms(odeop.op)
  ∂tkuh = uh
  for k in 0:order
    form = forms[k+1]
    dc = dc + form(μ,t,∂tkuh,v)
    if k < order
      ∂tkuh = ∂t(∂tkuh)
    end
  end

  vecdata = collect_cell_vector(test,dc)
  assemble_vector_add!(b,assem,vecdata)

  b
end

function Algebra.allocate_jacobian(
  odeop::ODEParamOpFromTFEOp,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache)

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
  trial = evaluate(get_trial(odeop.op),nothing)
  du = get_trial_fe_basis(trial)
  test = get_test(odeop.op)
  v = get_fe_basis(test)
  assem = get_assembler(odeop.op,r)

  μ,t = get_params(r),get_times(r)

  jacs = get_jacs(odeop.op)
  dc = DomainContribution()
  for k in 0:get_order(odeop.op)
    jac = jacs[k+1]
    dc = dc + jac(μ,t,uh,du,v)
  end
  matdata = collect_cell_matrix(trial,test,dc)
  allocate_matrix(assem,matdata)
end

function ODEs.jacobian_add!(
  A::AbstractMatrix,
  odeop::ODEParamOpFromTFEOp,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
  trial = evaluate(get_trial(odeop.op),nothing)
  du = get_trial_fe_basis(trial)
  test = get_test(odeop.op)
  v = get_fe_basis(test)
  assem = get_assembler(odeop.op,r)

  μ,t = get_params(r),get_times(r)

  jacs = get_jacs(odeop.op)
  dc = DomainContribution()
  for k in 0:get_order(odeop)
    w = ws[k+1]
    iszero(w) && continue
    jac = jacs[k+1]
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
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
  trial = evaluate(get_trial(odeop.op),nothing)
  du = get_trial_fe_basis(trial)
  test = get_test(odeop.op)
  v = get_fe_basis(test)
  assem = get_assembler(odeop.op,r)

  μ,t = get_params(r),get_times(r)

  jacs = get_jacs(odeop.op)
  dc = DomainContribution()
  for k in 0:get_order(odeop)
    w = ws[k+1]
    iszero(w) && continue
    if is_form_constant(odeop,k)
      axpy_entries!(w,odeopcache.const_forms[k+1],A)
    else
      jac = jacs[k+1]
      dc = dc + w * jac(μ,t,uh,du,v)
    end
  end

  if num_domains(dc) > 0
    matdata = collect_cell_matrix(trial,test,dc)
    assemble_matrix_add!(A,assem,matdata)
  end

  A
end

struct ODEParamOpFromTFEOpWithTrian{T} <: ODEParamOperatorWithTrian{T}
  op::TransientParamFEOperatorWithTrian{T}
end

Polynomials.get_order(odeop::ODEParamOpFromTFEOpWithTrian) = get_order(odeop.op)
FESpaces.get_test(odeop::ODEParamOpFromTFEOpWithTrian) = get_test(odeop.op)
FESpaces.get_trial(odeop::ODEParamOpFromTFEOpWithTrian) = get_trial(odeop.op)
realization(odeop::ODEParamOpFromTFEOpWithTrian;kwargs...) = realization(odeop.op;kwargs...)
get_fe_operator(odeop::ODEParamOpFromTFEOpWithTrian) = odeop.op
ODEs.get_num_forms(odeop::ODEParamOpFromTFEOpWithTrian) = get_num_forms(odeop.op)
ODEs.get_forms(odeop::ODEParamOpFromTFEOpWithTrian) = get_forms(odeop.op)
ODEs.is_form_constant(odeop::ODEParamOpFromTFEOpWithTrian,k::Integer) = is_form_constant(odeop.op,k)

function get_linear_operator(odeop::ODEParamOpFromTFEOpWithTrian)
  ODEParamOpFromTFEOpWithTrian(get_linear_operator(odeop.op))
end

function get_nonlinear_operator(odeop::ODEParamOpFromTFEOpWithTrian)
  ODEParamOpFromTFEOpWithTrian(get_nonlinear_operator(odeop.op))
end

function set_triangulation(odeop::ODEParamOpFromTFEOpWithTrian,trians_rhs,trians_lhs)
  ODEParamOpFromTFEOpWithTrian(set_triangulation(odeop.op,trians_rhs,trians_lhs))
end

function change_triangulation(odeop::ODEParamOpFromTFEOpWithTrian,trians_rhs,trians_lhs)
  ODEParamOpFromTFEOpWithTrian(change_triangulation(odeop.op,trians_rhs,trians_lhs))
end

function ODEs.allocate_odeopcache(
  odeop::ODEParamOpFromTFEOpWithTrian,
  r::TransientParamRealization,
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
  assem = get_assembler(odeop.op,r)

  const_forms = ()
  num_forms = get_num_forms(odeop.op)
  jacs = get_jacs(odeop.op)

  μ,t = get_params(r),get_times(r)

  dc = DomainContribution()
  for k in 0:order
    jac = jacs[k+1]
    dc = dc + jac(μ,t,uh,du,v)
  end
  matdata = collect_cell_matrix(trial,test,dc)
  A_full = allocate_matrix(assem,matdata)

  for k in 0:num_forms-1
    const_form = nothing
    if is_form_constant(odeop,k)
      jac = jacs[k+1]
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

function ODEs.update_odeopcache!(
  odeopcache,
  odeop::ODEParamOpFromTFEOpWithTrian,
  r::TransientParamRealization)

  trials = ()
  for k in 0:get_order(odeop)
    trials = (trials...,evaluate!(odeopcache.Us[k+1],odeopcache.Uts[k+1],r))
  end
  odeopcache.Us = trials

  tfeopcache,op = odeopcache.tfeopcache,odeop.op
  odeopcache.tfeopcache = update_tfeopcache!(tfeopcache,op,r)

  odeopcache
end

function Algebra.allocate_residual(
  odeop::ODEParamOpFromTFEOpWithTrian,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache)

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
  test = get_test(odeop.op)
  v = get_fe_basis(test)
  assem = get_assembler(odeop.op,r)

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
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache;
  add::Bool=false)

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
  test = get_test(odeop.op)
  v = get_fe_basis(test)
  assem = get_assembler(odeop.op,r)

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

function Algebra.residual!(
  b::Contribution,
  odeop::ODEParamOpFromTFEOp{SemilinearParamODE},
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache;
  add::Bool=false)

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
  test = get_test(odeop.op)
  v = get_fe_basis(test)
  assem = get_assembler(odeop.op,r)

  !add && fill!(b,zero(eltype(b)))

  μ,t = get_params(r),get_times(r)

  res = get_res(odeop.op)
  dc = res(μ,t,uh,v)

  order = get_order(odeop)
  mass = get_forms(odeop.op)[1]
  ∂tNuh = ∂t(uh,Val(order))
  dc = dc + mass(μ,t,∂tNuh,v)

  map(b.values,odeop.op.trian_res) do values,trian
    vecdata = collect_cell_vector_for_trian(test,dc,trian)
    assemble_vector_add!(values,assem,vecdata)
  end
  b
end

function Algebra.residual!(
  b::Contribution,
  odeop::ODEParamOpFromTFEOpWithTrian{LinearParamODE},
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache;
  add::Bool=false)

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
  test = get_test(odeop.op)
  v = get_fe_basis(test)
  assem = get_assembler(odeop.op,r)

  !add && fill!(b,zero(eltype(b)))

  μ,t = get_params(r),get_times(r)

  # Residual
  res = get_res(odeop.op)
  dc = res(μ,t,uh,v)

  # Forms
  order = get_order(odeop)
  forms = get_forms(odeop.op)
  ∂tkuh = uh
  for k in 0:order
    form = forms[k+1]
    dc = dc + form(μ,t,∂tkuh,v)
    if k < order
      ∂tkuh = ∂t(∂tkuh)
    end
  end

  map(b.values,odeop.op.trian_res) do values,trian
    vecdata = collect_cell_vector_for_trian(test,dc,trian)
    assemble_vector_add!(values,assem,vecdata)
  end
  b
end

function Algebra.allocate_jacobian(
  odeop::ODEParamOpFromTFEOpWithTrian,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache)

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
  trial = evaluate(get_trial(odeop.op),nothing)
  du = get_trial_fe_basis(trial)
  test = get_test(odeop.op)
  v = get_fe_basis(test)
  assem = get_assembler(odeop.op,r)

  μ,t = get_params(r),get_times(r)

  jacs = get_jacs(odeop.op)
  As = ()
  for k in 0:get_order(odeop.op)
    jac = jacs[k+1]
    trian_jac = odeop.op.trian_jacs[k+1]
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
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
  trial = evaluate(get_trial(odeop.op),nothing)
  du = get_trial_fe_basis(trial)
  test = get_test(odeop.op)
  v = get_fe_basis(test)
  assem = get_assembler(odeop.op,r)

  μ,t = get_params(r),get_times(r)

  jacs = get_jacs(odeop.op)
  for k in 0:get_order(odeop)
    A = As[k+1]
    w = ws[k+1]
    iszero(w) && continue
    jac = jacs[k+1]
    trian_jac = odeop.op.trian_jacs[k+1]
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

function ODEs.jacobian_add!(
  As::TupOfArrayContribution,
  odeop::ODEParamOpFromTFEOpWithTrian{LinearParamODE},
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
  trial = evaluate(get_trial(odeop.op),nothing)
  du = get_trial_fe_basis(trial)
  test = get_test(odeop.op)
  v = get_fe_basis(test)
  assem = get_assembler(odeop.op,r)

  μ,t = get_params(r),get_times(r)

  jacs = get_jacs(odeop.op)
  for k in 0:get_order(odeop)
    A = As[k+1]
    w = ws[k+1]
    iszero(w) && continue
    jac = jacs[k+1]
    if is_form_constant(odeop,k)
      axpy_entries!(w,odeopcache.const_forms[k+1],A)
    else
      trian_jac = odeop.op.trian_jacs[k+1]
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
