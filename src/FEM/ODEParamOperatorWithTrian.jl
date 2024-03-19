struct ODEParamOpFromTFEOpWithTrian{T} <: ODEParamOperator{T}
  tfeop::TransientParamFEOperatorWithTrian{T}
end

function Polynomials.get_order(odeop::ODEParamOpFromTFEOpWithTrian)
  get_order(odeop.tfeop)
end

function ODEs.get_num_forms(odeop::ODEParamOpFromTFEOpWithTrian)
  get_num_forms(odeop.tfeop)
end

function ODEs.get_forms(odeop::ODEParamOpFromTFEOpWithTrian)
  get_forms(odeop.tfeop)
end

function ODEs.is_form_constant(odeop::ODEParamOpFromTFEOpWithTrian,k::Integer)
  is_form_constant(odeop.tfeop,k)
end

function allocate_odeopcache(
  odeop::ODEParamOpFromTFEOpWithTrian,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}})

  order = get_order(odeop)
  Ut = get_trial(odeop.tfeop)
  U = allocate_space(Ut)
  Uts = (Ut,)
  Us = (U,)
  for k in 1:order
    Uts = (Uts...,∂t(Uts[k]))
    Us = (Us...,allocate_space(Uts[k+1]))
  end

  tfeopcache = allocate_tfeopcache(odeop.tfeop,r,us)

  uh = ODEs._make_uh_from_us(odeop,us,Us)
  V = get_test(odeop.tfeop)
  v = get_fe_basis(V)
  Ut = evaluate(get_trial(odeop.tfeop),nothing)
  du = get_trial_fe_basis(Ut)
  assem = get_assembler(odeop.tfeop,r)

  const_forms = ()
  num_forms = get_num_forms(odeop.tfeop)
  jacs = get_jacs(odeop.tfeop)

  μ,t = get_params(r),get_times(r)

  dc = DomainContribution()
  for k in 0:order
    jac = jacs[k+1]
    dc = dc + jac(μ,t,uh,du,v)
  end
  matdata = collect_cell_matrix(Ut,V,dc)
  A_full = allocate_matrix(assem,matdata)

  for k in 0:num_forms-1
    const_form = nothing
    if is_form_constant(odeop,k)
      jac = jacs[k+1]
      dc = jac(μ,t,uh,du,v)
      matdata = collect_cell_matrix(Ut,V,dc)
      const_form = copy(A_full)
      fillstored!(const_form,zero(eltype(const_form)))
      assemble_matrix_add!(const_form,assem,matdata)
    end
    const_forms = (const_forms...,const_form)
  end

  ODEOpFromTFEOpCache(Us,Uts,tfeopcache,const_forms)
end

function ODEs.update_odeopcache!(odeopcache,odeop::ODEParamOpFromTFEOpWithTrian,r::TransientParamRealization)
  Us = ()
  for k in 0:get_order(odeop)
    Us = (Us...,evaluate!(odeopcache.Us[k+1],odeopcache.Uts[k+1],r))
  end
  odeopcache.Us = Us

  tfeopcache,tfeop = odeopcache.tfeopcache,odeop.tfeop
  odeopcache.tfeopcache = update_tfeopcache!(tfeopcache,tfeop,r)

  odeopcache
end

function Algebra.allocate_residual(
  odeop::ODEParamOpFromTFEOpWithTrian,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache)

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
  V = get_test(odeop.tfeop)
  v = get_fe_basis(V)
  assem = get_assembler(odeop.tfeop,r)

  μ,t = get_params(r),get_times(r)
  res = get_res(odeop.tfeop)
  dc = res(μ,t,uh,v)

  b = contribution(op.trian_res) do trian
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
  V = get_test(odeop.tfeop)
  v = get_fe_basis(V)
  assem = get_assembler(odeop.tfeop,r)

  !add && fill!(b,zero(eltype(b)))

  μ,t = get_params(r),get_times(r)

  res = get_res(odeop.tfeop)
  dc = res(μ,t,uh,v)

  map(b.values,op.trian_res) do values,trian
    vecdata = collect_cell_vector_for_trian(V,dc,trian)
    assemble_vector_add!(values,assem,vecdata)
  end
  b
end

function Algebra.residual!(
  b::Contribution,
  odeop::ODEParamOpFromTFEOpWithTrian{<:AbstractLinearODE},
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache;
  add::Bool=false)

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
  V = get_test(odeop.tfeop)
  v = get_fe_basis(V)
  assem = get_assembler(odeop.tfeop,r)

  !add && fill!(b,zero(eltype(b)))

  μ,t = get_params(r),get_times(r)

  # Residual
  res = get_res(odeop.tfeop)
  dc = res(μ,t,uh,v)

  # Forms
  order = get_order(odeop)
  forms = get_forms(odeop.tfeop)
  ∂tkuh = uh
  for k in 0:order
    form = forms[k+1]
    dc = dc + form(μ,t,∂tkuh,v)
    if k < order
      ∂tkuh = ∂t(∂tkuh)
    end
  end

  map(b.values,op.trian_res) do values,trian
    vecdata = collect_cell_vector_for_trian(V,dc,trian)
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
  Ut = evaluate(get_trial(odeop.tfeop),nothing)
  du = get_trial_fe_basis(Ut)
  V = get_test(odeop.tfeop)
  v = get_fe_basis(V)
  assem = get_assembler(odeop.tfeop,r)

  μ,t = get_params(r),get_times(r)

  jacs = get_jacs(odeop.tfeop)
  As = ()
  for k in 0:get_order(odeop.tfeop)
    jac = jacs[k+1]
    trian_jac = op.trian_jacs[k+1]
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
  As::Tuple{Vararg{Contribution}},
  odeop::ODEParamOpFromTFEOpWithTrian,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
  Ut = evaluate(get_trial(odeop.tfeop),nothing)
  du = get_trial_fe_basis(Ut)
  V = get_test(odeop.tfeop)
  v = get_fe_basis(V)
  assem = get_assembler(odeop.tfeop,r)

  μ,t = get_params(r),get_times(r)

  jacs = get_jacs(odeop.tfeop)
  for k in 0:get_order(odeop)
    A = As[k+1]
    w = ws[k+1]
    iszero(w) && continue
    jac = jacs[k+1]
    trian_jac = op.trian_jacs[k+1]
    dc = w * jac(μ,t,uh,du,v)
    if num_domains(dc) > 0
      map(A.values,trian_jac) do values,trian
        matdata = collect_cell_matrix_for_trian(trial,test,dc,trian)
        assemble_matrix_add!(values,assem,matdata)
      end
    end
  end

  A
end

function ODEs.jacobian_add!(
  A::Tuple{Vararg{Contribution}},
  odeop::ODEParamOpFromTFEOpWithTrian{<:AbstractLinearODE},
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  uh = ODEs._make_uh_from_us(odeop,us,odeopcache.Us)
  Ut = evaluate(get_trial(odeop.tfeop),nothing)
  du = get_trial_fe_basis(Ut)
  V = get_test(odeop.tfeop)
  v = get_fe_basis(V)
  assem = get_assembler(odeop.tfeop,r)

  μ,t = get_params(r),get_times(r)

  jacs = get_jacs(odeop.tfeop)
  for k in 0:get_order(odeop)
    A = As[k+1]
    w = ws[k+1]
    iszero(w) && continue
    jac = jacs[k+1]
    if is_form_constant(odeop,k)
      axpy_entries!(w,odeopcache.const_forms[k+1],A)
    else
      trian_jac = op.trian_jacs[k+1]
      dc = w * jac(μ,t,uh,du,v)
      if num_domains(dc) > 0
        map(A.values,trian_jac) do values,trian
          matdata = collect_cell_matrix_for_trian(trial,test,dc,trian)
          assemble_matrix_add!(values,assem,matdata)
        end
      end
    end
  end
end
