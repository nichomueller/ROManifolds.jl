const ODEParamOpFromFEOp{O<:ODEParamOperatorType,T<:TriangulationStyle} = ParamOpFromFEOp{O,T}

const JointODEParamOpFromFEOp{O<:ODEParamOperatorType} = ODEParamOpFromFEOp{O,JointDomains}

function Algebra.allocate_residual(
  odeop::JointODEParamOpFromFEOp,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  paramcache)

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
  odeop::JointODEParamOpFromFEOp,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  paramcache;
  add::Bool=false)

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
  odeop::JointODEParamOpFromFEOp{LinearParamODE},
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  paramcache;
  add::Bool=false)

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
  odeop::JointODEParamOpFromFEOp,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  paramcache)

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
  odeop::JointODEParamOpFromFEOp,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache)

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

# function ODEs.jacobian_add!(
#   A::AbstractMatrix,
#   odeop::JointODEParamOpFromFEOp{LinearParamODE},
#   r::TransientRealization,
#   us::Tuple{Vararg{AbstractVector}},
#   ws::Tuple{Vararg{Real}},
#   cache::SystemCache)

#   paramcache = cache.paramcache
#   uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
#   trial = evaluate(get_trial(odeop.op),nothing)
#   du = get_trial_fe_basis(trial)
#   test = get_test(odeop.op)
#   v = get_fe_basis(test)
#   assem = get_param_assembler(odeop.op,r)

#   μ,t = get_params(r),get_times(r)

#   jacs = get_jacs(odeop.op)
#   dc = DomainContribution()
#   for k in 1:get_order(odeop)+1
#     w = ws[k]
#     iszero(w) && continue
#     if is_form_constant(odeop,k)
#       axpy_entries!(w,cache.A[k],A)
#     else
#       jac = jacs[k]
#       dc = dc + w * jac(μ,t,uh,du,v)
#     end
#   end

#   if num_domains(dc) > 0
#     matdata = collect_cell_matrix(trial,test,dc)
#     assemble_matrix_add!(A,assem,matdata)
#   end

#   A
# end

# function ParamAlgebra.allocate_systemcache(
#   odeop::JointODEParamOpFromFEOp{LinearParamODE},
#   r::TransientRealization,
#   us::Tuple{Vararg{AbstractVector}},
#   ws::Tuple{Vararg{Real}},
#   paramcache)

#   us0 = ()
#   for k in eachindex(us)
#     us0k = copy(us[k])
#     fill!(us0k,zero(eltype(us0k)))
#     us0 = (us0...,us0k)
#   end

#   b = residual(odeop,r,us0,paramcache)

#   feop = get_fe_operator(odeop)
#   uh0 = ODEs._make_uh_from_us(odeop,us0,paramcache.trial)
#   test = get_test(feop)
#   v = get_fe_basis(test)
#   trial = evaluate(get_trial(feop),nothing)
#   du = get_trial_fe_basis(trial)
#   assem = get_param_assembler(feop,r)
#   jacs = get_jacs(feop)
#   μ,t = get_params(r),get_times(r)

#   A = ()
#   for k in 1:get_order(feop)+1
#     jac = jacs[k]
#     w = ws[k]
#     iszero(w) && continue
#     dc = w*jac(μ,t,uh0,du,v)
#     matdata = collect_cell_matrix(trial,test,dc)
#     A = (A...,assemble_matrix(assem,matdata))
#   end

#   return A,b
# end

const SplitODEParamOpFromFEOp{O<:ODEParamOperatorType} = ODEParamOpFromFEOp{O,SplitDomains}

function Algebra.allocate_residual(
  odeop::SplitODEParamOpFromFEOp,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  paramcache)

  uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
  test = get_test(odeop.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(odeop.op,r)

  μ,t = get_params(r),get_times(r)
  trian_res = get_domains_res(odeop.op)
  res = get_res(odeop.op)
  dc = res(μ,t,uh,v)

  b = contribution(trian_res) do trian
    vecdata = collect_cell_vector_for_trian(test,dc,trian)
    allocate_vector(assem,vecdata)
  end
  b
end

function Algebra.residual!(
  b::ArrayContribution,
  odeop::SplitODEParamOpFromFEOp,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  paramcache;
  add::Bool=false)

  uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
  test = get_test(odeop.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(odeop.op,r)

  !add && fill!(b,zero(eltype(b)))

  μ,t = get_params(r),get_times(r)

  trian_res = get_domains_res(odeop.op)
  res = get_res(odeop.op)
  dc = res(μ,t,uh,v)

  map(b.values,trian_res) do values,trian
    vecdata = collect_cell_vector_for_trian(test,dc,trian)
    assemble_vector_add!(values,assem,vecdata)
  end
  b
end

function Algebra.residual(
  odeop::SplitODEParamOpFromFEOp,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  paramcache)

  uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
  test = get_test(odeop.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(odeop.op,r)

  μ,t = get_params(r),get_times(r)

  # Residual
  trian_res = get_domains_res(odeop.op)
  res = get_res(odeop.op)
  dc = res(μ,t,uh,v)

  contribution(trian_res) do trian
    vecdata = collect_cell_vector_for_trian(test,dc,trian)
    assemble_vector(assem,vecdata)
  end
end

function Algebra.allocate_jacobian(
  odeop::SplitODEParamOpFromFEOp,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  paramcache)

  uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
  trial = evaluate(get_trial(odeop.op),nothing)
  du = get_trial_fe_basis(trial)
  test = get_test(odeop.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(odeop.op,r)

  μ,t = get_params(r),get_times(r)

  trian_jacs = get_domains_jac(odeop.op)
  jacs = get_jacs(odeop.op)
  As = ()
  for k in 1:get_order(odeop.op)+1
    jac = jacs[k]
    trian_jac = trian_jacs[k]
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
  odeop::SplitODEParamOpFromFEOp,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache)

  uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
  trial = evaluate(get_trial(odeop.op),nothing)
  du = get_trial_fe_basis(trial)
  test = get_test(odeop.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(odeop.op,r)

  μ,t = get_params(r),get_times(r)

  trian_jacs = get_domains_jac(odeop.op)
  jacs = get_jacs(odeop.op)
  for k in 1:get_order(odeop)+1
    A = As[k]
    w = ws[k]
    iszero(w) && continue
    jac = jacs[k]
    trian_jac = trian_jacs[k]
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

# function ODEs.jacobian_add!(
#   As::TupOfArrayContribution,
#   odeop::SplitODEParamOpFromFEOp{LinearParamODE},
#   r::TransientRealization,
#   us::Tuple{Vararg{AbstractVector}},
#   ws::Tuple{Vararg{Real}},
#   cache::SystemCache)

#   paramcache = cache.paramcache
#   uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
#   trial = evaluate(get_trial(odeop.op),nothing)
#   du = get_trial_fe_basis(trial)
#   test = get_test(odeop.op)
#   v = get_fe_basis(test)
#   assem = get_param_assembler(odeop.op,r)

#   μ,t = get_params(r),get_times(r)

#   trian_jacs = get_domains_jac(odeop.op)
#   jacs = get_jacs(odeop.op)
#   for k in 1:get_order(odeop)+1
#     A = As[k]
#     w = ws[k]
#     iszero(w) && continue
#     if is_form_constant(odeop,k)
#       axpy_entries!(w,cache.A[k],A)
#     else
#       jac = jacs[k]
#       trian_jac = trian_jacs[k]
#       dc = w * jac(μ,t,uh,du,v)
#       if num_domains(dc) > 0
#         map(A.values,trian_jac) do values,trian
#           matdata = collect_cell_matrix_for_trian(trial,test,dc,trian)
#           assemble_matrix_add!(values,assem,matdata)
#         end
#       end
#     end
#   end

#   As
# end

function Algebra.jacobian(
  odeop::SplitODEParamOpFromFEOp,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache)

  uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
  trial = evaluate(get_trial(odeop.op),nothing)
  du = get_trial_fe_basis(trial)
  test = get_test(odeop.op)
  v = get_fe_basis(test)

  assem = get_param_assembler(odeop.op,r)
  μ,t = get_params(r),get_times(r)

  trian_jacs = get_domains_jac(odeop.op)
  jacs = get_jacs(odeop.op)
  As = ()
  for k in 1:get_order(odeop.op)+1
    w = ws[k]
    iszero(w) && continue
    jac = jacs[k]
    trian_jac = trian_jacs[k]
    dc = w * jac(μ,t,uh,du,v)
    A = contribution(trian_jac) do trian
      matdata = collect_cell_matrix_for_trian(trial,test,dc,trian)
      assemble_matrix(assem,matdata)
    end
    As = (As...,A)
  end

  As
end


const LinearNonlinearODEParamOpFromFEOp{T} = LinearNonlinearParamOpFromFEOp{LinearNonlinearParamODE,T}
