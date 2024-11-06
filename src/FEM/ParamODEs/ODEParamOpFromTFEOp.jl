"""
"""
struct ODEParamOpFromTFEOp{O,T} <: ODEParamOperator{O,T}
  op::TransientParamFEOperator{O,T}
end

ParamSteady.get_fe_operator(op::ODEParamOpFromTFEOp) = op.op

for f in (:(Utils.set_domains),:(Utils.change_domains))
  @eval begin
    function $f(odeop::ODEParamOpFromTFEOp,trians_rhs,trians_lhs)
      ODEParamOpFromTFEOp($f(odeop.op,trians_rhs,trians_lhs))
    end
  end
end

const JointODEParamOpFromTFEOp{O} = ODEParamOpFromTFEOp{O,JointTriangulation}

function Algebra.allocate_residual(
  odeop::ODEParamOpFromTFEOp,
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
  odeop::ODEParamOpFromTFEOp,
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
  odeop::ODEParamOpFromTFEOp{LinearParamODE},
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
  odeop::ODEParamOpFromTFEOp,
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
  odeop::ODEParamOpFromTFEOp,
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

function ODEs.jacobian_add!(
  A::AbstractMatrix,
  odeop::ODEParamOpFromTFEOp{LinearParamODE},
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  cache::ParamOpSysCache)

  paramcache = cache.paramcache
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
      axpy_entries!(w,cache.A[k],A)
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

function ParamSteady.allocate_systemcache(
  odeop::ODEParamOperator{LinearParamODE},
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache)

  us0 = ()
  for k in eachindex(us)
    us0k = copy(us[k])
    fill!(us0k,zero(eltype(us0k)))
    us0 = (us0...,us0k)
  end

  b = residual(odeop,r,us0,paramcache)

  feop = get_fe_operator(odeop)
  uh0 = ODEs._make_uh_from_us(odeop,us0,paramcache.trial)
  test = get_test(feop)
  v = get_fe_basis(test)
  trial = evaluate(get_trial(feop),nothing)
  du = get_trial_fe_basis(trial)
  assem = get_param_assembler(feop,r)
  jacs = get_jacs(feop)
  μ,t = get_params(r),get_times(r)

  A = ()
  for k in 1:get_order(feop)+1
    jac = jacs[k]
    w = ws[k]
    iszero(w) && continue
    dc = w*jac(μ,t,uh0,du,v)
    matdata = collect_cell_matrix(trial,test,dc)
    A = (A...,assemble_matrix(assem,matdata))
  end

  return A,b
end

const SplitODEParamOpFromTFEOp{O} = ODEParamOpFromTFEOp{O,SplitTriangulation}

function Algebra.allocate_residual(
  odeop::SplitODEParamOpFromTFEOp,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  paramcache)

  uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
  test = get_test(odeop.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(odeop.op,r)

  μ,t = get_params(r),get_times(r)
  trian_res = get_trian_res(odeop.op)
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
  odeop::SplitODEParamOpFromTFEOp,
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

  trian_res = get_trian_res(odeop.op)
  res = get_res(odeop.op)
  dc = res(μ,t,uh,v)

  map(b.values,trian_res) do values,trian
    vecdata = collect_cell_vector_for_trian(test,dc,trian)
    assemble_vector_add!(values,assem,vecdata)
  end
  b
end

function Algebra.residual(
  odeop::SplitODEParamOpFromTFEOp,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  paramcache)

  uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
  test = get_test(odeop.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(odeop.op,r)

  μ,t = get_params(r),get_times(r)

  # Residual
  trian_res = get_trian_res(odeop.op)
  res = get_res(odeop.op)
  dc = res(μ,t,uh,v)

  contribution(trian_res) do trian
    vecdata = collect_cell_vector_for_trian(test,dc,trian)
    assemble_vector(assem,vecdata)
  end
end

function Algebra.allocate_jacobian(
  odeop::SplitODEParamOpFromTFEOp,
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

  trian_jacs = get_trian_jac(odeop.op)
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
  odeop::SplitODEParamOpFromTFEOp,
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

  trian_jacs = get_trian_jac(odeop.op)
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

function ODEs.jacobian_add!(
  As::TupOfArrayContribution,
  odeop::SplitODEParamOpFromTFEOp{LinearParamODE},
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  cache::ParamOpSysCache)

  paramcache = cache.paramcache
  uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
  trial = evaluate(get_trial(odeop.op),nothing)
  du = get_trial_fe_basis(trial)
  test = get_test(odeop.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(odeop.op,r)

  μ,t = get_params(r),get_times(r)

  trian_jacs = get_trian_jac(odeop.op)
  jacs = get_jacs(odeop.op)
  for k in 1:get_order(odeop)+1
    A = As[k]
    w = ws[k]
    iszero(w) && continue
    if is_form_constant(odeop,k)
      axpy_entries!(w,cache.A[k],A)
    else
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
  end

  As
end

function Algebra.jacobian(
  odeop::SplitODEParamOpFromTFEOp,
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

  trian_jacs = get_trian_jac(odeop.op)
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

function ParamSteady.allocate_systemcache(
  odeop::SplitODEParamOpFromTFEOp{LinearParamODE},
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache)

  us0 = ()
  for k in eachindex(us)
    us0k = copy(us[k])
    fill!(us0k,zero(eltype(us0k)))
    us0 = (us0...,us0k)
  end

  b = residual(odeop,r,us0,paramcache)

  feop = get_fe_operator(odeop)
  uh0 = ODEs._make_uh_from_us(odeop,us0,paramcache.trial)
  test = get_test(feop)
  v = get_fe_basis(test)
  trial = evaluate(get_trial(feop),nothing)
  du = get_trial_fe_basis(trial)
  assem = get_param_assembler(feop,r)
  trian_jacs = get_trian_jac(feop)
  jacs = get_jacs(feop)
  μ,t = get_params(r),get_times(r)

  A = ()
  for k in 1:get_order(feop)+1
    w = ws[k]
    iszero(w) && continue
    jac = jacs[k]
    trian_jac = trian_jacs[k]
    dc = w * jac(μ,t,uh,du,v)
    Ak = contribution(trian_jac) do trian
      matdata = collect_cell_matrix_for_trian(trial,test,dc,trian)
      assemble_matrix(assem,matdata)
    end
    A = (A...,Ak)
  end

  return A,b
end

# linear - nonlinear case

struct LinearNonlinearParamOpFromTFEOp{T} <: ODEParamOperator{LinearNonlinearParamODE,T}
  op::LinearNonlinearTransientParamFEOperator{T}
end

ParamSteady.get_fe_operator(op::LinearNonlinearParamOpFromTFEOp) = op.op

function ParamSteady.get_linear_operator(op::LinearNonlinearParamOpFromTFEOp)
  get_algebraic_operator(get_linear_operator(op.op))
end

function ParamSteady.get_nonlinear_operator(op::LinearNonlinearParamOpFromTFEOp)
  get_algebraic_operator(get_nonlinear_operator(op.op))
end

function ParamSteady.allocate_paramcache(
  op::LinearNonlinearParamOpFromTFEOp,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}})

  op_lin = get_linear_operator(op)
  op_nlin = get_nonlinear_operator(op)

  paramcache = allocate_paramcache(op_nlin,r,us)
  A_lin,b_lin = allocate_systemcache(op_lin,r,us,paramcache)

  return ParamOpSysCache(paramcache,A_lin,b_lin)
end

function ParamSteady.update_paramcache!(
  cache,
  op::LinearNonlinearParamOpFromTFEOp,
  r::TransientRealization)

  update_paramcache!(cache.paramcache,get_nonlinear_operator(op.op),r)
end

function Algebra.allocate_residual(
  op::LinearNonlinearParamOpFromTFEOp,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  cache)

  b_lin = cache.b
  copy(b_lin)
end

function Algebra.allocate_jacobian(
  op::LinearNonlinearParamOpFromTFEOp,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  cache)

  A_lin = cache.A
  copy(A_lin)
end

function Algebra.residual!(
  b,
  op::LinearNonlinearParamOpFromTFEOp,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  cache)

  A_lin = cache.A
  b_lin = cache.b
  @check isa(A_lin,Tuple)
  paramcache = cache.paramcache
  residual!(b,get_nonlinear_operator(op),r,us,paramcache)
  for (_us,_A_lin) in zip(us,A_lin)
    mul!(b,_A_lin,_us,1,1)
  end
  axpy!(1,b_lin,b)
  b
end

function ODEs.jacobian_add!(
  A,
  op::LinearNonlinearParamOpFromTFEOp,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  cache)

  A_lin = cache.A
  @check isa(A_lin,Tuple)
  paramcache = cache.paramcache
  jacobian_add!(A,get_nonlinear_operator(op),r,us,paramcache)
  for _A_lin in A_lin
    axpy!(1,_A_lin,A)
  end
  A
end
