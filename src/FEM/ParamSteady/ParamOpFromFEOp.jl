"""
"""
struct ParamOpFromFEOp{T} <: ParamOperator{T}
  op::ParamFEOperator{T}
end

get_fe_operator(op::ParamOpFromFEOp) = op.op

"""
    allocate_paramcache(op::ParamOpFromFEOp,μ::Realization,u::AbstractVector
      ) -> CacheType

Similar to [`allocate_odeparamcache`](@ref) in [`Gridap`](@ref), when dealing with steady
parametric problems

"""
function allocate_paramcache(
  op::ParamOpFromFEOp,
  μ::Realization,
  u::AbstractVector)

  ptrial = get_trial(op.op)
  trial = evaluate(ptrial,μ)
  fe_cache = allocate_feopcache(op.op,μ,u)

  if is_jac_constant(op.op)
    uh = EvaluationFunction(trial,u)
    test = get_test(op.op)
    v = get_fe_basis(test)
    du = get_trial_fe_basis(trial)
    assem = get_param_assembler(op.op,μ)

    jac = get_jac(op.op)
    matdata = collect_cell_matrix(trial,test,jac(μ,uh,du,v))
    const_jac = assemble_matrix(assem,matdata)
  else
    const_jac = nothing
  end

  ParamCache(trial,ptrial,fe_cache,const_jac)
end

"""
    update_paramcache!(paramcache, op::ParamOpFromFEOp, μ::Realization, u::AbstractVector
      ) -> CacheType

Similar to [`update_odeparamcache!`](@ref) in [`Gridap`](@ref), when dealing with steady
parametric problems

"""
function update_paramcache!(paramcache,μ::Realization,op::ParamOpFromFEOp)
  paramcache.trial = evaluate!(paramcache.trial,paramcache.ptrial,μ)
  paramcache
end

function Algebra.allocate_residual(
  op::ParamOpFromFEOp,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  uh = EvaluationFunction(paramcache.trial,u)
  test = get_test(op.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op.op,μ)

  res = get_res(op.op)
  dc = res(μ,uh,v)
  vecdata = collect_cell_vector(test,dc)
  b = allocate_vector(assem,vecdata)

  b
end

fill_dvalues!(uh::SingleFieldParamFEFunction,z) = fill!(uh.dirichlet_values,z)
fill_dvalues!(uh::MultiFieldParamFEFunction,z) = map(uhi->fill_dvalues!(uhi,z),uh.single_fe_functions)

function Algebra.residual!(
  b::AbstractVector,
  op::ParamOpFromFEOp,
  μ::Realization,
  u::AbstractVector,
  paramcache;
  add::Bool=false)

  !add && fill!(b,zero(eltype(b)))

  uh = EvaluationFunction(paramcache.trial,u)
  test = get_test(op.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op.op,μ)

  res = get_res(op.op)
  dc = res(μ,uh,v)
  vecdata = collect_cell_vector(test,dc)
  assemble_vector_add!(b,assem,vecdata)

  b
end

function Algebra.allocate_jacobian(
  op::ParamOpFromFEOp,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  if is_jac_constant(op.op)
    return copy(paramcache.const_forms)
  end

  uh = EvaluationFunction(paramcache.trial,u)
  trial = evaluate(get_trial(op.op),nothing)
  du = get_trial_fe_basis(trial)
  test = get_test(op.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op.op,μ)

  jac = get_jac(op.op)
  matdata = collect_cell_matrix(trial,test,jac(μ,uh,du,v))
  A = allocate_matrix(assem,matdata)

  A
end

function ODEs.jacobian_add!(
  A::AbstractMatrix,
  op::ParamOpFromFEOp,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  if is_jac_constant(op.op)
    return paramcache.const_forms
  end

  uh = EvaluationFunction(paramcache.trial,u)
  trial = evaluate(get_trial(op.op),nothing)
  du = get_trial_fe_basis(trial)
  test = get_test(op.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op.op,μ)

  jac = get_jac(op.op)
  dc = jac(μ,uh,du,v)
  matdata = collect_cell_matrix(trial,test,dc)
  assemble_matrix_add!(A,assem,matdata)

  A
end

"""
"""
struct ParamOpFromFEOpWithTrian{T} <: ParamOperator{T}
  op::ParamFEOperatorWithTrian{T}
end

get_fe_operator(op::ParamOpFromFEOpWithTrian) = op.op

function set_triangulation(op::ParamOpFromFEOpWithTrian,trians_rhs,trians_lhs)
  ParamOpFromFEOpWithTrian(set_triangulation(op.op,trians_rhs,trians_lhs))
end

function change_triangulation(op::ParamOpFromFEOpWithTrian,trians_rhs,trians_lhs)
  ParamOpFromFEOpWithTrian(change_triangulation(op.op,trians_rhs,trians_lhs))
end

function allocate_paramcache(
  op::ParamOpFromFEOpWithTrian,
  μ::Realization,
  u::AbstractVector)

  ptrial = get_trial(op.op)
  trial = evaluate(ptrial,μ)
  fe_cache = allocate_feopcache(op.op,μ,u)

  if is_jac_constant(op.op)
    uh = EvaluationFunction(trial,u)
    test = get_test(op.op)
    v = get_fe_basis(test)
    du = get_trial_fe_basis(trial)
    assem = get_param_assembler(op.op,μ)

    jac = get_jac(op.op)
    dc = jac(μ,uh,du,v)
    const_jac = contribution(op.op.trian_jac) do trian
      matdata = collect_cell_matrix_for_trian(trial,test,dc,trian)
      assemble_matrix(assem,matdata)
    end
  else
    const_jac = nothing
  end

  ParamCache(trial,ptrial,fe_cache,const_jac)
end

function update_paramcache!(
  paramcache,
  μ::Realization,
  op::ParamOpFromFEOpWithTrian)

  paramcache.trial = evaluate!(paramcache.trial,paramcache.ptrial,μ)
  paramcache
end

function Algebra.allocate_residual(
  op::ParamOpFromFEOpWithTrian,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  uh = EvaluationFunction(paramcache.trial,u)
  test = get_test(op.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op.op,μ)

  res = get_res(op.op)
  dc = res(μ,uh,v)
  contribution(op.op.trian_res) do trian
    vecdata = collect_cell_vector_for_trian(test,dc,trian)
    allocate_vector(assem,vecdata)
  end
end

function Algebra.residual!(
  b::Contribution,
  op::ParamOpFromFEOpWithTrian,
  μ::Realization,
  u::AbstractVector,
  paramcache;
  add::Bool=false)

  !add && fill!(b,zero(eltype(b)))

  uh = EvaluationFunction(paramcache.trial,u)
  test = get_test(op.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op.op,μ)

  res = get_res(op.op)
  dc = res(μ,uh,v)

  map(b.values,op.op.trian_res) do values,trian
    vecdata = collect_cell_vector_for_trian(test,dc,trian)
    assemble_vector_add!(values,assem,vecdata)
  end

  b
end

function Algebra.allocate_jacobian(
  op::ParamOpFromFEOpWithTrian,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  if is_jac_constant(op.op)
    return copy(paramcache.const_forms)
  end

  uh = EvaluationFunction(paramcache.trial,u)
  trial = evaluate(get_trial(op.op),nothing)
  du = get_trial_fe_basis(trial)
  test = get_test(op.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op.op,μ)

  jac = get_jac(op.op)
  dc = jac(μ,uh,du,v)
  contribution(op.op.trian_jac) do trian
    matdata = collect_cell_matrix_for_trian(trial,test,dc,trian)
    allocate_matrix(assem,matdata)
  end
end

function ODEs.jacobian_add!(
  A::Contribution,
  op::ParamOpFromFEOpWithTrian,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  if is_jac_constant(op.op)
    return paramcache.const_forms
  end

  uh = EvaluationFunction(paramcache.trial,u)
  trial = evaluate(get_trial(op.op),nothing)
  du = get_trial_fe_basis(trial)
  test = get_test(op.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op.op,μ)

  jac = get_jac(op.op)
  dc = jac(μ,uh,du,v)
  map(A.values,op.op.trian_jac) do values,trian
    matdata = collect_cell_matrix_for_trian(trial,test,dc,trian)
    assemble_matrix_add!(values,assem,matdata)
  end

  A
end

struct LinearNonlinearParamOpFromFEOp <: ParamOperator{LinearNonlinearParamEq}
  op::LinearNonlinearParamFEOperator
end

get_fe_operator(op::LinearNonlinearParamOpFromFEOp) = op.op

function get_linear_operator(op::LinearNonlinearParamOpFromFEOp)
  get_algebraic_operator(get_linear_operator(op.op))
end

function get_nonlinear_operator(op::LinearNonlinearParamOpFromFEOp)
  get_algebraic_operator(get_nonlinear_operator(op.op))
end

function allocate_paramcache(
  op::LinearNonlinearParamOpFromFEOp,
  μ::Realization,
  u::AbstractVector)

  paramcache = allocate_paramcache(get_nonlinear_operator(op),μ,u)
  op_lin = get_linear_operator(op)
  A_lin = jacobian(op_lin,μ,u,paramcache)
  b_lin = residual(op_lin,μ,u,paramcache)
  return LinearNonlinearParamCache(paramcache,A_lin,b_lin)
end

function update_paramcache!(
  cache,
  μ::Realization,
  op::LinearNonlinearParamOpFromFEOp)

  update_paramcache!(cache.paramcache,get_nonlinear_operator(op),μ)
end

function Algebra.allocate_residual(
  op::LinearNonlinearParamOpFromFEOp,
  μ::Realization,
  u::AbstractVector,
  cache)

  b_lin = cache.b_lin
  copy(b_lin)
end

function Algebra.allocate_jacobian(
  op::LinearNonlinearParamOpFromFEOp,
  μ::Realization,
  u::AbstractVector,
  cache)

  A_lin = cache.A_lin
  copy(A_lin)
end

function Algebra.residual!(
  b,
  op::LinearNonlinearParamOpFromFEOp,
  μ::Realization,
  u::AbstractVector,
  cache)

  A_lin = cache.A_lin
  b_lin = cache.b_lin
  paramcache = cache.paramcache
  residual!(b,get_nonlinear_operator(op),μ,u,paramcache)
  mul!(b,A_lin,u,true,true)
  axpy!(1.0,b_lin,b)
  b
end

function ODEs.jacobian_add!(
  A,
  op::LinearNonlinearParamOpFromFEOp,
  μ::Realization,
  u::AbstractVector,
  cache)

  A_lin = cache.A_lin
  paramcache = cache.paramcache
  jacobian_add!(A,get_nonlinear_operator(op),μ,u,paramcache)
  axpy!(1.0,A_lin,A)
  A
end
