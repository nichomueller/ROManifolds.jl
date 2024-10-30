"""
"""
struct ParamOpFromFEOp{T} <: ParamOperator{T}
  op::ParamFEOperator{T}
end

get_fe_operator(op::ParamOpFromFEOp) = op.op

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

  op_lin = get_linear_operator(op)
  op_nlin = get_nonlinear_operator(op)

  paramcache = allocate_paramcache(op_nlin,μ,u)
  A_lin,b_lin = allocate_systemcache(op_lin,μ,u,paramcache)

  return ParamOpSysCache(paramcache,A_lin,b_lin)
end

function update_paramcache!(
  cache,
  op::LinearNonlinearParamOpFromFEOp,
  μ::Realization)

  update_paramcache!(cache.paramcache,get_nonlinear_operator(op),μ)
end

function Algebra.allocate_residual(
  op::LinearNonlinearParamOpFromFEOp,
  μ::Realization,
  u::AbstractVector,
  cache)

  b_lin = cache.b
  copy(b_lin)
end

function Algebra.allocate_jacobian(
  op::LinearNonlinearParamOpFromFEOp,
  μ::Realization,
  u::AbstractVector,
  cache)

  A_lin = cache.A
  copy(A_lin)
end

function Algebra.residual!(
  b,
  op::LinearNonlinearParamOpFromFEOp,
  μ::Realization,
  u::AbstractVector,
  cache)

  A_lin = cache.A
  b_lin = cache.b
  paramcache = cache.paramcache
  residual!(b,get_nonlinear_operator(op),μ,u,paramcache)
  mul!(b,A_lin,u,1,1)
  axpy!(1,b_lin,b)
  b
end

function ODEs.jacobian_add!(
  A,
  op::LinearNonlinearParamOpFromFEOp,
  μ::Realization,
  u::AbstractVector,
  cache)

  A_lin = cache.A
  paramcache = cache.paramcache
  jacobian_add!(A,get_nonlinear_operator(op),μ,u,paramcache)
  axpy!(1,A_lin,A)
  A
end
