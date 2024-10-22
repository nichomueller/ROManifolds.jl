"""
"""
struct ParamOpFromFEOp{T} <: ParamOperator{T}
  op::ParamFEOperator{T}
  μ::Realization
end

get_fe_operator(op::ParamOpFromFEOp) = op.op
get_realization(op::ParamOpFromFEOp) = op.μ

"""
    allocate_paramcache(op::ParamOpFromFEOp,μ::Realization,u::AbstractVector
      ) -> CacheType

Similar to [`allocate_odeparamcache`](@ref) in [`Gridap`](@ref), when dealing with steady
parametric problems

"""
function allocate_paramcache(
  op::ParamOpFromFEOp,
  u::AbstractVector)

  μ = get_realization(op)
  ptrial = get_trial(op.op)
  trial = evaluate(ptrial,μ)
  fe_cache = allocate_feopcache(op.feop,μ,u)

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
function update_paramcache!(paramcache,op::ParamOpFromFEOp)
  μ = get_realization(op)
  paramcache.trial = evaluate!(paramcache.trial,paramcache.ptrial,μ)
  paramcache
end

function Algebra.allocate_residual(
  op::ParamOpFromFEOp,
  u::AbstractVector,
  paramcache)

  μ = get_realization(op)

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
  u::AbstractVector,
  paramcache;
  add::Bool=false)

  !add && fill!(b,zero(eltype(b)))

  μ = get_realization(op)

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
  u::AbstractVector,
  paramcache)

  μ = get_realization(op)

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
  u::AbstractVector,
  paramcache)

  if is_jac_constant(op.op)
    return paramcache.const_forms
  end

  μ = get_realization(op)

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
struct ParamOpFromFEOpWithTrian{T} <: ParamOperatorWithTrian{T}
  op::ParamFEOperatorWithTrian{T}
  μ::Realization
end

get_fe_operator(op::ParamOpFromFEOpWithTrian) = op.op
get_realization(op::ParamOpFromFEOpWithTrian) = op.μ

function set_triangulation(op::ParamOpFromFEOpWithTrian,trians_rhs,trians_lhs)
  ParamOpFromFEOpWithTrian(set_triangulation(op.op,trians_rhs,trians_lhs))
end

function change_triangulation(op::ParamOpFromFEOpWithTrian,trians_rhs,trians_lhs)
  ParamOpFromFEOpWithTrian(change_triangulation(op.op,trians_rhs,trians_lhs))
end

function allocate_paramcache(
  op::ParamOpFromFEOpWithTrian,
  u::AbstractVector)

  μ = get_realization(op)
  ptrial = get_trial(op.op)
  trial = evaluate(ptrial,μ)
  fe_cache = allocate_feopcache(op.feop,μ,u)

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
  op::ParamOpFromFEOpWithTrian)

  μ = get_realization(op)
  paramcache.trial = evaluate!(paramcache.trial,paramcache.ptrial,μ)
  paramcache
end

function Algebra.allocate_residual(
  op::ParamOpFromFEOpWithTrian,
  u::AbstractVector,
  paramcache)

  μ = get_realization(op)

  res = get_res(op.op)
  dc = res(μ,uh,v)
  b = contribution(op.op.trian_res) do trian
    vecdata = collect_cell_vector_for_trian(test,dc,trian)
    allocate_vector(assem,vecdata)
  end

  b
end

function Algebra.residual!(
  b::Contribution,
  op::ParamOpFromFEOpWithTrian,
  u::AbstractVector,
  paramcache;
  add::Bool=false)

  !add && fill!(b,zero(eltype(b)))

  μ = get_realization(op)

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
end

function Algebra.allocate_jacobian(
  op::ParamOpFromFEOpWithTrian,
  u::AbstractVector,
  paramcache)

  μ = get_realization(op)

  uh = EvaluationFunction(paramcache.trial,u)
  trial = evaluate(get_trial(op.op),nothing)
  du = get_trial_fe_basis(trial)
  test = get_test(op.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op.op,μ)

  jac = get_jac(op.op)
  dc = jac(μ,uh,du,v)
  const_jac = contribution(op.op.trian_jac) do trian
    matdata = collect_cell_matrix_for_trian(trial,test,dc,trian)
    allocate_matrix(assem,matdata)
  end
end

function ODEs.jacobian_add!(
  A::Contribution,
  op::ParamOpFromFEOpWithTrian,
  u::AbstractVector,
  paramcache)

  if is_jac_constant(op.op)
    return paramcache.const_forms
  end

  μ = get_realization(op)

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

struct LinearCache{A,B}
  matrix::A
  vector::B
end

struct LinearNonlinearParamOpFromFEOp <: ParamOperator{LinearNonlinearParamEq}
  op::LinearNonlinearParamFEOperator
  μ::Realization
  lcache::LinearCache
end

function LinearNonlinearParamOpFromFEOp(
  op::LinearNonlinearParamFEOperator,
  μ::Realization)

  op_lin = get_linear_operator(op)
  algop_lin = get_algebraic_operator(op_lin,μ)
  paramcache = allocate_paramcache(algop_lin,u)
  u = zero_free_values(paramcache.trial)
  A = allocate_jacobian(algop_lin,u,paramcache)
  b = allocate_residual(algop_lin,u,paramcache)
  lcache = LinearCache(A,b)
  LinearNonlinearParamOpFromFEOp(op,μ,lcache)
end

get_fe_operator(op::LinearNonlinearParamOpFromFEOp) = op.op
get_realization(op::LinearNonlinearParamOpFromFEOp) = op.μ

function get_linear_operator(op::LinearNonlinearParamOpFromFEOp)
  get_algebraic_operator(get_linear_operator(op.op),get_realization(op))
end

function get_nonlinear_operator(op::LinearNonlinearParamOpFromFEOp)
  get_algebraic_operator(get_nonlinear_operator(op.op),get_realization(op))
end

function allocate_paramcache(
  op::LinearNonlinearParamOpFromFEOp,
  u::AbstractVector)

  allocate_paramcache(get_nonlinear_operator(op.op),u)
end

function update_paramcache!(
  paramcache,
  op::LinearNonlinearParamOpFromFEOp)

  update_paramcache!(paramcache,get_nonlinear_operator(op.op))
end

function Algebra.allocate_residual(
  op::LinearNonlinearParamOpFromFEOp,
  u::AbstractVector,
  paramcache)

  lcache = op.lcache
  copy(lcache.vector)
end

function Algebra.allocate_jacobian(
  op::LinearNonlinearParamOpFromFEOp,
  u::AbstractVector,
  paramcache)

  lcache = op.lcache
  copy(lcache.matrix)
end

function Algebra.residual!(
  b,
  op::LinearNonlinearParamOpFromFEOp,
  u::AbstractVector,
  paramcache)

  lcache = op.lcache
  copy_entries!(b,lcache.vector)
  residual!(b,get_nonlinear_operator(op.op),u,paramcache;add=true)
end

function ODEs.jacobian_add!(
  A,
  op::LinearNonlinearParamOpFromFEOp,
  u::AbstractVector,
  paramcache)

  lcache = op.lcache
  copy_entries!(A,lcache.matrix)
  jacobian_add!(A,get_nonlinear_operator(op.op),u,paramcache)
end
