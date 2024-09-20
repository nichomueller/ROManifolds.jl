"""
"""
struct ParamOpFromFEOp{T} <: ParamOperator{T}
  op::ParamFEOperator{T}
end

FESpaces.get_test(op::ParamOpFromFEOp) = get_test(op.op)
FESpaces.get_trial(op::ParamOpFromFEOp) = get_trial(op.op)
ParamDataStructures.realization(op::ParamOpFromFEOp;kwargs...) = realization(op.op;kwargs...)
get_fe_operator(op::ParamOpFromFEOp) = op.op
get_vector_index_map(op::ParamOpFromFEOp) = get_vector_index_map(op.op)
get_matrix_index_map(op::ParamOpFromFEOp) = get_matrix_index_map(op.op)

function get_linear_operator(op::ParamOpFromFEOp)
  ParamOpFromFEOp(get_linear_operator(op.op))
end

function get_nonlinear_operator(op::ParamOpFromFEOp)
  ParamOpFromFEOp(get_nonlinear_operator(op.op))
end

"""
    allocate_paramcache(op::ParamOpFromFEOp,r::Realization,u::AbstractParamVector
      ) -> CacheType

Similar to [`allocate_odecache`](@ref) in [`Gridap`](@ref), when dealing with steady
parametric problems

"""
function allocate_paramcache(
  op::ParamOpFromFEOp,
  r::Realization,
  u::AbstractParamVector)

  ptrial = get_trial(op.op)
  trial = allocate_space(ptrial,r)

  pfeopcache = allocate_pfeopcache(op.op,r,u)

  uh = EvaluationFunction(trial,u)
  test = get_test(op.op)
  v = get_fe_basis(test)
  trial = evaluate(get_trial(op.op),nothing)
  du = get_trial_fe_basis(trial)
  assem = get_param_assembler(op.op,r)

  jac = get_jac(op.op)
  μ = get_params(r)
  matdata = collect_cell_matrix(trial,test,jac(μ,uh,du,v))
  A = allocate_matrix(assem,matdata)

  OpFromFEOpCache(trial,ptrial,pfeopcache,A)
end

"""
    update_paramcache!(opcache, op::ParamOpFromFEOp, r::Realization, u::AbstractParamVector
      ) -> CacheType

Similar to [`update_odeopcache!`](@ref) in [`Gridap`](@ref), when dealing with steady
parametric problems

"""
function update_paramcache!(opcache,op::ParamOpFromFEOp,r::Realization)
  opcache.Us = evaluate!(opcache.Us,opcache.Ups,r)
  pfeopcache,op = opcache.pfeopcache,op.op
  opcache.pfeopcache = update_pfeopcache!(pfeopcache,op,r)
  opcache
end

function Algebra.allocate_residual(
  op::ParamOpFromFEOp,
  r::Realization,
  u::AbstractParamVector,
  opcache)

  uh = EvaluationFunction(opcache.Us,u)
  test = get_test(op.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op.op,r)

  μ = get_params(r)

  res = get_res(op.op)
  vecdata = collect_cell_vector(test,res(μ,uh,v))
  allocate_vector(assem,vecdata)
end

function Algebra.residual!(
  b::AbstractVector,
  op::ParamOpFromFEOp,
  r::Realization,
  u::AbstractParamVector,
  opcache)

  uh = EvaluationFunction(opcache.Us,u)
  test = get_test(op.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op.op,r)

  μ = get_params(r)
  res = get_res(op.op)
  dc = res(μ,uh,v)
  vecdata = collect_cell_vector(test,dc)
  assemble_vector!(b,assem,vecdata)

  b
end

function Algebra.allocate_jacobian(
  op::ParamOpFromFEOp,
  r::Realization,
  u::AbstractParamVector,
  opcache)

  uh = EvaluationFunction(opcache.Us,u)
  trial = evaluate(get_trial(op.op),nothing)
  du = get_trial_fe_basis(trial)
  test = get_test(op.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op.op,r)

  μ = get_params(r)
  jac = get_jac(op.op)
  dc = jac(μ,uh,du,v)

  matdata = collect_cell_matrix(trial,test,dc)
  allocate_matrix(assem,matdata)
end

function Algebra.jacobian!(
  A::AbstractMatrix,
  op::ParamOpFromFEOp,
  r::Realization,
  u::AbstractParamVector,
  opcache)

  uh = EvaluationFunction(opcache.Us,u)
  trial = evaluate(get_trial(op.op),nothing)
  du = get_trial_fe_basis(trial)
  test = get_test(op.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op.op,r)

  μ = get_params(r)
  jac = get_jac(op.op)
  dc = jac(μ,uh,du,v)
  matdata = collect_cell_matrix(trial,test,dc)
  assemble_matrix!(A,assem,matdata)

  A
end

"""
"""
struct ParamOpFromFEOpWithTrian{T} <: ParamOperatorWithTrian{T}
  op::ParamFEOperatorWithTrian{T}
end

FESpaces.get_test(op::ParamOpFromFEOpWithTrian) = get_test(op.op)
FESpaces.get_trial(op::ParamOpFromFEOpWithTrian) = get_trial(op.op)
ParamDataStructures.realization(op::ParamOpFromFEOpWithTrian;kwargs...) = realization(op.op;kwargs...)
get_fe_operator(op::ParamOpFromFEOpWithTrian) = op.op

function get_linear_operator(op::ParamOpFromFEOpWithTrian)
  ParamOpFromFEOpWithTrian(get_linear_operator(op.op))
end

function get_nonlinear_operator(op::ParamOpFromFEOpWithTrian)
  ParamOpFromFEOpWithTrian(get_nonlinear_operator(op.op))
end

function set_triangulation(op::ParamOpFromFEOpWithTrian,trians_rhs,trians_lhs)
  ParamOpFromFEOpWithTrian(set_triangulation(op.op,trians_rhs,trians_lhs))
end

function change_triangulation(op::ParamOpFromFEOpWithTrian,trians_rhs,trians_lhs)
  ParamOpFromFEOpWithTrian(change_triangulation(op.op,trians_rhs,trians_lhs))
end

function allocate_paramcache(
  op::ParamOpFromFEOpWithTrian,
  r::Realization,
  u::AbstractParamVector)

  ptrial = get_trial(op.op)
  trial = allocate_space(ptrial,r)

  pfeopcache = allocate_pfeopcache(op.op,r,u)

  uh = EvaluationFunction(trial,u)
  test = get_test(op.op)
  v = get_fe_basis(test)
  trial = evaluate(get_trial(op.op),nothing)
  du = get_trial_fe_basis(trial)
  assem = get_param_assembler(op.op,r)

  jac = get_jac(op.op)
  μ = get_params(r)
  dc = jac(μ,uh,du,v)
  matdata = collect_cell_matrix(trial,test,dc)
  A = allocate_matrix(assem,matdata)

  OpFromFEOpCache(trial,ptrial,pfeopcache,A)
end

function update_paramcache!(
  opcache,
  op::ParamOpFromFEOpWithTrian,
  r::Realization)

  opcache.Us = evaluate!(opcache.Us,opcache.Ups,r)
  pfeopcache,op = opcache.pfeopcache,op.op
  opcache.pfeopcache = update_pfeopcache!(pfeopcache,op,r)
  opcache
end

function Algebra.allocate_residual(
  op::ParamOpFromFEOpWithTrian,
  r::Realization,
  u::AbstractParamVector,
  opcache)

  uh = EvaluationFunction(opcache.Us,u)
  test = get_test(op.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op.op,r)

  μ = get_params(r)
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
  r::Realization,
  u::AbstractParamVector,
  opcache)

  uh = EvaluationFunction(opcache.Us,u)
  test = get_test(op.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op.op,r)

  μ = get_params(r)
  res = get_res(op.op)
  dc = res(μ,uh,v)

  map(b.values,op.op.trian_res) do values,trian
    vecdata = collect_cell_vector_for_trian(test,dc,trian)
    assemble_vector!(values,assem,vecdata)
  end
end

function Algebra.allocate_jacobian(
  op::ParamOpFromFEOpWithTrian,
  r::Realization,
  u::AbstractParamVector,
  opcache)

  uh = EvaluationFunction(opcache.Us,u)
  trial = evaluate(get_trial(op.op),nothing)
  du = get_trial_fe_basis(trial)
  test = get_test(op.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op.op,r)

  μ = get_params(r)
  jac = get_jac(op.op)
  dc = jac(μ,uh,du,v)
  contribution(op.op.trian_res) do trian
    matdata = collect_cell_matrix_for_trian(trial,test,dc,trian)
    allocate_matrix(assem,matdata)
  end
end

function Algebra.jacobian!(
  A::Contribution,
  op::ParamOpFromFEOpWithTrian,
  r::Realization,
  u::AbstractParamVector,
  opcache)

  uh = EvaluationFunction(opcache.Us,u)
  trial = evaluate(get_trial(op.op),nothing)
  du = get_trial_fe_basis(trial)
  test = get_test(op.op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op.op,r)

  μ = get_params(r)
  jac = get_jac(op.op)
  dc = jac(μ,uh,du,v)
  map(A.values,op.op.trian_jac) do values,trian
    matdata = collect_cell_matrix_for_trian(trial,test,dc,trian)
    assemble_matrix!(values,assem,matdata)
  end

  A
end
