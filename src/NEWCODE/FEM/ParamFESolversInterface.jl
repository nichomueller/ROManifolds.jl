abstract type ParamOp{C} <: GridapType end
const AffineParamOp = ParamOp{Affine}

"""
A wrapper of `ParamFEOperator` that transforms it to `ParamOp`, i.e.,
takes A(μ,uh,vh) and returns A(μ,uF), where uF represents the free values
of the `EvaluationFunction` uh
"""
struct ParamOpFromFEOp{C} <: ParamOp{C}
  feop::ParamFEOperator{C}
end

function Gridap.ODEs.TransientFETools.allocate_residual(op::ParamOpFromFEOp,uh)
  allocate_residual(op.feop,uh)
end

function Gridap.ODEs.TransientFETools.allocate_jacobian(op::ParamOpFromFEOp,uh)
  allocate_jacobian(op.feop,uh)
end

# Affine

function _allocate_matrix_and_vector(op::ParamOpFromFEOp,uh)
  b = allocate_residual(op,uh)
  A = allocate_jacobian(op,uh)
  A,b
end

function _matrix!(
  A::AbstractMatrix,
  op::ParamOpFromFEOp,
  uh,
  μ::AbstractVector)

  z = zero(eltype(A))
  LinearAlgebra.fillstored!(A,z)
  jacobian!(A,op.feop,μ,uh)
end

function _vector!(
  b::AbstractVector,
  op::ParamOpFromFEOp,
  uh,
  μ::AbstractVector)

  residual!(b,op.feop,μ,uh)
  b .*= -1.0
end

# Nonlinear

struct ParamNonlinearOperator{T} <: NonlinearOperator
  param_op::ParamOp
  uh::T
  μ::AbstractVector
  cache
end

function Gridap.ODEs.TransientFETools.residual!(
  b::AbstractVector,
  op::ParamNonlinearOperator,
  x::AbstractVector)

  feop = op.param_op.feop
  trial = get_trial(feop)
  u = EvaluationFunction(trial(op.μ),x)
  residual!(b,feop,op.μ,u)
end

function Gridap.ODEs.TransientFETools.jacobian!(
  A::AbstractMatrix,
  op::ParamNonlinearOperator,
  x::AbstractVector)

  feop = op.param_op.feop
  trial = get_trial(feop)
  u = EvaluationFunction(trial(op.μ),x)
  z = zero(eltype(A))
  LinearAlgebra.fillstored!(A,z)
  jacobian!(A,feop,op.μ,u)
end

function Gridap.ODEs.TransientFETools.allocate_residual(
  op::ParamNonlinearOperator,
  x::AbstractVector)

  feop = op.param_op.feop
  trial = get_trial(feop)
  u = EvaluationFunction(trial(op.μ),x)
  allocate_residual(feop,u)
end

function Gridap.ODEs.TransientFETools.allocate_jacobian(
  op::ParamNonlinearOperator,
  x::AbstractVector)

  feop = op.param_op.feop
  trial = get_trial(feop)
  u = EvaluationFunction(trial(op.μ),x)
  allocate_jacobian(feop,u)
end

# MDEIM snapshots generation interface

function _vecdata_residual(
  op::ParamFEOperator,
  sols::AbstractMatrix,
  params::Table,
  args...)

  trial = get_trial(op)
  test = get_test(op)
  dv = get_fe_basis(test)
  sol_μ = _as_function(sols,params)
  u(μ) = EvaluationFunction(trial(μ),sol_μ(μ))
  μ -> collect_cell_vector(test,op.res(μ,u(μ),dv))
end

function Gridap.ODEs.TransientFETools._matdata_jacobian(
  op::ParamFEOperator,
  sols::AbstractMatrix,
  params::Table,
  args...)

  trial = get_trial(op)
  test = get_test(op)
  dv = get_fe_basis(test)
  du = get_trial_fe_basis(trial(nothing))
  sol_μ = _as_function(sols,params)
  u(μ) = EvaluationFunction(trial(μ),sol_μ(μ))
  μ -> collect_cell_matrix(trial(μ),test,op.jac(μ,u(μ),dv,du))
end
