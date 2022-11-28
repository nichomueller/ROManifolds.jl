abstract type ParamOperator{C} <: GridapType end
const AffineParamOperator = ParamOperator{Affine}

"""
A wrapper of `ParamFEOperator` that transforms it to `ParamOperator`, i.e.,
takes A(μ,uh,vh) and returns A(μ,uF), where uF represents the free values
of the `EvaluationFunction` uh
"""
struct ParamOpFromFEOp{C} <: ParamOperator{C}
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
  μ::Param)

  z = zero(eltype(A))
  LinearAlgebra.fillstored!(A,z)
  jacobian!(A,op.feop,μ,uh)
end

function _vector!(
  b::AbstractVector,
  op::ParamOpFromFEOp,
  uh,
  μ::Param)

  residual!(b,op.feop,μ,uh)
  b .*= -1.0
end

# Nonlinear

struct ParamNonlinearOperator{T} <: NonlinearOperator
  param_op::ParamOperator
  uh::T
  μ::Param
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
