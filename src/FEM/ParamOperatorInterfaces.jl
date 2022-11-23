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
  Gridap.ODEs.TransientFETools.allocate_residual(op.feop,uh)
end

function Gridap.ODEs.TransientFETools.allocate_jacobian(op::ParamOpFromFEOp,uh)
  Gridap.ODEs.TransientFETools.allocate_jacobian(op.feop,uh)
end

function _allocate_matrix_and_vector(op::ParamOpFromFEOp,uh)
  b = Gridap.ODEs.TransientFETools.allocate_residual(op,uh)
  A = Gridap.ODEs.TransientFETools.allocate_jacobian(op,uh)
  A,b
end

function _matrix!(
  A::AbstractMatrix,
  op::ParamOpFromFEOp,
  uh,
  μ::Vector{Float})

  z = zero(eltype(A))
  LinearAlgebra.fillstored!(A,z)
  Gridap.ODEs.TransientFETools.jacobian!(A,op.feop,μ,uh)
end

function _vector!(
  b::AbstractVector,
  op::ParamOpFromFEOp,
  uh,
  μ::Vector{Float})

  Gridap.ODEs.TransientFETools.residual!(b,op.feop,μ,uh)
  b .*= -1.0
end

struct ParamNonlinearOperator{T} <: Gridap.Algebra.NonlinearOperator
  param_op::ParamOperator
  uh::T
  μ::Vector{Float}
  cache
end

function Gridap.ODEs.TransientFETools.residual!(
  b::AbstractVector,
  op::ParamNonlinearOperator,
  x::AbstractVector)

  feop = op.param_op.feop
  trial = Gridap.ODEs.TransientFETools.get_trial(feop)
  u = EvaluationFunction(trial(op.μ),x)
  Gridap.ODEs.TransientFETools.residual!(b,feop,op.μ,u)
end

function Gridap.ODEs.TransientFETools.jacobian!(
  A::AbstractMatrix,
  op::ParamNonlinearOperator,
  x::AbstractVector)

  feop = op.param_op.feop
  trial = Gridap.ODEs.TransientFETools.get_trial(feop)
  u = EvaluationFunction(trial(op.μ),x)
  z = zero(eltype(A))
  LinearAlgebra.fillstored!(A,z)
  Gridap.ODEs.TransientFETools.jacobian!(A,feop,op.μ,u)
end

function Gridap.ODEs.TransientFETools.allocate_residual(
  op::ParamNonlinearOperator,
  x::AbstractVector)

  feop = op.param_op.feop
  trial = Gridap.ODEs.TransientFETools.get_trial(feop)
  u = EvaluationFunction(trial(op.μ),x)
  Gridap.ODEs.TransientFETools.allocate_residual(feop,u)
end

function Gridap.ODEs.TransientFETools.allocate_jacobian(
  op::ParamNonlinearOperator,
  x::AbstractVector)

  feop = op.param_op.feop
  trial = Gridap.ODEs.TransientFETools.get_trial(feop)
  u = EvaluationFunction(trial(op.μ),x)
  Gridap.ODEs.TransientFETools.allocate_jacobian(feop,u)
end

function zero_initial_guess(op::ParamNonlinearOperator)
  xh = similar(op.uh)
  fill!(xh,zero(eltype(xh)))
  xh
end













abstract type ParamODEOperator{C} <: ODEOperator{C} end
const AffineParamODEOperator = ParamODEOperator{<:Affine}

"""
A wrapper of `ParamTransientFEOperator` that transforms it to `ParamODEOperator`, i.e.,
takes A(μ,t,uh,∂tuh,∂t^2uh,...,∂t^Nuh,vh) and returns A(μ,t,uF,∂tuF,...,∂t^NuF)
where uF,∂tuF,...,∂t^NuF represent the free values of the `EvaluationFunction`
uh,∂tuh,∂t^2uh,...,∂t^Nuh.
"""
struct ParamODEOpFromFEOp{C} <: ParamODEOperator{C}
  feop::ParamTransientFEOperator{C}
end
