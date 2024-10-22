abstract type UnEvalOperatorType <: GridapType end

struct LinearParamEq <: UnEvalOperatorType end
struct NonlinearParamEq <: UnEvalOperatorType end
struct LinearNonlinearParamEq <: UnEvalOperatorType end

"""
    abstract type ParamOperator{T<:UnEvalOperatorType} <: NonlinearOperator end

Similar to [`ODEOperator`](@ref) in [`Gridap`](@ref), when dealing with steady
parametric problems

Subtypes:
- [`ParamOperatorWithTrian`](@ref)
- [`ParamOpFromFEOp`](@ref)

"""
abstract type ParamOperator{T<:UnEvalOperatorType} <: NonlinearOperator end

get_realization(op::ParamOperator) = @abstractmethod

function allocate_paramcache(
  op::ParamOperator,
  u::AbstractVector)

  nothing
end

function update_paramcache!(
  paramcache,
  op::ParamOperator,
  u::AbstractVector)

  paramcache
end

function Algebra.allocate_residual(
  op::ParamOperator,
  u::AbstractVector,
  paramcache)

  @abstractmethod
end

function Algebra.residual!(
  b::AbstractVector,
  op::ParamOperator,
  u::AbstractVector,
  paramcache)

  @abstractmethod
end

function Algebra.residual(
  op::ParamOperator,
  u::AbstractVector)

  paramcache = allocate_paramcache(op,u)
  b = allocate_residual(op,u,paramcache)
  residual!(b,op,u,paramcache)
  b
end

function Algebra.allocate_jacobian(
  op::ParamOperator,
  u::AbstractVector,
  paramcache)

  @abstractmethod
end

function Algebra.jacobian!(
  A::AbstractMatrix,
  op::ParamOperator,
  u::AbstractVector,
  paramcache)

  LinearAlgebra.fillstored!(A,zero(eltype(A)))
  jacobian_add!(A,op,u,paramcache)
  A
end

function Algebra.jacobian(
  op::ParamOperator,
  u::AbstractVector)

  paramcache = allocate_paramcache(op,u)
  A = allocate_jacobian(op,u,paramcache)
  jacobian!(A,op,u,paramcache)
  A
end

FESpaces.get_test(op::ParamOperator) = get_test(get_fe_operator(op))
FESpaces.get_trial(op::ParamOperator) = get_trial(get_fe_operator(op))
IndexMaps.get_vector_index_map(op::ParamOperator) = get_vector_index_map(get_fe_operator(op))
IndexMaps.get_matrix_index_map(op::ParamOperator) = get_matrix_index_map(get_fe_operator(op))

mutable struct ParamCache <: GridapType
  trial
  ptrial
  feop_cache
  const_forms
end

"""
    abstract type ParamOperatorWithTrian{T<:UnEvalOperatorType} <: ParamOperator{T} end

Is to a ParamOperator as a ParamFEOperatorWithTrian is to a ParamFEOperator.

Suptypes:
- [`ParamOpFromFEOpWithTrian`](@ref)
- [`RBOperator`](@ref)

"""
abstract type ParamOperatorWithTrian{T<:UnEvalOperatorType} <: ParamOperator{T} end

function Algebra.allocate_residual(
  op::ParamOperatorWithTrian,
  u::AbstractVector,
  paramcache)

  @abstractmethod
end

function Algebra.residual!(
  b::Contribution,
  op::ParamOperatorWithTrian,
  u::AbstractVector,
  paramcache)

  @abstractmethod
end

function Algebra.allocate_jacobian(
  op::ParamOperator,
  u::AbstractVector,
  paramcache)

  @abstractmethod
end

function Algebra.jacobian!(
  A::Contribution,
  op::ParamOperatorWithTrian,
  u::AbstractVector,
  paramcache)

  LinearAlgebra.fillstored!(A,zero(eltype(A)))
  jacobian_add!(A,op,u,paramcache)
  A
end
