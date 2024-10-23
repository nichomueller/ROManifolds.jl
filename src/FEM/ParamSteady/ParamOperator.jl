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

get_fe_operator(op::ParamOperator) = @abstractmethod

function allocate_paramcache(
  op::ParamOperator,
  μ::Realization,
  u::AbstractVector)

  nothing
end

function update_paramcache!(
  paramcache,
  op::ParamOperator,
  μ::Realization,
  u::AbstractVector)

  paramcache
end

function Algebra.allocate_residual(
  op::ParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  @abstractmethod
end

function Algebra.residual!(
  b,
  op::ParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  @abstractmethod
end

function Algebra.residual(
  op::ParamOperator,
  μ::Realization,
  u::AbstractVector)

  paramcache = allocate_paramcache(op,μ,u)
  b = allocate_residual(op,μ,u,paramcache)
  residual!(b,op,μ,u,paramcache)
  b
end

function Algebra.allocate_jacobian(
  op::ParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  @abstractmethod
end

function Algebra.jacobian!(
  A,
  op::ParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  LinearAlgebra.fillstored!(A,zero(eltype(A)))
  jacobian_add!(A,op,μ,u,paramcache)
  A
end

function Algebra.jacobian(
  op::ParamOperator,
  μ::Realization,
  u::AbstractVector)

  paramcache = allocate_paramcache(op,μ,u)
  A = allocate_jacobian(op,μ,u,paramcache)
  jacobian!(A,op,μ,u,paramcache)
  A
end

FESpaces.get_test(op::ParamOperator) = get_test(get_fe_operator(op))
FESpaces.get_trial(op::ParamOperator) = get_trial(get_fe_operator(op))
IndexMaps.get_vector_index_map(op::ParamOperator) = get_vector_index_map(get_fe_operator(op))
IndexMaps.get_matrix_index_map(op::ParamOperator) = get_matrix_index_map(get_fe_operator(op))

abstract type AbstractParamCache <: GridapType end

mutable struct ParamCache <: AbstractParamCache
  trial
  ptrial
  feop_cache
  const_forms
end

struct LinearNonlinearParamCache{A,B} <: AbstractParamCache
  paramcache::ParamCache
  A_lin::A
  b_lin::B
end

struct ParamNonlinearOperator <: NonlinearOperator
  op::ParamOperator
  r::Realization
  paramcache
end

function ParamNonlinearOperator(op::ParamOperator,μ::Realization)
  trial = get_trial(op)(r)
  u = zero_free_values(trial)
  paramcache = allocate_paramcache(op,μ,u)
  ParamNonlinearOperator(op,μ,paramcache)
end

function Algebra.allocate_residual(
  nlop::ParamNonlinearOperator,
  x::AbstractVector)

  allocate_residual(nlop.op,nlop.r,x,nlop.paramcache)
end

function Algebra.residual!(
  b::AbstractVector,
  nlop::ParamNonlinearOperator,
  x::AbstractVector)

  residual!(b,nlop.op,nlop.r,x,nlop.paramcache)
end

function Algebra.allocate_jacobian(
  nlop::ParamNonlinearOperator,
  x::AbstractVector)

  allocate_jacobian(nlop.op,nlop.r,x,nlop.paramcache)
end

function Algebra.jacobian!(
  A::AbstractMatrix,
  nlop::ParamNonlinearOperator,
  x::AbstractVector)

  jacobian!(A,nlop.op,nlop.r,x,nlop.paramcache)
end
