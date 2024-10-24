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
  residual(op,μ,u,paramcache)
end

function Algebra.residual(
  op::ParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  b = allocate_residual(op,μ,u,paramcache)
  residual!(b,op,μ,u,paramcache)
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
  jacobian(op,μ,u,paramcache)
end

function Algebra.jacobian(
  op::ParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  A = allocate_jacobian(op,μ,u,paramcache)
  jacobian!(A,op,μ,u,paramcache)
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

struct ParamSystemCache{Ta,Tb} <: AbstractParamCache
  paramcache::ParamCache
  A::Ta
  b::Tb
end

abstract type ParamNonlinearOperator <: NonlinearOperator end

struct GenericParamNonlinearOperator <: NonlinearOperator
  op::ParamOperator
  μ::Realization
  paramcache::AbstractParamCache
end

function ParamNonlinearOperator(op::ParamOperator,μ::Realization)
  trial = get_trial(op)(μ)
  u = zero_free_values(trial)
  paramcache = allocate_paramcache(op,μ,u)
  GenericParamNonlinearOperator(op,μ,paramcache)
end

function Algebra.allocate_residual(
  nlop::GenericParamNonlinearOperator,
  x::AbstractVector)

  allocate_residual(nlop.op,nlop.μ,x,nlop.paramcache)
end

function Algebra.residual!(
  b::AbstractVector,
  nlop::GenericParamNonlinearOperator,
  x::AbstractVector)

  residual!(b,nlop.op,nlop.μ,x,nlop.paramcache)
end

function Algebra.allocate_jacobian(
  nlop::GenericParamNonlinearOperator,
  x::AbstractVector)

  allocate_jacobian(nlop.op,nlop.μ,x,nlop.paramcache)
end

function Algebra.jacobian!(
  A::AbstractMatrix,
  nlop::GenericParamNonlinearOperator,
  x::AbstractVector)

  jacobian!(A,nlop.op,nlop.μ,x,nlop.paramcache)
end
