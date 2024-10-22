abstract type ParamOperatorType end

struct NonlinearParamEq <: ParamOperatorType end
struct LinearParamEq <: ParamOperatorType end
struct LinearNonlinearParamEq <: ParamOperatorType end

"""
    abstract type ParamOperator{T<:ParamOperatorType} <: NonlinearOperator end

Similar to [`ODEOperator`](@ref) in [`Gridap`](@ref), when dealing with steady
parametric problems

Subtypes:
- [`ParamOperatorWithTrian`](@ref)
- [`ParamOpFromFEOp`](@ref)

"""
abstract type ParamOperator{T<:ParamOperatorType} <: GridapType end

function allocate_paramcache(
  op::ParamOperator,
  r::Realization,
  u::AbstractVector)

  nothing
end

function update_paramcache!(
  paramcache,
  op::ParamOperator,
  r::Realization)

  paramcache
end

function Algebra.allocate_residual(
  op::ParamOperator,
  r::Realization,
  u::AbstractVector,
  paramcache)

  @abstractmethod
end

function Algebra.residual!(
  b::AbstractVector,
  op::ParamOperator,
  r::Realization,
  u::AbstractVector,
  paramcache)

  @abstractmethod
end

function Algebra.residual(
  op::ParamOperator,
  r::Realization,
  u::AbstractVector)

  paramcache = allocate_paramcache(op,r,u)
  b = allocate_residual(op,r,u,paramcache)
  residual!(b,op,r,u,paramcache)
  b
end

function Algebra.allocate_jacobian(
  op::ParamOperator,
  r::Realization,
  u::AbstractVector,
  paramcache)

  @abstractmethod
end

function Algebra.jacobian!(
  A::AbstractMatrix,
  op::ParamOperator,
  r::Realization,
  u::AbstractVector,
  paramcache)

  @abstractmethod
end

function Algebra.jacobian(
  op::ParamOperator,
  r::Realization,
  u::AbstractVector)

  paramcache = allocate_paramcache(op,r,u)
  A = allocate_jacobian(op,r,u,paramcache)
  jacobian!(A,op,r,u,paramcache)
  A
end

mutable struct ParamCache <: GridapType
  trial
  ptrial
  A
  b
end

"""
    abstract type ParamOperatorWithTrian{T<:ParamOperatorType} <: ParamOperator{T} end

Is to a ParamOperator as a ParamFEOperatorWithTrian is to a ParamFEOperator.

Suptypes:
- [`ParamOpFromFEOpWithTrian`](@ref)
- [`RBOperator`](@ref)

"""
abstract type ParamOperatorWithTrian{T<:ParamOperatorType} <: ParamOperator{T} end

function Algebra.allocate_residual(
  op::ParamOperatorWithTrian,
  r::Realization,
  u::AbstractVector,
  paramcache)

  @abstractmethod
end

function Algebra.residual!(
  b::Contribution,
  op::ParamOperatorWithTrian,
  r::Realization,
  u::AbstractVector,
  paramcache)

  @abstractmethod
end

function Algebra.allocate_jacobian(
  op::ParamOperator,
  r::ParamOperatorWithTrian,
  u::AbstractVector,
  paramcache)

  @abstractmethod
end

function Algebra.jacobian!(
  A::Contribution,
  op::ParamOperatorWithTrian,
  r::Realization,
  u::AbstractVector,
  paramcache)

  @abstractmethod
end

abstract type ParamNonlinearOperator <: NonlinearOperator end

struct GenericParamNonlinearOperator <: ParamNonlinearOperator
  op::ParamOperator
  r::Realization
  paramcache::ParamCache
end

function GenericParamNonlinearOperator(op::ParamOperator,r::Realization,u::AbstractVector)
  paramcache = allocate_paramcache(op,r,u)
  GenericParamNonlinearOperator(op,r,paramcache)
end

function Algebra.allocate_residual(
  nlop::GenericParamNonlinearOperator,
  x::AbstractVector)

  allocate_residual(nlop.op,nlop.r,x,nlop.paramcache)
end

function Algebra.residual!(
  b::AbstractVector,
  nlop::GenericParamNonlinearOperator,
  x::AbstractVector)

  residual!(b,nlop.op,nlop.r,x,nlop.paramcache)
end

function Algebra.allocate_jacobian(
  nlop::GenericParamNonlinearOperator,
  x::AbstractVector)

  allocate_jacobian(nlop.op,nlop.r,x,nlop.paramcache)
end

function Algebra.jacobian!(
  A::AbstractMatrix,
  nlop::GenericParamNonlinearOperator,
  x::AbstractVector)

  jacobian!(A,nlop.op,nlop.r,x,nlop.paramcache)
end
