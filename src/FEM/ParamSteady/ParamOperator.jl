abstract type ParamOperatorType <: ODEOperatorType end

abstract type NonlinearParamOperatorType <: ODEOperatorType end
abstract type LinearParamOperatorType <: ODEOperatorType end
abstract type LinearNonlinearParamOperatorType <: ODEOperatorType end

struct NonlinearParamEq <: NonlinearParamOperatorType end
struct LinearParamEq <: LinearParamOperatorType end
struct LinearNonlinearParamEq <: LinearNonlinearParamOperatorType end

abstract type ParamOperator{T<:ParamOperatorType} <: NonlinearOperator end

function allocate_paramcache(
  op::ParamOperator,
  r::ParamRealization,
  u::AbstractParamVector)

  nothing
end

function update_paramcache!(
  paramcache,
  op::ParamOperator,
  r::ParamRealization)

  paramcache
end

function Algebra.allocate_residual(
  op::ParamOperator,
  r::ParamRealization,
  u::AbstractParamVector,
  paramcache)

  @abstractmethod
end

function Algebra.residual!(
  b::AbstractParamVector,
  op::ParamOperator,
  r::ParamRealization,
  u::AbstractParamVector,
  paramcache)

  @abstractmethod
end

function Algebra.residual(
  op::ParamOperator,
  r::ParamRealization,
  u::AbstractParamVector,
  paramcache)

  b = allocate_residual(op,r,u,paramcache)
  residual!(b,op,r,u,paramcache)
  b
end

function Algebra.allocate_jacobian(
  op::ParamOperator,
  r::ParamRealization,
  u::AbstractParamVector,
  paramcache)

  @abstractmethod
end

function Algebra.jacobian!(
  A::AbstractParamMatrix,
  op::ParamOperator,
  r::ParamRealization,
  u::AbstractParamVector,
  paramcache)

  @abstractmethod
end

function Algebra.jacobian(
  op::ParamOperator,
  r::ParamRealization,
  u::AbstractParamVector,
  paramcache)

  A = allocate_jacobian(op,r,u,paramcache)
  jacobian!(A,op,r,u,paramcache)
  A
end

mutable struct ParamOpFromFEOpCache <: GridapType
  Us
  Ups
  pfeopcache
  form
end

abstract type ParamOperatorWithTrian{T<:ParamOperatorType} <: ParamOperator{T} end

function Algebra.allocate_residual(
  op::ParamOperatorWithTrian,
  r::ParamRealization,
  u::AbstractParamVector,
  paramcache)

  @abstractmethod
end

function Algebra.residual!(
  b::Contribution,
  op::ParamOperatorWithTrian,
  r::ParamRealization,
  u::AbstractParamVector,
  paramcache)

  @abstractmethod
end

function Algebra.residual(
  op::ParamOperatorWithTrian,
  r::ParamRealization,
  u::AbstractParamVector,
  paramcache)

  b = allocate_residual(op,r,u,paramcache)
  residual!(b,op,r,u,paramcache)
  b
end

function Algebra.allocate_jacobian(
  op::ParamOperator,
  r::ParamOperatorWithTrian,
  u::AbstractParamVector,
  paramcache)

  @abstractmethod
end

function Algebra.jacobian!(
  A::Contribution,
  op::ParamOperatorWithTrian,
  r::ParamRealization,
  u::AbstractParamVector,
  paramcache)

  @abstractmethod
end

function Algebra.jacobian(
  op::ParamOperatorWithTrian,
  r::ParamRealization,
  u::AbstractParamVector,
  paramcache)

  A = allocate_jacobian(op,r,u,paramcache)
  jacobian!(A,op,r,u,paramcache)
  A
end
