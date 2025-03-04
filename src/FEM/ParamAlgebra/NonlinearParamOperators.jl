"""
    abstract type NonlinearParamOperator <: NonlinearOperator end
"""
abstract type NonlinearParamOperator <: NonlinearOperator end

function Algebra.allocate_residual(
  nlop::NonlinearParamOperator,
  μ::Realization,
  x::AbstractVector,
  paramcache)

  @abstractmethod
end

function Algebra.residual!(
  b,
  nlop::NonlinearParamOperator,
  μ::Realization,
  x::AbstractVector,
  paramcache)

  @abstractmethod
end

function Algebra.residual(
  nlop::NonlinearParamOperator,
  μ::Realization,
  x::AbstractVector)

  paramcache = allocate_paramcache(nlop,μ,x)
  residual(nlop,μ,x,paramcache)
end

function Algebra.residual(
  nlop::NonlinearParamOperator,
  μ::Realization,
  x::AbstractVector,
  paramcache)

  b = allocate_residual(nlop,μ,x,paramcache)
  residual!(b,nlop,μ,x,paramcache)
  b
end

function Algebra.allocate_jacobian(
  nlop::NonlinearParamOperator,
  μ::Realization,
  x::AbstractVector,
  paramcache)

  @abstractmethod
end

function Algebra.jacobian!(
  A,
  nlop::NonlinearParamOperator,
  μ::Realization,
  x::AbstractVector,
  paramcache)

  LinearAlgebra.fillstored!(A,zero(eltype(A)))
  jacobian_add!(A,nlop,μ,x,paramcache)
  A
end

function Algebra.jacobian(
  nlop::NonlinearParamOperator,
  μ::Realization,
  x::AbstractVector)

  paramcache = allocate_paramcache(nlop,μ,x)
  jacobian(nlop,μ,x,paramcache)
end

function Algebra.jacobian(
  nlop::NonlinearParamOperator,
  μ::Realization,
  x::AbstractVector,
  paramcache)

  A = allocate_jacobian(nlop,μ,x,paramcache)
  jacobian!(A,nlop,μ,x,paramcache)
  A
end

# caches

"""
    abstract type AbstractParamCache <: GridapType end
"""
abstract type AbstractParamCache <: GridapType end

"""
    allocate_paramcache(paramcache::AbstractParamCache,nlop::NonlinearParamOperator,μ::AbstractRealization) -> AbstractParamCache

Similar to `allocate_odecache` in `Gridap`, when dealing with parametric problems
"""
function allocate_paramcache(paramcache::AbstractParamCache,nlop::NonlinearParamOperator,μ::AbstractRealization)
  @abstractmethod
end

"""
    update_paramcache!(paramcache::AbstractParamCache,nlop::NonlinearParamOperator,μ::AbstractRealization) -> AbstractParamCache

Similar to `update_odecache!` in `Gridap`, when dealing with parametric problems
"""
function update_paramcache!(paramcache::AbstractParamCache,nlop::NonlinearParamOperator,μ::AbstractRealization)
  @abstractmethod
end

reset_index!(nlop::NonlinearParamOperator) = @abstractmethod
next_index!(nlop::NonlinearParamOperator) = @abstractmethod
empty_matvecdata!(nlop::NonlinearParamOperator) = @abstractmethod

"""
    mutable struct ParamCache <: AbstractParamCache
      trial
      ptrial
    end
"""
mutable struct ParamCache <: AbstractParamCache
  trial
  ptrial
end

function update_paramcache!(c::ParamCache,nlop::NonlinearParamOperator,μ::Realization)
  c.trial = evaluate!(c.trial,c.ptrial,μ)
  c
end

"""
    struct GenericParamNonlinearOperator <: NonlinearParamOperator
      op::NonlinearParamOperator
      μ::Realization
      paramcache::ParamCache
    end

Fields:
- `op`: `NonlinearParamOperator` representing a parametric differential problem
- `μ`: `Realization` representing the parameters at which the problem is solved
- `paramcache`: cache of the problem
"""
struct GenericParamNonlinearOperator <: NonlinearParamOperator
  op::NonlinearParamOperator
  μ::Realization
  paramcache::AbstractParamCache
end

function ParamDataStructures.parameterize(op::NonlinearParamOperator,μ::Realization)
  paramcache = allocate_paramcache(op,μ)
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

# system caches

"""
    struct SystemCache
      A
      b
    end
"""
struct SystemCache
  A
  b
end

function allocate_systemcache(nlop::NonlinearParamOperator,x::AbstractVector)
  A = allocate_jacobian(nlop,x)
  b = allocate_residual(nlop,x)
  return SystemCache(A,b)
end
