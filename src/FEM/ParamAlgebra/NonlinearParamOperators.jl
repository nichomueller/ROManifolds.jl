"""
    abstract type NonlinearParamOperator <: NonlinearOperator end
"""
abstract type NonlinearParamOperator <: NonlinearOperator end

"""
    allocate_lazy_residual(
      nlop::NonlinearParamOperator,
      x::AbstractVector
      ) -> AbstractVector

Allocates a parametric residual in a lazy manner, one parameter at the time
"""
function allocate_lazy_residual(
  nlop::NonlinearParamOperator,
  x::AbstractVector)

  @abstractmethod
end

"""
    lazy_residual!(
      b::AbstractVector,
      nlop::NonlinearParamOperator,
      x::AbstractVector
      ) -> Nothing

Builds in-place a parametric residual in a lazy manner, one parameter at the time
"""
function lazy_residual!(
  b::AbstractVector,
  nlop::NonlinearParamOperator,
  x::AbstractVector)

  @abstractmethod
end

"""
    lazy_residual(
      nlop::NonlinearParamOperator,
      x::AbstractVector
      ) -> AbstractVector

Builds a parametric residual in a lazy manner, one parameter at the time
"""
function lazy_residual(nlop::NonlinearParamOperator,x::AbstractVector)
  b = allocate_lazy_residual(nlop,x)
  lazy_residual!(b,nlop,x)
end

"""
    allocate_lazy_jacobian(
      nlop::NonlinearParamOperator,
      x::AbstractVector
      ) -> AbstractMatrix

Allocates a parametric Jacobian in a lazy manner, one parameter at the time
"""
function allocate_lazy_jacobian(
  nlop::NonlinearParamOperator,
  x::AbstractVector)

  @abstractmethod
end

"""
    lazy_jacobian_add!(
      A::AbstractMatrix,
      nlop::NonlinearParamOperator,
      x::AbstractVector
      ) -> Nothing

Adds in-place the values of a parametric Jacobian in a lazy manner, one parameter at the time
"""
function lazy_jacobian!(
  A::AbstractMatrix,
  nlop::NonlinearParamOperator,
  x::AbstractVector)

  @abstractmethod
end

"""
    lazy_jacobian(
      nlop::NonlinearParamOperator,
      x::AbstractVector
      ) -> AbstractMatrix

Builds a parametric Jacobian in a lazy manner, one parameter at the time
"""
function lazy_jacobian(nlop::NonlinearParamOperator,x::AbstractVector)
  A = allocate_lazy_jacobian(nlop,x)
  lazy_jacobian!(A,nlop,x)
end

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

function allocate_lazy_residual(
  nlop::NonlinearParamOperator,
  μ::Realization,
  x::AbstractVector,
  paramcache)

  @abstractmethod
end

function lazy_residual!(
  b,
  nlop::NonlinearParamOperator,
  μ::Realization,
  x::AbstractVector,
  paramcache)

  @abstractmethod
end

function lazy_residual(
  nlop::NonlinearParamOperator,
  μ::Realization,
  x::AbstractVector)

  paramcache = allocate_paramcache(nlop,μ,x)
  lazy_residual(nlop,μ,x,paramcache)
end

function lazy_residual(
  nlop::NonlinearParamOperator,
  μ::Realization,
  x::AbstractVector,
  paramcache)

  b = allocate_lazy_residual(nlop,μ,x,paramcache)
  lazy_residual!(b,nlop,μ,x,paramcache)
  b
end

function allocate_lazy_jacobian(
  nlop::NonlinearParamOperator,
  μ::Realization,
  x::AbstractVector,
  paramcache)

  @abstractmethod
end

function lazy_jacobian_add!(
  A,
  nlop::NonlinearParamOperator,
  μ::Realization,
  x::AbstractVector,
  paramcache)

  @abstractmethod
end

function lazy_jacobian!(
  A,
  nlop::NonlinearParamOperator,
  μ::Realization,
  x::AbstractVector,
  paramcache)

  LinearAlgebra.fillstored!(A,zero(eltype(A)))
  lazy_jacobian_add!(A,nlop,μ,x,paramcache)
  A
end

function lazy_jacobian(
  nlop::NonlinearParamOperator,
  μ::Realization,
  x::AbstractVector)

  paramcache = allocate_paramcache(nlop,μ,x)
  lazy_jacobian(nlop,μ,x,paramcache)
end

function lazy_jacobian(
  nlop::NonlinearParamOperator,
  μ::Realization,
  x::AbstractVector,
  paramcache)

  A = allocate_lazy_jacobian(nlop,μ,x,paramcache)
  lazy_jacobian!(A,nlop,μ,x,paramcache)
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
    allocate_lazy_paramcache(paramcache::AbstractParamCache,nlop::NonlinearParamOperator,μ::AbstractRealization) -> AbstractParamCache

Similar to [`allocate_lazy_paramcache`](@ref), the cache allows to solve lazily
the parametric equation
"""
function allocate_lazy_paramcache(paramcache::AbstractParamCache,nlop::NonlinearParamOperator,μ::AbstractRealization)
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
    mutable struct LazyParamCache <: AbstractParamCache
      trial
      ptrial
      matdata
      vecdata
      index
    end
"""
mutable struct LazyParamCache <: AbstractParamCache
  trial
  ptrial
  matdata
  vecdata
  index
end

LazyParamCache(trial,ptrial,index::Int=1) = LazyParamCache(trial,ptrial,nothing,nothing,index)

get_matdata(c::LazyParamCache) = c.matdata
get_vecdata(c::LazyParamCache) = c.vecdata
isstored_matdata(c::LazyParamCache) = isnothing(c.matdata)
isstored_vecdata(c::LazyParamCache) = isnothing(c.vecdata)
fill_matdata!(c::LazyParamCache,matdata) = c.matdata=matdata
fill_vecdata!(c::LazyParamCache,vecdata) = c.vecdata=vecdata
empty_matdata!(c::LazyParamCache) = c.matdata=nothing
empty_vecdata!(c::LazyParamCache) = c.vecdata=nothing
function empty_matvecdata!(c::LazyParamCache)
  empty_matdata!(c)
  empty_vecdata!(c)
end

next_index!(c::LazyParamCache) = c.index+=1
reset_index!(c::LazyParamCache) = c.index=1

function update_paramcache!(c::LazyParamCache,nlop::NonlinearParamOperator,μ::Realization)
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

"""
    struct LazyParamNonlinearOperator <: NonlinearParamOperator
      op::NonlinearParamOperator
      μ::Realization
      paramcache::LazyParamCache
    end

Fields:
- `op`: `NonlinearParamOperator` representing a parametric differential problem
- `μ`: `Realization` representing the parameters at which the problem is solved
- `paramcache`: cache of the problem
"""
struct LazyParamNonlinearOperator <: NonlinearParamOperator
  op::NonlinearParamOperator
  μ::Realization
  paramcache::LazyParamCache
end

function ParamDataStructures.lazy_parameterize(op::NonlinearParamOperator,μ::Realization)
  paramcache = allocate_lazy_paramcache(op,μ)
  LazyParamNonlinearOperator(op,μ,paramcache)
end

function Algebra.allocate_residual(
  nlop::LazyParamNonlinearOperator,
  x::AbstractVector)

  allocate_lazy_residual(nlop.op,nlop.μ,x,nlop.paramcache)
end

function Algebra.residual!(
  b::AbstractVector,
  nlop::LazyParamNonlinearOperator,
  x::AbstractVector)

  lazy_residual!(b,nlop.op,nlop.μ,x,nlop.paramcache)
end

function Algebra.allocate_jacobian(
  nlop::LazyParamNonlinearOperator,
  x::AbstractVector)

  allocate_lazy_jacobian(nlop.op,nlop.μ,x,nlop.paramcache)
end

function Algebra.jacobian!(
  A::AbstractMatrix,
  nlop::LazyParamNonlinearOperator,
  x::AbstractVector)

  lazy_jacobian!(A,nlop.op,nlop.μ,x,nlop.paramcache)
end

reset_index!(nlop::LazyParamNonlinearOperator) = reset_index!(nlop.paramcache)
next_index!(nlop::LazyParamNonlinearOperator) = next_index!(nlop.paramcache)
empty_matvecdata!(nlop::LazyParamNonlinearOperator) = empty_matvecdata!(nlop.paramcache)

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
