# caches

"""
    abstract type AbstractParamCache <: GridapType end
"""
abstract type AbstractParamCache <: GridapType end

"""
    update_paramcache!(paramcache::AbstractParamCache,op::ParamOperator,μ::AbstractRealization) -> AbstractParamCache

Similar to `update_odecache!` in `Gridap`, when dealing with
parametric problems
"""
function update_paramcache!(paramcache::AbstractParamCache,op::ParamOperator,μ::AbstractRealization)
  @abstractmethod
end

"""
    mutable struct ParamOpCache <: AbstractParamCache
      trial
      ptrial
    end
"""
mutable struct ParamOpCache <: AbstractParamCache
  trial
  ptrial
end

"""
    allocate_paramcache(op::ParamOperator,μ::Realization,u::AbstractVector
      ) -> ParamOpCache

Similar to `allocate_odecache` in `Gridap`, when dealing with
parametric problems
"""
function allocate_paramcache(
  op::ParamOperator,
  μ::Realization,
  u::AbstractVector)

  feop = get_fe_operator(op)
  ptrial = get_trial(feop)
  trial = evaluate(ptrial,μ)
  ParamOpCache(trial,ptrial)
end

function update_paramcache!(paramcache::ParamOpCache,op::ParamOperator,μ::Realization)
  paramcache.trial = evaluate!(paramcache.trial,paramcache.ptrial,μ)
  paramcache
end

"""
    mutable struct AssemCache{A,B}
      matdata::A
      vecdata::B
    end
"""
mutable struct AssemCache{A,B}
  matdata::A
  vecdata::B
end

get_matdata(c::AssemCache) = c.matdata
get_vecdata(c::AssemCache) = c.vecdata
isstored_matdata(c::AssemCache) = isnothing(c.matdata)
isstored_vecdata(c::AssemCache) = isnothing(c.vecdata)
fill_matdata!(c::AssemCache,matdata) = (c.matdata=matdata)
fill_vecdata!(c::AssemCache,vecdata) = (c.vecdata=vecdata)
empty_matdata!(c::AssemCache) = (c.matdata=nothing)
empty_vecdata!(c::AssemCache) = (c.vecdata=nothing)
empty_matvecdata!(c::AssemCache) = empty_matdata!(c); empty_vecdata!(c)

"""
    mutable struct LazyParamOpCache <: AbstractParamCache
      index
      trial
      ptrial
      assemcache
    end
"""
mutable struct LazyParamOpCache <: AbstractParamCache
  trial
  ptrial
  index
  assemcache::AssemCache
end

"""
    allocate_lazy_paramcache(op::ParamOperator,μ::Realization,u::AbstractVector
      ) -> LazyParamOpCache

Solves a parametric problem by lazily iterating over the parameters
"""
function allocate_lazy_paramcache(
  op::ParamOperator,
  μ::Realization,
  u::AbstractVector)

  index = 1
  feop = get_fe_operator(op)
  ptrial = get_trial(feop)
  trial = evaluate(ptrial,μ)
  LazyParamOpCache(trial,ptrial,index)
end

function update_paramcache!(paramcache::LazyParamOpCache,op::ParamOperator,μ::Realization)
  paramcache.trial = evaluate!(paramcache.trial,paramcache.ptrial,μ)
  paramcache
end

get_matdata(c::LazyParamOpCache) = get_matdata(c.assemcache)
get_vecdata(c::LazyParamOpCache) = get_vecdata(c.assemcache)
isstored_matdata(c::LazyParamOpCache) = isstored_matdata(c.assemcache)
isstored_vecdata(c::LazyParamOpCache) = isstored_vecdata(c.assemcache)
fill_matdata!(c::LazyParamOpCache,matdata) = fill_matdata!(c.assemcache,matdata)
fill_vecdata!(c::LazyParamOpCache,vecdata) = fill_vecdata!(c.assemcache,vecdata)
empty_matdata!(c::LazyParamOpCache) = empty_matdata!(c.assemcache)
empty_vecdata!(c::LazyParamOpCache) = empty_vecdata!(c.assemcache)
empty_matvecdata!(c::LazyParamOpCache) = empty_matvecdata!(c.assemcache)

next_index!(paramcache::LazyParamOpCache) = (paramcache.index += 1)
reset_index!(paramcache::LazyParamOpCache) = (paramcache.index = 1)

"""
    struct ParamOpSysCache <: AbstractParamCache
      paramcache
      A
      b
    end
"""
struct ParamOpSysCache <: AbstractParamCache
  paramcache
  A
  b
end

function allocate_systemcache(
  op::ParamOperator{LinearParamEq},
  μ::Realization,
  u::AbstractVector,
  paramcache)

  u0 = copy(u)
  fill!(u0,zero(eltype(u0)))
  A = jacobian(op,μ,u0,paramcache)
  b = residual(op,μ,u0,paramcache)
  return A,b
end

function update_paramcache!(paramcache::ParamOpSysCache,op::ParamOperator,μ::Realization)
  update_paramcache!(paramcache.paramcache,op,μ)
end

function next_index!(paramcache::ParamOpSysCache,op::ParamOperator,μ::Realization)
  next_index!(paramcache.paramcache,op,μ)
end

# operators

"""
    abstract type ParamNonlinearOperator <: NonlinearOperator end
"""
abstract type ParamNonlinearOperator <: NonlinearOperator end

"""
    allocate_lazy_residual(
      nlop::ParamNonlinearOperator,
      x::AbstractVector
      ) -> AbstractVector

Allocates a parametric residual in a lazy manner, one parameter at the time
"""
function allocate_lazy_residual(
  nlop::ParamNonlinearOperator,
  x::AbstractVector)

  @abstractmethod
end

"""
    lazy_residual!(
      b::AbstractVector,
      nlop::ParamNonlinearOperator,
      x::AbstractVector
      ) -> Nothing

Builds in-place a parametric residual in a lazy manner, one parameter at the time
"""
function lazy_residual!(
  b::AbstractVector,
  nlop::ParamNonlinearOperator,
  x::AbstractVector)

  @abstractmethod
end

"""
    lazy_residual(
      nlop::ParamNonlinearOperator,
      x::AbstractVector
      ) -> AbstractVector

Builds a parametric residual in a lazy manner, one parameter at the time
"""
function lazy_residual(nlop::ParamNonlinearOperator,x::AbstractVector)
  b = allocate_lazy_residual(nlop,x)
  lazy_residual!(b,nlop,x)
end

"""
    allocate_lazy_jacobian(
      nlop::ParamNonlinearOperator,
      x::AbstractVector
      ) -> AbstractMatrix

Allocates a parametric Jacobian in a lazy manner, one parameter at the time
"""
function allocate_lazy_jacobian(
  nlop::ParamNonlinearOperator,
  x::AbstractVector)

  @abstractmethod
end

"""
    lazy_jacobian_add!(
      A::AbstractMatrix,
      nlop::ParamNonlinearOperator,
      x::AbstractVector
      ) -> Nothing

Adds in-place the values of a parametric Jacobian in a lazy manner, one parameter at the time
"""
function lazy_jacobian!(
  A::AbstractMatrix,
  nlop::ParamNonlinearOperator,
  x::AbstractVector)

  @abstractmethod
end

"""
    lazy_jacobian(
      nlop::ParamNonlinearOperator,
      x::AbstractVector
      ) -> AbstractMatrix

Builds a parametric Jacobian in a lazy manner, one parameter at the time
"""
function lazy_jacobian(nlop::ParamNonlinearOperator,x::AbstractVector)
  A = allocate_lazy_jacobian(nlop,x)
  lazy_jacobian!(A,nlop,x)
end

"""
    struct GenericParamNonlinearOperator <: ParamNonlinearOperator
      op::ParamOperator
      μ::Realization
      paramcache::AbstractParamCache
    end

Fields:
- `op`: `ParamOperator` representing a parametric differential problem
- `μ`: `Realization` representing the parameters at which the problem is solved
- `paramcache`: cache of the problem
"""
struct GenericParamNonlinearOperator <: ParamNonlinearOperator
  op::ParamOperator
  μ::Realization
  paramcache::AbstractParamCache
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
    struct LazyParamNonlinearOperator <: ParamNonlinearOperator
      op::ParamOperator
      μ::Realization
      paramcache::LazyParamCache
    end

Fields:
- `op`: `ParamOperator` representing a parametric differential problem
- `μ`: `Realization` representing the parameters at which the problem is solved
- `paramcache`: cache of the problem
"""
struct LazyParamNonlinearOperator <: ParamNonlinearOperator
  op::ParamOperator
  μ::Realization
  paramcache::LazyParamCache
end

next_index!(nlop::LazyParamNonlinearOperator) = next_index!(nlop.paramcache)
reset_index!(nlop::LazyParamNonlinearOperator) = reset_index!(nlop.paramcache)

function Algebra.allocate_residual(
  nlop::LazyParamNonlinearOperator,
  x::AbstractVector)

  allocate_lazy_residual(nlop,x)
end

function Algebra.residual!(
  b::AbstractVector,
  nlop::LazyParamNonlinearOperator,
  x::AbstractVector)

  lazy_residual!(b,nlop,x)
end

function Algebra.allocate_jacobian(
  nlop::LazyParamNonlinearOperator,
  x::AbstractVector)

  allocate_lazy_jacobian(nlop,x)
end

function Algebra.jacobian!(
  A::AbstractMatrix,
  nlop::LazyParamNonlinearOperator,
  x::AbstractVector)

  lazy_jacobian!(A,nlop,x)
end
