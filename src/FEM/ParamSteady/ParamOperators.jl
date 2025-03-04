"""
    abstract type UnEvalOperatorType <: GridapType end

Type representing operators that are not evaluated yet. This may include operators
representing transient problems (although the implementation in `Gridap`
differs), parametric problems, and a combination thereof. Could become a supertype
of `ODEOperatorType` in `Gridap`
"""
abstract type UnEvalOperatorType <: GridapType end

"""
    struct LinearParamEq <: UnEvalOperatorType end
"""
struct LinearParamEq <: UnEvalOperatorType end

"""
    struct NonlinearParamEq <: UnEvalOperatorType end
"""
struct NonlinearParamEq <: UnEvalOperatorType end

"""
    struct LinearNonlinearParamEq <: UnEvalOperatorType end
"""
struct LinearNonlinearParamEq <: UnEvalOperatorType end

"""
    abstract type TriangulationStyle <: GridapType end

Subtypes:

- [`JointDomains`](@ref)
- [`SplitDomains`](@ref)
"""
abstract type TriangulationStyle <: GridapType end

"""
    struct JointDomains <: TriangulationStyle end

Trait for a FE operator indicating that residuals/Jacobiansin this operator
should be computed summing the contributions relative to each triangulation as
occurs in `Gridap`
"""
struct JointDomains <: TriangulationStyle end

"""
    struct SplitDomains <: TriangulationStyle end

Trait for a FE operator indicating that residuals/Jacobiansin this operator
should be computed keeping the contributions relative to each triangulation separate
"""
struct SplitDomains <: TriangulationStyle end

"""
    abstract type ParamOperator{O<:UnEvalOperatorType,T<:TriangulationStyle} <: NonlinearOperator end

Type representing algebraic operators (i.e. `NonlinearOperator` in `Gridap`) when
solving parametric differential problems
"""
abstract type ParamOperator{O<:UnEvalOperatorType,T<:TriangulationStyle} <: NonlinearOperator end

"""
    get_fe_operator(op::ParamOperator) -> ParamFEOperator

Fetches the underlying FE operator of an algebraic operator `op`
"""
get_fe_operator(op::ParamOperator) = @abstractmethod

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
  jacobian(op,μ,u,paramcache)
end

function Algebra.jacobian(
  op::ParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  A = allocate_jacobian(op,μ,u,paramcache)
  jacobian!(A,op,μ,u,paramcache)
  A
end

"""
    allocate_lazy_residual(
      op::ParamOperator,
      μ::Realization,
      u::AbstractVector,
      paramcache
      ) -> AbstractVector

Allocates a parametric residual in a lazy manner, one parameter at the time
"""
function allocate_lazy_residual(
  op::ParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  @abstractmethod
end

"""
    lazy_residual!(
      b,
      op::ParamOperator,
      μ::Realization,
      u::AbstractVector,
      paramcache
      ) -> Nothing

Builds in-place a parametric residual in a lazy manner, one parameter at the time
"""
function lazy_residual!(
  b,
  op::ParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  @abstractmethod
end

"""
    lazy_residual(
      op::ParamOperator,
      μ::Realization,
      u::AbstractVector,
      paramcache
      ) -> AbstractVector

Builds a parametric residual in a lazy manner, one parameter at the time
"""
function lazy_residual(
  op::ParamOperator,
  μ::Realization,
  u::AbstractVector)

  index = 1
  paramcache = allocate_paramcache(op,μ,u,index)
  residual(op,μ,u,paramcache)
end

function lazy_residual(
  op::ParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  b = lazy_allocate_residual(op,μ,u,paramcache)
  lazy_residual!(b,op,μ,u,paramcache)
  b
end

"""
    allocate_lazy_jacobian(
      op::ParamOperator,
      μ::Realization,
      u::AbstractVector,
      paramcache
      ) -> AbstractVector

Allocates a parametric Jacobian in a lazy manner, one parameter at the time
"""
function allocate_lazy_jacobian(
  op::ParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  @abstractmethod
end

"""
    lazy_jacobian_add!(
      A,
      op::ParamOperator,
      μ::Realization,
      u::AbstractVector,
      paramcache
      ) -> AbstractVector

Adds in-place the values of a parametric Jacobian in a lazy manner, one parameter at the time
"""
function lazy_jacobian_add!(
  A,
  op::ParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  @abstractmethod
end

"""
    lazy_jacobian!(
      A,
      op::ParamOperator,
      μ::Realization,
      u::AbstractVector,
      paramcache
      ) -> Nothing

Builds in-place a parametric Jacobian in a lazy manner, one parameter at the time
"""
function lazy_jacobian!(
  A,
  op::ParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  LinearAlgebra.fillstored!(A,zero(eltype(A)))
  lazy_jacobian_add!(A,op,μ,u,paramcache)
  A
end

function lazy_jacobian(
  op::ParamOperator,
  μ::Realization,
  u::AbstractVector)

  index = 1
  paramcache = allocate_paramcache(op,μ,u,index)
  jacobian(op,μ,u,paramcache)
end

function lazy_jacobian(
  op::ParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  A = lazy_allocate_jacobian(op,μ,u,paramcache)
  lazy_jacobian!(A,op,μ,u,paramcache)
  A
end

FESpaces.get_test(op::ParamOperator) = get_test(get_fe_operator(op))
FESpaces.get_trial(op::ParamOperator) = get_trial(get_fe_operator(op))
DofMaps.get_dof_map(op::ParamOperator) = get_dof_map(get_fe_operator(op))
DofMaps.get_internal_dof_map(op::ParamOperator) = get_internal_dof_map(get_fe_operator(op))
DofMaps.get_sparse_dof_map(op::ParamOperator) = get_sparse_dof_map(get_fe_operator(op))
get_dof_map_at_domains(op::ParamOperator) = get_dof_map_at_domains(get_fe_operator(op))
get_sparse_dof_map_at_domains(op::ParamOperator) = get_sparse_dof_map_at_domains(get_fe_operator(op))

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
      matcache
      veccache
    end
"""
mutable struct AssemCache{A,B}
  matdata::A
  vecdata::B
  matcache
  veccache
end

isstored_matdata(c::AssemCache) = isnothing(c.matdata)
isstored_vecdata(c::AssemCache) = isnothing(c.vecdata)
fill_matdata!(c::AssemCache,matdata) = (c.matdata=matdata)
fill_vecdata!(c::AssemCache,vecdata) = (c.vecdata=vecdata)
fill_matcache!(c::AssemCache,matcache) = (c.matcache=matcache)
fill_veccache!(c::AssemCache,vecdcache) = (c.veccache=veccache)
empty_matdata!(c::AssemCache) = (c.matdata=nothing)
empty_vecdata!(c::AssemCache) = (c.vecdata=nothing)
empty_matvecdata!(c::AssemCache) = empty_matdata!(c); empty_vecdata!(c)
get_mat_data_cache(c::AssemCache) = (c.matdata,c.matcache)
get_vec_data_cache(c::AssemCache) = (c.vecdata,c.veccache)

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
    allocate_paramcache(op::ParamOperator,μ::Realization,u::AbstractVector
      ) -> LazyParamOpCache

Similar to `allocate_odecache` in `Gridap`, when dealing with
parametric problems
"""
function allocate_paramcache(
  op::ParamOperator,
  μ::Realization,
  u::AbstractVector,
  index::Int)

  feop = get_fe_operator(op)
  ptrial = get_trial(feop)
  trial = evaluate(ptrial,μ)
  LazyParamOpCache(trial,ptrial,index)
end

function update_paramcache!(paramcache::LazyParamOpCache,op::ParamOperator,μ::Realization)
  paramcache.trial = evaluate!(paramcache.trial,paramcache.ptrial,μ)
  paramcache
end

isstored_matdata(c::LazyParamOpCache) = isstored_matdata(c.assemcache)
isstored_vecdata(c::LazyParamOpCache) = isstored_vecdata(c.assemcache)
fill_matdata!(c::LazyParamOpCache,matdata) = fill_matdata!(c.assemcache,matdata)
fill_vecdata!(c::LazyParamOpCache,vecdata) = fill_vecdata!(c.assemcache,vecdata)
fill_matcache!(c::LazyParamOpCache,matcache) = fill_matcache!(c.assemcache,matcache)
fill_veccache!(c::LazyParamOpCache,vecdcache) = fill_veccache!(c.assemcache,veccache)
empty_matdata!(c::LazyParamOpCache) = empty_matdata!(c.assemcache)
empty_vecdata!(c::LazyParamOpCache) = empty_vecdata!(c.assemcache)
empty_matvecdata!(c::LazyParamOpCache) = empty_matvecdata!(c.assemcache)
get_mat_data_cache(c::LazyParamOpCache) = get_mat_data_cache(c.assemcache)
get_vec_data_cache(c::LazyParamOpCache) = get_vec_data_cache(c.assemcache)

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

"""
    struct ParamNonlinearOperator <: NonlinearOperator
      op::ParamOperator
      μ::Realization
      paramcache::AbstractParamCache
    end

Fields:
- `op`: `ParamOperator` representing a parametric differential problem
- `μ`: `Realization` representing the parameters at which the problem is solved
- `paramcache`: cache of the problem
"""
struct ParamNonlinearOperator <: NonlinearOperator
  op::ParamOperator
  μ::Realization
  paramcache::AbstractParamCache
end

function ParamNonlinearOperator(op::ParamOperator,μ::Realization)
  trial = get_trial(op)(μ)
  u = zero_free_values(trial)
  paramcache = allocate_paramcache(op,μ,u)
  ParamNonlinearOperator(op,μ,paramcache)
end

function Algebra.allocate_residual(
  nlop::ParamNonlinearOperator,
  x::AbstractVector)

  allocate_residual(nlop.op,nlop.μ,x,nlop.paramcache)
end

function Algebra.residual!(
  b::AbstractVector,
  nlop::ParamNonlinearOperator,
  x::AbstractVector)

  residual!(b,nlop.op,nlop.μ,x,nlop.paramcache)
end

function Algebra.allocate_jacobian(
  nlop::ParamNonlinearOperator,
  x::AbstractVector)

  allocate_jacobian(nlop.op,nlop.μ,x,nlop.paramcache)
end

function Algebra.jacobian!(
  A::AbstractMatrix,
  nlop::ParamNonlinearOperator,
  x::AbstractVector)

  jacobian!(A,nlop.op,nlop.μ,x,nlop.paramcache)
end

function allocate_residual(
  nlop::ParamNonlinearOperator,
  x::AbstractVector)

  allocate_lazy_residual(nlop.op,nlop.μ,x,nlop.paramcache)
end

function lazy_residual!(
  b::AbstractVector,
  nlop::ParamNonlinearOperator,
  x::AbstractVector)

  lazy_residual!(b,nlop.op,nlop.μ,x,nlop.paramcache)
end

function allocate_lazy_jacobian(
  nlop::ParamNonlinearOperator,
  x::AbstractVector)

  allocate_lazy_jacobian(nlop.op,nlop.μ,x,nlop.paramcache)
end

function lazy_jacobian!(
  A::AbstractMatrix,
  nlop::ParamNonlinearOperator,
  x::AbstractVector)

  lazy_jacobian!(A,nlop.op,nlop.μ,x,nlop.paramcache)
end
