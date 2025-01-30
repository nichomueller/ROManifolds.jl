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

Trait for a FE operator indicating that residuals/jacobians in this operator
should be computed summing the contributions relative to each triangulation as
occurs in `Gridap`
"""
struct JointDomains <: TriangulationStyle end

"""
    struct SplitDomains <: TriangulationStyle end

Trait for a FE operator indicating that residuals/jacobians in this operator
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

FESpaces.get_test(op::ParamOperator) = get_test(get_fe_operator(op))
FESpaces.get_trial(op::ParamOperator) = get_trial(get_fe_operator(op))
DofMaps.get_dof_map(op::ParamOperator) = get_dof_map(get_fe_operator(op))
DofMaps.get_sparse_dof_map(op::ParamOperator) = get_sparse_dof_map(get_fe_operator(op))
get_dof_map_at_domains(op::ParamOperator) = get_dof_map_at_domains(get_fe_operator(op))
get_sparse_dof_map_at_domains(op::ParamOperator) = get_sparse_dof_map_at_domains(get_fe_operator(op))

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

"""
    update_paramcache!(paramcache,op::ParamOperator,μ::Realization) -> ParamOpCache

Similar to `update_odecache!` in `Gridap`, when dealing with
parametric problems
"""
function update_paramcache!(paramcache,op::ParamOperator,μ::Realization)
  paramcache.trial = evaluate!(paramcache.trial,paramcache.ptrial,μ)
  paramcache
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

"""
    abstract type AbstractParamCache <: GridapType end
"""
abstract type AbstractParamCache <: GridapType end

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
    struct ParamOpSysCache{Ta,Tb} <: AbstractParamCache
      paramcache::ParamOpCache
      A::Ta
      b::Tb
    end
"""
struct ParamOpSysCache{Ta,Tb} <: AbstractParamCache
  paramcache::ParamOpCache
  A::Ta
  b::Tb
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
