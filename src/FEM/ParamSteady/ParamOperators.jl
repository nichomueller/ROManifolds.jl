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

function ParamAlgebra.allocate_lazy_residual(
  op::ParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  @abstractmethod
end

function ParamAlgebra.lazy_residual!(
  b,
  op::ParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  @abstractmethod
end

function ParamAlgebra.lazy_residual(
  op::ParamOperator,
  μ::Realization,
  u::AbstractVector)

  paramcache = allocate_lazy_paramcache(op,μ,u)
  residual(op,μ,u,paramcache)
end

function ParamAlgebra.lazy_residual(
  op::ParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  b = allocate_lazy_residual(op,μ,u,paramcache)
  lazy_residual!(b,op,μ,u,paramcache)
  b
end

function ParamAlgebra.allocate_lazy_jacobian(
  op::ParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  @abstractmethod
end

function ParamAlgebra.lazy_jacobian_add!(
  A,
  op::ParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  @abstractmethod
end

function ParamAlgebra.lazy_jacobian!(
  A,
  op::ParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  LinearAlgebra.fillstored!(A,zero(eltype(A)))
  lazy_jacobian_add!(A,op,μ,u,paramcache)
  A
end

function ParamAlgebra.lazy_jacobian(
  op::ParamOperator,
  μ::Realization,
  u::AbstractVector)

  paramcache = allocate_lazy_paramcache(op,μ,u)
  jacobian(op,μ,u,paramcache)
end

function ParamAlgebra.lazy_jacobian(
  op::ParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  A = allocate_lazy_jacobian(op,μ,u,paramcache)
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

function ParamDataStructures.parameterize(op::ParamOperator,μ::Realization)
  trial = get_trial(op)(μ)
  u = zero_free_values(trial)
  paramcache = allocate_paramcache(op,μ,u)
  GenericParamNonlinearOperator(op,μ,paramcache)
end

function ParamDataStructures.lazy_parameterize(op::ParamOperator,μ::Realization)
  trial = get_trial(op)(μ)
  u = zero_free_values(trial)
  paramcache = allocate_lazy_paramcache(op,μ,u)
  LazyParamNonlinearOperator(op,μ,paramcache)
end
