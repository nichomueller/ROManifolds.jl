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
    abstract type ParamOperator{O<:UnEvalOperatorType,T<:TriangulationStyle} <: NonlinearParamOperator end

Type representing algebraic operators when solving parametric differential problems
"""
abstract type ParamOperator{O<:UnEvalOperatorType,T<:TriangulationStyle} <: NonlinearParamOperator end

"""
    get_fe_operator(op::ParamOperator) -> ParamFEOperator

Fetches the underlying FE operator of an algebraic operator `op`
"""
get_fe_operator(op::ParamOperator) = @abstractmethod

FESpaces.get_test(op::ParamOperator) = get_test(get_fe_operator(op))
FESpaces.get_trial(op::ParamOperator) = get_trial(get_fe_operator(op))
DofMaps.get_dof_map(op::ParamOperator) = get_dof_map(get_fe_operator(op))
DofMaps.get_internal_dof_map(op::ParamOperator) = get_internal_dof_map(get_fe_operator(op))
DofMaps.get_sparse_dof_map(op::ParamOperator) = get_sparse_dof_map(get_fe_operator(op))
get_dof_map_at_domains(op::ParamOperator) = get_dof_map_at_domains(get_fe_operator(op))
get_sparse_dof_map_at_domains(op::ParamOperator) = get_sparse_dof_map_at_domains(get_fe_operator(op))

function ParamAlgebra.allocate_paramcache(op::ParamOperator,μ::Realization)
  feop = get_fe_operator(op)
  ptrial = get_trial(feop)
  trial = evaluate(ptrial,μ)
  ParamCache(trial,ptrial)
end

const LinearNonlinearParamOperator{T} = ParamOperator{LinearNonlinearParamEq,T}

get_fe_operator(op::LinearNonlinearParamOperator) = get_fe_operator(get_nonlinear_operator(op))

function ParamAlgebra.allocate_paramcache(op::LinearNonlinearParamOperator,μ::Realization)
  op_nlin = get_nonlinear_operator(op)
  allocate_paramcache(op_nlin,μ)
end

function ParamAlgebra.allocate_systemcache(op::LinearNonlinearParamOperator,u::AbstractVector)
  op_nlin = get_nonlinear_operator(op)
  allocate_systemcache(op_nlin,u)
end

function ParamAlgebra.update_paramcache!(
  paramcache,
  op::LinearNonlinearParamOperator,
  μ::Realization)

  op_nlin = get_nonlinear_operator(op)
  update_paramcache!(paramcache,op_nlin,μ)
end

function Algebra.allocate_residual(
  op::LinearNonlinearParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  @notimplemented "This is inefficient. Instead, assemble the nonlinear system
  by defining a [`LinearNonlinearParamOperator`](@ref)"
end

function Algebra.allocate_jacobian(
  op::LinearNonlinearParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  @notimplemented "This is inefficient. Instead, assemble the nonlinear system
  by defining a [`LinearNonlinearParamOperator`](@ref)"
end

function Algebra.residual!(
  b,
  op::LinearNonlinearParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  @notimplemented "This is inefficient. Instead, assemble the nonlinear system
  by defining a [`LinearNonlinearParamOperator`](@ref)"
end

function ODEs.jacobian_add!(
  A,
  op::LinearNonlinearParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  @notimplemented "This is inefficient. Instead, assemble the nonlinear system
  by defining a [`LinearNonlinearParamOperator`](@ref)"
end

function ParamDataStructures.parameterize(op::LinearNonlinearParamOperator,μ::Realization)
  op_lin = parameterize(get_linear_operator(op),μ)
  op_nlin = parameterize(get_nonlinear_operator(op),μ)
  syscache_lin = allocate_systemcache(op_lin)
  LinNonlinParamOperator(op_lin,op_nlin,syscache_lin)
end
