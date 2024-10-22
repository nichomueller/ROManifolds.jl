"""
    abstract type ODEParamOperatorType <: UnEvalOperatorType end

Parametric extension of the type [`UnEvalOperatorType`](@ref) in [`Gridap`](@ref)

"""
abstract type ODEParamOperatorType <: UnEvalOperatorType end

struct NonlinearParamODE <: ODEParamOperatorType end

abstract type AbstractLinearParamODE <: ODEParamOperatorType end
struct SemilinearParamODE <: AbstractLinearParamODE end
struct LinearParamODE <: AbstractLinearParamODE end
struct LinearNonlinearParamODE <: ODEParamOperatorType end

"""
    abstract type ODEParamOperator{T<:ODEParamOperatorType} <: ODEOperator{T} end

Parametric extension of the type [`ODEOperator`](@ref) in [`Gridap`](@ref).

Subtypes:
- [`ODEParamOperatorWithTrian`](@ref)
- [`ODEParamOpFromTFEOp`](@ref)

"""
abstract type ODEParamOperator{T<:ODEParamOperatorType} <: ParamOperator{T} end

function ParamSteady.allocate_paramcache(
  odeop::ODEParamOperator,
  us::Tuple{Vararg{AbstractVector}})

  nothing
end

function ParamSteady.update_paramcache!(
  paramcache,
  odeop::ODEParamOperator,
  us::Tuple{Vararg{AbstractVector}})

  paramcache
end

function Algebra.allocate_residual(
  odeop::ODEParamOperator,
  us::Tuple{Vararg{AbstractVector}},
  paramcache)

  @abstractmethod
end

function Algebra.residual!(
  b::AbstractVector,
  odeop::ODEParamOperator,
  us::Tuple{Vararg{AbstractVector}},
  paramcache;
  add::Bool=false)

  @abstractmethod
end

function Algebra.residual(
  odeop::ODEParamOperator,
  us::Tuple{Vararg{AbstractVector}},
  paramcache)

  b = allocate_residual(odeop,us,paramcache)
  residual!(b,odeop,us,paramcache)
  b
end

function Algebra.allocate_jacobian(
  odeop::ODEParamOperator,
  us::Tuple{Vararg{AbstractVector}},
  paramcache)

  @abstractmethod
end

function ODEs.jacobian_add!(
  A::AbstractMatrix,
  odeop::ODEParamOperator,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache)

  @abstractmethod
end

function Algebra.jacobian!(
  A::AbstractMatrix,
  odeop::ODEParamOperator,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache)

  LinearAlgebra.fillstored!(A,zero(eltype(A)))
  jacobian_add!(A,odeop,us,ws,paramcache)
  A
end

function Algebra.jacobian(
  odeop::ODEParamOperator,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache)

  A = allocate_jacobian(odeop,us,paramcache)
  jacobian!(A,odeop,us,ws,paramcache)
  A
end

Polynomials.get_order(odeop::ODEParamOperator) = get_order(get_fe_operator(op))
ODEs.get_num_forms(odeop::ODEParamOperator) = get_num_forms(get_fe_operator(op))
ODEs.is_form_constant(odeop::ODEParamOperator,k::Integer) = is_form_constant(get_fe_operator(op),k)

"""
    abstract type ODEParamOperatorWithTrian{T<:ODEParamOperatorType} <: ODEParamOperator{T} end

Is to a ODEParamOperator as a TransientParamFEOperatorWithTrian is to a TransientParamFEOperator.

Suptypes:
- [`ODEParamOpFromTFEOpWithTrian`](@ref)
- [`TransientRBOperator`](@ref)

"""
abstract type ODEParamOperatorWithTrian{T<:ODEParamOperatorType} <: ODEParamOperator{T} end

function Algebra.residual!(
  b::Contribution,
  odeop::ODEParamOperatorWithTrian,
  us::Tuple{Vararg{AbstractVector}},
  paramcache;
  add::Bool=false)

  @abstractmethod
end

function Algebra.residual(
  odeop::ODEParamOperatorWithTrian,
  us::Tuple{Vararg{AbstractVector}},
  paramcache)

  b = allocate_residual(odeop,us,paramcache)
  residual!(b,odeop,us,paramcache)
  b
end

function ODEs.jacobian_add!(
  A::TupOfArrayContribution,
  odeop::ODEParamOperatorWithTrian,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache)

  @abstractmethod
end

function Algebra.jacobian!(
  A::TupOfArrayContribution,
  odeop::ODEParamOperatorWithTrian,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache)

  LinearAlgebra.fillstored!(A,zero(eltype(A)))
  jacobian_add!(A,odeop,us,ws,paramcache)
  A
end
