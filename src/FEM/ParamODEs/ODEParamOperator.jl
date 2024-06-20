"""
    abstract type ODEParamOperatorType <: ODEOperatorType end

Parametric extension of the type [`ODEOperatorType`](@ref) in [`Gridap`](@ref)

"""
abstract type ODEParamOperatorType <: ODEOperatorType end

struct NonlinearParamODE <: ODEParamOperatorType end

abstract type AbstractLinearParamODE <: ODEParamOperatorType end
struct QuasilinearParamODE <: AbstractLinearParamODE end
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
abstract type ODEParamOperator{T<:ODEParamOperatorType} <: ODEOperator{T} end

function ODEs.allocate_odeopcache(
  odeop::ODEParamOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  args...)

  nothing
end

function ODEs.update_odeopcache!(
  odeopcache,
  odeop::ODEParamOperator,
  r::TransientParamRealization,
  args...)

  odeopcache
end

function Algebra.allocate_residual(
  odeop::ODEParamOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache)

  @abstractmethod
end

function Algebra.residual!(
  b::AbstractVector,
  odeop::ODEParamOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache;
  add::Bool=false)

  @abstractmethod
end

function Algebra.residual(
  odeop::ODEParamOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache)

  b = allocate_residual(odeop,r,us,odeopcache)
  residual!(b,odeop,r,us,odeopcache)
  b
end

function Algebra.allocate_jacobian(
  odeop::ODEParamOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache)

  @abstractmethod
end

function ODEs.jacobian_add!(
  A::AbstractMatrix,
  odeop::ODEParamOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  @abstractmethod
end

function Algebra.jacobian!(
  A::AbstractMatrix,
  odeop::ODEParamOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  LinearAlgebra.fillstored!(A,zero(eltype(A)))
  jacobian_add!(A,odeop,r,us,ws,odeopcache)
  A
end

function Algebra.jacobian(
  odeop::ODEParamOperator,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  A = allocate_jacobian(odeop,r,us,odeopcache)
  jacobian!(A,odeop,r,us,ws,odeopcache)
  A
end

mutable struct ParamODEOpFromTFEOpCache <: GridapType
  Us
  Uts
  tfeopcache
  const_forms
end

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
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache;
  add::Bool=false)

  @abstractmethod
end

function Algebra.residual(
  odeop::ODEParamOperatorWithTrian,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  odeopcache)

  b = allocate_residual(odeop,r,us,odeopcache)
  residual!(b,odeop,r,us,odeopcache)
  b
end

function ODEs.jacobian_add!(
  A::TupOfArrayContribution,
  odeop::ODEParamOperatorWithTrian,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  @abstractmethod
end

function Algebra.jacobian!(
  A::TupOfArrayContribution,
  odeop::ODEParamOperatorWithTrian,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  LinearAlgebra.fillstored!(A,zero(eltype(A)))
  jacobian_add!(A,odeop,r,us,ws,odeopcache)
  A
end
