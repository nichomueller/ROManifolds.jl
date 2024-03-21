abstract type ODEParamOperatorType <: ODEOperatorType end
struct NonlinearParamODE <: ODEParamOperatorType end
struct QuasilinearParamODE <: ODEParamOperatorType end
struct SemilinearParamODE <: ODEParamOperatorType end
struct LinearParamODE <: ODEParamOperatorType end

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

  fillstored!(A,zero(eltype(A)))
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
  A::Tuple{Vararg{Contribution}},
  odeop::ODEParamOperatorWithTrian,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  @abstractmethod
end

function Algebra.jacobian!(
  A::Tuple{Vararg{Contribution}},
  odeop::ODEParamOperatorWithTrian,
  r::TransientParamRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  odeopcache)

  for Ak in A
    fillstored!(Ak,zero(eltype(Ak)))
  end
  jacobian_add!(A,odeop,r,us,ws,odeopcache)
  A
end
