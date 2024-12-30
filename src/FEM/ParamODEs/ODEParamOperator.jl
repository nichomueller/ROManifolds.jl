"""
    abstract type ODEParamOperatorType <: UnEvalOperatorType end

Parametric extension of the type [`ODEOperatorType`](@ref) in [`Gridap`](@ref)
"""
abstract type ODEParamOperatorType <: UnEvalOperatorType end

struct NonlinearParamODE <: ODEParamOperatorType end
struct LinearParamODE <: ODEParamOperatorType end
struct LinearNonlinearParamODE <: ODEParamOperatorType end

"""
    abstract type ODEParamOperator{T<:ODEParamOperatorType,T<:TriangulationStyle} <: ParamOperator{O,T} end

Parametric extension of the type [`ODEOperator`](@ref) in [`Gridap`](@ref).

Subtypes:
- [`ODEParamOpFromTFEOp`](@ref)
- [`LinearNonlinearParamOpFromTFEOp`](@ref)
- [`TransientRBOperator`](@ref)
"""
abstract type ODEParamOperator{O<:ODEParamOperatorType,T<:TriangulationStyle} <: ParamOperator{O,T} end

Polynomials.get_order(odeop::ODEParamOperator) = get_order(get_fe_operator(odeop))
ODEs.is_form_constant(odeop::ODEParamOperator,k::Integer) = is_form_constant(get_fe_operator(odeop),k)

function Algebra.allocate_residual(
  odeop::ODEParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  paramcache)

  @abstractmethod
end

function Algebra.residual!(
  b,
  odeop::ODEParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  paramcache;
  add::Bool=false)

  @abstractmethod
end

function Algebra.residual(
  odeop::ODEParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}})

  paramcache = allocate_paramcache(odeop,r,us;evaluated=true)
  residual(odeop,r,us,paramcache)
end

function Algebra.residual(
  odeop::ODEParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  paramcache)

  b = allocate_residual(odeop,r,us,paramcache)
  residual!(b,odeop,r,us,paramcache)
  b
end

function Algebra.allocate_jacobian(
  odeop::ODEParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  paramcache)

  @abstractmethod
end

function ODEs.jacobian_add!(
  A,
  odeop::ODEParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache)

  @abstractmethod
end

function Algebra.jacobian!(
  A,
  odeop::ODEParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache)

  LinearAlgebra.fillstored!(A,zero(eltype(A)))
  jacobian_add!(A,odeop,r,us,ws,paramcache)
  A
end

function Algebra.jacobian(
  odeop::ODEParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}})

  paramcache = allocate_paramcache(odeop,r,us;evaluated=true)
  jacobian(odeop,r,us,ws,paramcache)
end

function Algebra.jacobian(
  odeop::ODEParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache)

  A = allocate_jacobian(odeop,r,us,paramcache)
  jacobian!(A,odeop,r,us,ws,paramcache)
  A
end

function ParamSteady.allocate_paramcache(
  odeop::ODEParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}};
  evaluated=false)

  feop = get_fe_operator(odeop)
  order = get_order(odeop)
  pttrial = get_trial(feop)
  trial = evaluated ? evaluate(pttrial,r) : allocate_space(pttrial,r)
  pttrials = (pttrial,)
  trials = (trial,)
  for k in 1:order
    pttrials = (pttrials...,∂t(pttrials[k]))
    trialk = evaluated ? evaluate(pttrials[k+1],r) : allocate_space(pttrials[k+1],r)
    trials = (trials...,trialk)
  end
  feop_cache = allocate_feopcache(feop,r,us)
  ParamOpCache(trials,pttrials,feop_cache)
end

function ParamSteady.update_paramcache!(
  paramcache,
  odeop::ODEParamOperator,
  r::TransientRealization)

  trials = ()
  for k in 1:get_order(odeop)+1
    trials = (trials...,evaluate!(paramcache.trial[k],paramcache.ptrial[k],r))
  end
  paramcache.trial = trials
  paramcache
end
