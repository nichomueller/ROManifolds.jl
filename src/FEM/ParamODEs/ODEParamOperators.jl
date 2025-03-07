"""
    abstract type ODEParamOperatorType <: UnEvalOperatorType end

Parametric extension of the type `ODEOperatorType` in `Gridap`.

Subtypes:

- [`NonlinearParamODE`](@ref)
- [`LinearParamODE`](@ref)
- [`LinearNonlinearParamODE`](@ref)
"""
abstract type ODEParamOperatorType <: UnEvalOperatorType end

"""
    struct NonlinearParamODE <: ODEParamOperatorType end
"""
struct NonlinearParamODE <: ODEParamOperatorType end

"""
    struct LinearParamODE <: ODEParamOperatorType end
"""
struct LinearParamODE <: ODEParamOperatorType end

"""
    struct LinearNonlinearParamODE <: ODEParamOperatorType end
"""
struct LinearNonlinearParamODE <: ODEParamOperatorType end

"""
    const ODEParamOperator{T<:ODEParamOperatorType,T<:TriangulationStyle} <: ParamOperator{O,T}

Transient extension of the type [`ParamOperator`](@ref).
"""
const ODEParamOperator{O<:ODEParamOperatorType,T<:TriangulationStyle} = ParamOperator{O,T}

"""
    const JointODEParamOperator{O<:ODEParamOperatorType} = ODEParamOperator{O,JointDomains}
"""
const JointODEParamOperator{O<:ODEParamOperatorType} = ODEParamOperator{O,JointDomains}

"""
    const SplitODEParamOperator{O<:ODEParamOperatorType} = ODEParamOperator{O,SplitDomains}
"""
const SplitODEParamOperator{O<:ODEParamOperatorType} = ODEParamOperator{O,SplitDomains}

get_order(odeop::ODEParamOperator) = get_order(get_fe_operator(odeop))
ODEs.is_form_constant(odeop::ODEParamOperator,k::Integer) = is_form_constant(get_fe_operator(odeop),k)

function ParamAlgebra.allocate_paramcache(
  odeop::ODEParamOperator,
  r::TransientRealization;
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
  ParamCache(trials,pttrials)
end

function ParamAlgebra.update_paramcache!(
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

# constructors

function TransientParamLinearOperator(args...;kwargs...)
  feop = TransientParamLinearFEOperator(args...;kwargs...)
  get_algebraic_operator(feop)
end

function TransientParamOperator(args...;kwargs...)
  feop = TransientParamFEOperator(args...;kwargs...)
  get_algebraic_operator(feop)
end

function LinearNonlinearTransientParamOperator(op_lin::ParamOperator,op_nlin::ParamOperator)
  feop_lin = get_fe_operator(op_lin)
  feop_nlin = get_fe_operator(op_nlin)
  feop = LinearNonlinearParamFEOperator(feop_lin,feop_nlin)
  get_algebraic_operator(feop)
end
