abstract type ParamOperator{C<:FunctionalStyle} <: GridapType end
const AffineParamOperator = ParamOperator{Affine}

"""
A wrapper of `ParamFEOperator` that transforms it to `ParamOperator`, i.e.,
takes A(μ,uh,vh) and returns A(μ,uF), where uF represents the free values
of the `EvaluationFunction` uh
"""
struct ParamOpFromFEOp{C} <: ParamOperator{C}
  feop::ParamFEOperator{C}
end

abstract type ParamODEOperator{C<:FunctionalStyle} <: ODEOperator{C} end
const AffineParamODEOperator = ParamODEOperator{<:Affine}

"""
A wrapper of `ParamTransientFEOperator` that transforms it to `ParamODEOperator`, i.e.,
takes A(μ,t,uh,∂tuh,∂t^2uh,...,∂t^Nuh,vh) and returns A(μ,t,uF,∂tuF,...,∂t^NuF)
where uF,∂tuF,...,∂t^NuF represent the free values of the `EvaluationFunction`
uh,∂tuh,∂t^2uh,...,∂t^Nuh.
"""
struct ParamODEOpFromFEOp{C} <: ParamODEOperator{C}
  feop::ParamTransientFEOperator{C}
end
