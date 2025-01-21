"""
    TrivialParamFESpace{S} <: SingleFieldParamFESpace{S}

Wrapper for nonparametric FE spaces that we wish assumed a parametric length `plength`
"""
struct TrivialParamFESpace{S} <: SingleFieldParamFESpace{S}
  space::S
  plength::Int
end

FESpaces.get_fe_space(f::TrivialParamFESpace) = f.space

ParamDataStructures.param_length(f::TrivialParamFESpace) = f.plength
ParamDataStructures.to_param_quantity(f::SingleFieldParamFESpace,plength::Integer) = f
ParamDataStructures.to_param_quantity(f::SingleFieldFESpace,plength::Integer) = TrivialParamFESpace(f,plength)
ParamDataStructures.param_getindex(f::TrivialParamFESpace,index::Integer) = f.space

function FESpaces.TrialFESpace(tf::TrivialParamFESpace)
  f = tf.space
  U = TrialFESpace(f)
  TrivialParamFESpace(U,param_length(tf))
end

# utils

remove_layer(f::TrivialParamFESpace) = TrivialParamFESpace(f.space.space,param_length(f))
remove_layer(f::TrivialParamFESpace{<:UnconstrainedFESpace}) = f
