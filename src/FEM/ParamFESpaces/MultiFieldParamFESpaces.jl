function ParamDataStructures.param_length(spaces::SingleFieldFESpace...)
  pspaces = filter(x->isa(x,SingleFieldParamFESpace),spaces)
  L = length_free_values.(pspaces)
  @check all(L .== first(L))
  return first(L)
end

function _MultiFieldParamFESpace(
  spaces::Vector{<:SingleFieldFESpace};
  style = ConsecutiveMultiFieldStyle())

  if isa(style,BlockMultiFieldStyle)
    style = BlockMultiFieldStyle(style,spaces)
    VT = typeof(mortar(map(zero_free_values,spaces)))
  else
    VT = promote_type(map(get_vector_type,spaces)...)
  end
  MultiFieldFESpace(VT,spaces,style)
end

function MultiFieldParamFESpace(
  spaces::Vector{<:SingleFieldFESpace};
  style = ConsecutiveMultiFieldStyle())

  if any(isa.(spaces,SingleFieldParamFESpace))
    L = param_length(spaces...)
    spaces′ = FESpaceToParamFESpace.(spaces,L)
    _MultiFieldParamFESpace(spaces′,style=style)
  elseif any(isa.(spaces,TransientTrialParamFESpace))
    _MultiFieldParamFESpace(spaces,style=style)
  else
    MultiFieldFESpace(spaces,style=style)
  end
end

function MultiFieldParamFESpace(
  ::Type{V},
  spaces::Vector{<:SingleFieldFESpace};
  style = ConsecutiveMultiFieldStyle()) where V

  MultiFieldFESpace(V,spaces;style)
end
