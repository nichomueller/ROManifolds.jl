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

function MultiField._restrict_to_field(
  f,
  ::Union{<:ConsecutiveMultiFieldStyle,<:BlockMultiFieldStyle},
  free_values::AbstractParamVector,
  field)

  @notimplemented
end

function MultiField._restrict_to_field(
  f,
  mfs::BlockMultiFieldStyle{NB,SB,P},
  free_values::BlockVectorOfVectors,
  field
  ) where {NB,SB,P}

  @check blocklength(free_values) == NB
  U = f.spaces

  # Find the block for this field
  block_ranges = get_block_ranges(NB,SB,P)
  block_idx    = findfirst(range -> field ∈ range, block_ranges)
  block_free_values = blocks(free_values)[block_idx]

  # Within the block, restrict to field
  offsets = compute_field_offsets(f,mfs)
  pini = offsets[field] + 1
  pend = offsets[field] + num_free_dofs(U[field])
  return view(block_free_values,pini:pend)
end
