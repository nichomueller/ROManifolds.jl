"""
    const MultiFieldParamFESpace = MultiFieldFESpace
"""
const MultiFieldParamFESpace = MultiFieldFESpace

function MultiFieldParamFESpace(
  spaces::Vector{<:SingleFieldParamFESpace};
  style = BlockMultiFieldStyle())

  @notimplementedif !isa(style,BlockMultiFieldStyle)
  style = BlockMultiFieldStyle(style,spaces)
  fv = mortar(map(zero_free_values,spaces))
  V = typeof(fv)
  L = param_length(fv)
  PV = ParamType{V,L}
  MultiFieldFESpace(PV,spaces,style)
end

#TODO the code does not compile when defining these functions
# function MultiFieldParamFESpace(
#   spaces::Vector{<:Union{<:SingleFieldFESpace,<:SingleFieldParamFESpace}};
#   style = BlockMultiFieldStyle())

#   spaces′ = ParamDataStructures.to_param_quantities(spaces...)
#   MultiFieldFESpace(spaces′,style=style)
# end

# function MultiFieldParamFESpace(
#   ::Type{V},
#   spaces::Vector{<:Union{<:SingleFieldFESpace,<:SingleFieldParamFESpace}};
#   style = BlockMultiFieldStyle()) where V

#   spaces′ = ParamDataStructures.to_param_quantities(spaces...)
#   MultiFieldParamFESpace(spaces;style)
# end

FESpaces.get_dof_value_type(f::MultiFieldFESpace{MS,CS,<:ParamType{V}}) where {MS,CS,T,V<:AbstractParamArray{T}} = T

function MultiField._restrict_to_field(
  f,
  ::Union{<:ConsecutiveMultiFieldStyle,<:BlockMultiFieldStyle},
  free_values::AbstractParamVector,
  field)

  U = f.spaces
  offsets = MultiField._compute_field_offsets(U)
  pini = offsets[field] + 1
  pend = offsets[field] + num_free_dofs(U[field])
  get_param_entry(free_values,pini:pend)
end

function MultiField._restrict_to_field(
  f,
  mfs::BlockMultiFieldStyle{NB,SB,P},
  free_values::BlockParamVector,
  field
  ) where {NB,SB,P}

  @check blocklength(free_values) == NB
  U = f.spaces

  # Find the block for this field
  block_ranges = MultiField.get_block_ranges(NB,SB,P)
  block_idx    = findfirst(range -> field ∈ range,block_ranges)
  block_free_values = blocks(free_values)[block_idx]

  # Within the block,restrict to field
  offsets = compute_field_offsets(f,mfs)
  pini = offsets[field] + 1
  pend = offsets[field] + num_free_dofs(U[field])
  return get_param_entry(block_free_values,pini:pend)
end

function FESpaces.interpolate!(objects,free_values::AbstractParamVector,fe::MultiFieldFESpace)
  blocks = SingleFieldParamFEFunction[]
  for (field,(U,object)) in enumerate(zip(fe.spaces,objects))
    free_values_i = restrict_to_field(fe,free_values,field)
    uhi = interpolate!(object,free_values_i,U)
    push!(blocks,uhi)
  end
  MultiFieldFEFunction(free_values,fe,blocks)
end

function FESpaces.interpolate_everywhere(
  objects::AbstractVector{<:AbstractParamFunction},
  fe::MultiFieldFESpace)

  free_values = zero_free_values(fe)
  blocks = SingleFieldParamFEFunction[]
  for (field,(U,object)) in enumerate(zip(fe.spaces,objects))
    free_values_i = restrict_to_field(fe,free_values,field)
    dirichlet_values_i = zero_dirichlet_values(U)
    uhi = interpolate_everywhere!(object,free_values_i,dirichlet_values_i,U)
    push!(blocks,uhi)
  end
  MultiFieldFEFunction(free_values,fe,blocks)
end

function FESpaces.interpolate_everywhere!(
  objects,
  free_values::AbstractParamVector,
  dirichlet_values::AbstractParamVector,
  fe::MultiFieldFESpace)

  blocks = SingleFieldParamFEFunction[]
  for (field,(U,object)) in enumerate(zip(fe.spaces,objects))
    free_values_i = restrict_to_field(fe,free_values,field)
    dirichlet_values_i = dirichlet_values[field]
    uhi = interpolate_everywhere!(object,free_values_i,dirichlet_values_i,U)
    push!(blocks,uhi)
  end
  MultiFieldFEFunction(free_values,fe,blocks)
end

function FESpaces.interpolate_dirichlet(
  objects::AbstractVector{<:AbstractParamFunction},
  fe::MultiFieldFESpace)

  free_values = zero_free_values(fe)
  blocks = SingleFieldParamFEFunction[]
  for (field,(U,object)) in enumerate(zip(fe.spaces,objects))
    free_values_i = restrict_to_field(fe,free_values,field)
    dirichlet_values_i = zero_dirichlet_values(U)
    uhi = interpolate_dirichlet!(object,free_values_i,dirichlet_values_i,U)
    push!(blocks,uhi)
  end
  MultiFieldFEFunction(free_values,fe,blocks)
end
