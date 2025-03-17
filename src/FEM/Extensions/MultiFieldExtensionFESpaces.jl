struct MultiFieldExtensionFESpace{CS<:ConstraintStyle,V} <: FESpace
  vector_type::Type{V}
  spaces::Vector{<:SingleFieldExtensionFESpace}
  constraint_style::CS

  function MultiFieldExtensionFESpace(
    ::Type{V},
    spaces::Vector{<:SingleFieldExtensionFESpace}
    ) where V

    @assert length(spaces) > 0
    if any( map(has_constraints,spaces) )
      constraint_style = Constrained()
    else
      constraint_style = UnConstrained()
    end
    CS = typeof(constraint_style)
    new{CS,V}(V,spaces,constraint_style)
  end
end

function MultiField.MultiFieldFESpace(
  spaces::Vector{<:SingleFieldExtensionFESpace};
  style=MultiField.BlockMultiFieldStyle()
  )

  @notimplementedif !isa(style,MultiField.BlockMultiFieldStyle)

  Ts = map(get_dof_value_type,spaces)
  VT = get_vector_type(first(spaces))
  @check all(get_vector_type(f)==VT for f in spaces)
  MultiFieldExtensionFESpace(VT,spaces)
end

function MultiField.MultiFieldFESpace(::Type{V},spaces::Vector{<:SingleFieldExtensionFESpace}) where V
  MultiFieldExtensionFESpace(V,spaces)
end

function MultiField.MultiFieldStyle(f::MultiFieldExtensionFESpace)
  MultiField.BlockMultiFieldStyle(2*length(f.spaces))
end

MultiField.MultiFieldStyle(::Type{<:MultiFieldExtensionFESpace}) = @notimplemented

function get_internal_space(f::MultiFieldExtensionFESpace{CS,V}) where {CS,V}
  spaces = map(get_internal_space,f.spaces)
  MultiFieldFESpace(V,spaces,style=BlockMultiFieldStyle(num_fields(f)))
end

function get_external_space(f::MultiFieldExtensionFESpace{CS,V}) where {CS,V}
  spaces = map(get_external_space,f.spaces)
  MultiFieldFESpace(V,spaces,style=BlockMultiFieldStyle(num_fields(f)))
end

function MultiField.num_fields(f::MultiFieldExtensionFESpace)
  length(f.spaces)
end

Base.length(f::MultiFieldExtensionFESpace) = 2*num_fields(f)

function Base.iterate(f::MultiFieldExtensionFESpace)
  state = 1
  return f.spaces[1][1],state+1
end

function Base.iterate(f::MultiFieldExtensionFESpace,state)
  if state > length(f)
    return nothing
  end
  intext = fast_index(state,2)
  ifield = slow_index(state,2)
  return f.spaces[ifield][intext],state+1
end

function Base.getindex(f::MultiFieldExtensionFESpace,i)
  intext = fast_index(i,2)
  ifield = slow_index(i,2)
  return f.spaces[ifield][intext]
end

function FESpaces.get_triangulation(f::MultiFieldExtensionFESpace)
  s1 = first(f.spaces)
  trian = get_triangulation(s1)
  @check all(map(i->trian===get_triangulation(i),f.spaces))
  trian
end

function FESpaces.num_free_dofs(f::MultiFieldExtensionFESpace)
  n = 0
  for spaces in f
    n += num_free_dofs(spaces)
  end
  n
end

function FESpaces.get_free_dof_ids(f::MultiFieldExtensionFESpace)
  get_free_dof_ids(f,MultiField.MultiFieldStyle(f))
end

function FESpaces.get_free_dof_ids(f::MultiFieldExtensionFESpace,::BlockMultiFieldStyle{NB,SB,P}) where {NB,SB,P}
  block_ranges = MultiField.get_block_ranges(NB,SB,P)
  block_num_dofs = map(range->sum(map(num_free_dofs,f[range])),block_ranges)
  return BlockArrays.blockedrange(block_num_dofs)
end

function FESpaces.zero_dirichlet_values(f::MultiFieldExtensionFESpace)
  map(zero_dirichlet_values,f)
end

FESpaces.get_dof_value_type(f::MultiFieldExtensionFESpace{CS,V}) where {CS,V} = eltype(V)

FESpaces.get_vector_type(f::MultiFieldExtensionFESpace) = f.vector_type

FESpaces.ConstraintStyle(::Type{MultiFieldExtensionFESpace{CS,V}}) where {CS,V} = CS()

function FESpaces.get_fe_basis(f::MultiFieldExtensionFESpace)
  nspaces = length(f)
  all_febases = MultiField.MultiFieldFEBasisComponent[]
  for (ispace,space) in enumerate(f)
    dv_i = get_fe_basis(space)
    @assert BasisStyle(dv_i) == TestBasis()
    dv_i_b = MultiField.MultiFieldFEBasisComponent(dv_i,ispace,nspaces)
    push!(all_febases,dv_i_b)
  end
  MultiFieldCellField(all_febases)
end

function FESpaces.get_trial_fe_basis(f::MultiFieldExtensionFESpace)
  nspaces = length(f)
  all_febases = MultiField.MultiFieldFEBasisComponent[]
  for (ispace,space) in enumerate(f)
    dv_i = get_trial_fe_basis(space)
    @assert BasisStyle(dv_i) == TrialBasis()
    dv_i_b = MultiField.MultiFieldFEBasisComponent(dv_i,ispace,nspaces)
    push!(all_febases,dv_i_b)
  end
  MultiFieldCellField(all_febases)
end

function FESpaces.FEFunction(f::MultiFieldExtensionFESpace,fv)
  blocks = map(1:length(f)) do i
    fvi = restrict_to_field(f,fv,i)
    FEFunction(f[i],fvi)
  end
  MultiFieldFEFunction(fv,f,blocks)
end

function FESpaces.FEFunction(
  f::MultiFieldExtensionFESpace,
  fv::AbstractVector,
  dv::Vector{<:AbstractVector})

  @check length(dir_values) == length(f)
  blocks = map(1:length(f)) do i
    fvi = restrict_to_field(f,fv,i)
    dvi = dir_values[i]
    FEFunction(f[i],fvi,dvi)
  end
  MultiFieldFEFunction(fv,f,blocks)
end

function FESpaces.EvaluationFunction(f::MultiFieldExtensionFESpace,fv)
  blocks = map(1:length(f)) do i
    fvi = restrict_to_field(f,fv,i)
    EvaluationFunction(f[i],fvi)
  end
  MultiFieldFEFunction(fv,f,blocks)
end

function CellData.CellField(f::MultiFieldExtensionFESpace,cv)
  single_fields = map(1:length(f)) do i
    cvi = lazy_map(a->a.array[i],cv)
    CellField(f[i],cvi)
  end
  MultiFieldCellField(single_fields)
end

for f in (
  :(FESpaces.get_cell_isconstrained),
  :(FESpaces.get_cell_is_dirichlet),
  :(FESpaces.get_cell_constraints),
  :(FESpaces.get_cell_dof_ids))
  @eval begin
    function $f(f::MultiFieldExtensionFESpace)
      msg = """\n
      This method does not make sense for multi-field
      since each field can be defined on a different triangulation.
      Pass a triangulation in the second argument to get
      the constrain flag for the corresponding cells.
      """
      @notimplemented msg
    end
  end
end

function FESpaces.get_cell_isconstrained(f::MultiFieldExtensionFESpace,trian::Triangulation)
  data = map(f) do space
    trian_i = get_triangulation(space)
    if is_change_possible(trian_i,trian)
      get_cell_isconstrained(space,trian)
    else
      Fill(false,num_cells(trian))
    end
  end
  lazy_map( (args...) -> +(args...)>0,  data...)
end

function FESpaces.get_cell_is_dirichlet(f::MultiFieldExtensionFESpace,trian::Triangulation)
  data = map(f) do space
    trian_i = get_triangulation(space)
    if is_change_possible(trian_i,trian)
      get_cell_is_dirichlet(space,trian)
    else
      Fill(false,num_cells(trian))
    end
  end
  lazy_map( (args...) -> +(args...)>0,  data...)
end

function FESpaces.get_cell_constraints(f::MultiFieldExtensionFESpace,trian::Triangulation)
  nspaces = length(f)
  blockmask = [ is_change_possible(get_triangulation(Vi),trian) for Vi in f ]
  active_block_ids = findall(blockmask)
  active_block_data = Any[ get_cell_constraints(f[i],trian) for i in active_block_ids ]
  blockshape = (nspaces,nspaces)
  blockindices = [(i,i) for i in active_block_ids]
  lazy_map(BlockMap(blockshape,blockindices),active_block_data...)
end

function FESpaces.get_cell_dof_ids(f::MultiFieldExtensionFESpace,trian::Triangulation)
  offsets = MultiField.compute_field_offsets(f)
  nspaces = length(f)
  blockmask = [ is_change_possible(get_triangulation(Vi),trian) for Vi in f ]
  active_block_ids = findall(blockmask)
  active_block_data = Any[]
  for i in active_block_ids
    cell_dofs_i = get_cell_dof_ids(f[i],trian)
    if i == 1
      push!(active_block_data,cell_dofs_i)
    else
      offset = Int32(offsets[i])
      o = Fill(offset,length(cell_dofs_i))
      cell_dofs_i_b = lazy_map(Broadcasting(MultiField._sum_if_first_positive),cell_dofs_i,o)
      push!(active_block_data,cell_dofs_i_b)
    end
  end
  lazy_map(BlockMap(nspaces,active_block_ids),active_block_data...)
end

function FESpaces.interpolate(objects,f::MultiFieldExtensionFESpace)
  interpolate!(objects,zero_free_values(f),f)
end

function FESpaces.interpolate!(objects,fv::AbstractVector,f::MultiFieldExtensionFESpace)
  blocks = SingleFieldFEFunction[]
  for (i,(space,object)) in enumerate(zip(f,objects))
    fvi = restrict_to_field(f,fv,i)
    uhi = interpolate!(object,fvi,space)
    push!(blocks,uhi)
  end
  MultiFieldFEFunction(fv,f,blocks)
end

"""
like interpolate, but also compute new degrees of freedom for the dirichlet component.
The resulting MultiFieldFEFunction does not necessary belongs to the underlying space
"""
function FESpaces.interpolate_everywhere(objects,f::MultiFieldExtensionFESpace)
  fv = zero_free_values(f)
  blocks = SingleFieldFEFunction[]
  for (i,(space,object)) in enumerate(zip(f,objects))
    fvi = restrict_to_field(f,fv,i)
    dvi = zero_dirichlet_values(space)
    uhi = interpolate_everywhere!(object,fvi,dvi,space)
    push!(blocks,uhi)
  end
  MultiFieldFEFunction(fv,f,blocks)
end

function FESpaces.interpolate_everywhere!(objects,fv::AbstractVector,dv::Vector,f::MultiFieldExtensionFESpace)
  blocks = SingleFieldFEFunction[]
  for (i,(space,object)) in enumerate(zip(f,objects))
    fvi = restrict_to_field(f,fv,i)
    dvi = dv[i]
    uhi = interpolate_everywhere!(object,fvi,dvi,space)
    push!(blocks,uhi)
  end
  MultiFieldFEFunction(fv,f,blocks)
end

"""
"""
function FESpaces.interpolate_dirichlet(objects,f::MultiFieldExtensionFESpace)
  fv = zero_free_values(f)
  blocks = SingleFieldFEFunction[]
  for (i,(space,object)) in enumerate(zip(f,objects))
    fvi = restrict_to_field(f,fv,i)
    dvi = zero_dirichlet_values(space)
    uhi = interpolate_dirichlet!(object,fvi,dvi,space)
    push!(blocks,uhi)
  end
  MultiFieldFEFunction(fv,f,blocks)
end

# utils

function MultiField.restrict_to_field(
  f::MultiFieldExtensionFESpace,
  fv::AbstractVector,
  i::Integer)

  MultiField._restrict_to_field(f,MultiFieldStyle(f),fv,i)
end

function MultiField._restrict_to_field(
  f::MultiFieldExtensionFESpace,
  mfs::BlockMultiFieldStyle{NB,SB,P},
  fv::BlockVector,
  i
  ) where {NB,SB,P}

  @check blocklength(fv) == NB

  # Find the block for this field
  block_ranges = get_block_ranges(NB,SB,P)
  block_idx = findfirst(range -> i ∈ range,block_ranges)
  block_free_values = blocks(fv)[block_idx]

  # Within the block, restrict to field
  offsets = MultiField.compute_field_offsets(f,mfs)
  pini = offsets[i] + 1
  pend = offsets[i] + num_free_dofs(f[i])
  return view(block_free_values,pini:pend)
end

function MultiField.compute_field_offsets(f::MultiFieldExtensionFESpace)
  MultiField.compute_field_offsets(f,MultiField.MultiFieldStyle(f))
end

function MultiField.compute_field_offsets(
  f::MultiFieldExtensionFESpace,
  ::MultiField.BlockMultiFieldStyle{NB,SB,P}
  ) where {NB,SB,P}

  block_ranges = get_block_ranges(NB,SB,P)
  block_offsets = vcat(map(range->MultiField._compute_field_offsets(f[range]),block_ranges)...)
  offsets = map(p->block_offsets[p],P)
  return offsets
end
