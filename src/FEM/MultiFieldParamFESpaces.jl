struct MultiFieldParamFESpace{MS<:MultiFieldStyle,CS<:ConstraintStyle,V} <: FESpace
  vector_type::Type{V}
  spaces::Vector{<:SingleFieldFESpace}
  multi_field_style::MS
  constraint_style::CS
  function MultiFieldParamFESpace(
    ::Type{V},
    spaces::Vector{<:SingleFieldFESpace},
    multi_field_style::MultiFieldStyle) where V
    @assert length(spaces) > 0

    MS = typeof(multi_field_style)
    if any(map(has_constraints,spaces))
      constraint_style = Constrained()
    else
      constraint_style = UnConstrained()
    end
    CS = typeof(constraint_style)
    new{MS,CS,V}(V,spaces,multi_field_style,constraint_style)
  end
end

function MultiFieldParamFESpace(
  spaces::Vector{<:SingleFieldParamFESpace};
  style = ConsecutiveMultiFieldStyle())

  T = get_vector_type(first(spaces))
  @check all(map(get_vector_type,spaces) .== T)
  if isa(style,BlockMultiFieldStyle)
    style = BlockMultiFieldStyle(style,spaces)
    VT = typeof(mortar(map(zero_free_values,spaces)))
  else
    VT = T
  end
  MultiFieldParamFESpace(VT,spaces,style)
end

function _find_length(spaces)
  pspaces = filter(x->isa(x,SingleFieldParamFESpace),spaces)
  L = length_free_values.(pspaces)
  @check all(L .== first(L))
  return first(L)
end

function MultiFieldParamFESpace(
  spaces::Vector{<:SingleFieldFESpace};
  style = ConsecutiveMultiFieldStyle())

  if any(isa.(spaces,SingleFieldParamFESpace))
    L = _find_length(spaces)
    MultiFieldParamFESpace(FESpaceToParamFESpace.(spaces,L),style=style)
  else
    MultiFieldFESpace(spaces,style=style)
  end
end

function MultiFieldParamFESpace(::Type{V},spaces::Vector{<:SingleFieldFESpace}) where V
  MultiFieldParamFESpace(V,spaces,ConsecutiveMultiFieldStyle())
end

const ParamFESpace = Union{SingleFieldParamFESpace,MultiFieldParamFESpace}

MultiField.MultiFieldStyle(::Type{MultiFieldParamFESpace{S,B,V}}) where {S,B,V} = S()
MultiField.MultiFieldStyle(f::MultiFieldParamFESpace) = MultiFieldStyle(typeof(f))

function FESpaces.get_triangulation(f::MultiFieldParamFESpace)
  s1 = first(f.spaces)
  trian = get_triangulation(s1)
  @check all(map(i->trian===get_triangulation(i),f.spaces))
  trian
end

function FESpaces.num_free_dofs(f::MultiFieldParamFESpace)
  n = 0
  for U in f.spaces
    n += num_free_dofs(U)
  end
  n
end

function FESpaces.get_free_dof_ids(f::MultiFieldParamFESpace)
  get_free_dof_ids(f,MultiFieldStyle(f))
end

function FESpaces.get_free_dof_ids(::MultiFieldParamFESpace,::MultiFieldStyle)
  @abstractmethod
end

function FESpaces.get_free_dof_ids(f::MultiFieldParamFESpace,::ConsecutiveMultiFieldStyle)
  block_num_dofs = Int[]
  for U in f.spaces
    push!(block_num_dofs,num_free_dofs(U))
  end
  blockedrange(block_num_dofs)
end

function FESpaces.get_free_dof_ids(f::MultiFieldParamFESpace,::BlockMultiFieldStyle{NB,SB,P}) where {NB,SB,P}
  block_ranges   = MultiField.get_block_ranges(NB,SB,P)
  block_num_dofs = map(range->sum(map(num_free_dofs,f.spaces[range])),block_ranges)
  return BlockArrays.blockedrange(block_num_dofs)
end

function FESpaces.zero_free_values(f::MultiFieldParamFESpace)
  V = get_vector_type(f)
  allocate_vector(V,num_free_dofs(f))
end

function FESpaces.zero_free_values(f::MultiFieldParamFESpace{<:BlockMultiFieldStyle{NB,SB,P}}) where {NB,SB,P}
  block_ranges   = MultiField.get_block_ranges(NB,SB,P)
  block_num_dofs = map(range->sum(map(num_free_dofs,f.spaces[range])),block_ranges)
  block_vtypes   = map(range->get_vector_type(first(f.spaces[range])),block_ranges)
  return mortar(map(allocate_vector,block_vtypes,block_num_dofs))
end

FESpaces.get_dof_value_type(::MultiFieldParamFESpace{MS,CS,V}) where {MS,CS,V} = eltype(V)

FESpaces.get_vector_type(f::MultiFieldParamFESpace) = f.vector_type

FESpaces.ConstraintStyle(::Type{MultiFieldParamFESpace{S,B,V}}) where {S,B,V} = B()

function FESpaces.get_fe_basis(f::MultiFieldParamFESpace)
  nfields = length(f.spaces)
  all_febases = MultiField.MultiFieldFEBasisComponent[]
  for field_i in 1:nfields
    dv_i = get_fe_basis(f.spaces[field_i])
    @assert BasisStyle(dv_i) == FESpaces.TestBasis()
    dv_i_b = MultiField.MultiFieldFEBasisComponent(dv_i,field_i,nfields)
    push!(all_febases,dv_i_b)
  end
  MultiFieldCellField(all_febases)
end

function FESpaces.get_trial_fe_basis(f::MultiFieldParamFESpace)
  nfields = length(f.spaces)
  all_febases = MultiField.MultiFieldFEBasisComponent[]
  for field_i in 1:nfields
    du_i = get_trial_fe_basis(f.spaces[field_i])
    @assert BasisStyle(du_i) == FESpaces.TrialBasis()
    du_i_b = MultiField.MultiFieldFEBasisComponent(du_i,field_i,nfields)
    push!(all_febases,du_i_b)
  end
  MultiFieldCellField(all_febases)
end

function MultiField.restrict_to_field(f::MultiFieldParamFESpace,free_values::AbstractVector,field::Integer)
  MultiField._restrict_to_field(f,MultiFieldStyle(f),free_values,field)
end

function MultiField._restrict_to_field(
  f,
  mfs::BlockMultiFieldStyle{NB,SB,P},
  free_values::ParamBlockVector,
  field) where {NB,SB,P}

  @check blocklength(free_values) == NB
  U = f.spaces

  block_ranges = MultiField.get_block_ranges(NB,SB,P)
  block_idx    = findfirst(range -> field ∈ range, block_ranges)
  block_free_values = map(free_values) do free_values
    free_values[Block(block_idx)]
  end
  pblock_free_values = ParamArray(block_free_values)

  offsets = compute_field_offsets(f,mfs)
  pini = offsets[field] + 1
  pend = offsets[field] + num_free_dofs(U[field])
  return SubVector(pblock_free_values,pini,pend)
end

function MultiField.compute_field_offsets(f::MultiFieldParamFESpace)
  compute_field_offsets(f,MultiFieldStyle(f))
end

function MultiField.compute_field_offsets(f::MultiFieldParamFESpace,::MultiFieldStyle)
  @notimplemented
end

function MultiField.compute_field_offsets(f::MultiFieldParamFESpace,::ConsecutiveMultiFieldStyle)
  MultiField._compute_field_offsets(f.spaces)
end

function MultiField.compute_field_offsets(f::MultiFieldParamFESpace,::BlockMultiFieldStyle{NB,SB,P}) where {NB,SB,P}
  U = f.spaces
  block_ranges  = MultiField.get_block_ranges(NB,SB,P)
  block_offsets = vcat(map(range->MultiField._compute_field_offsets(U[range]),block_ranges)...)
  offsets = map(p->block_offsets[p],P)
  return offsets
end

function MultiField._compute_field_offsets(spaces::Vector{<:SingleFieldParamFESpace})
  n = length(spaces)
  offsets = zeros(Int,n)
  for i in 1:(n-1)
    Ui = spaces[i]
    offsets[i+1] = offsets[i] + num_free_dofs(Ui)
  end
  return offsets
end

function FESpaces.get_cell_isconstrained(f::MultiFieldParamFESpace)
  msg = """\n
  This method does not make sense for multi-field
  since each field can be defined on a different triangulation.
  Pass a triangulation in the second argument to get
  the constrain flag for the corresponding cells.
  """
  trians = map(get_triangulation,f.spaces)
  trian = first(trians)
  @check all(map(t->t===trian,trians)) msg
  get_cell_isconstrained(f,trian)
end

function FESpaces.get_cell_isconstrained(f::MultiFieldParamFESpace,trian::Triangulation)
  data = map(f.spaces) do space
    trian_i = get_triangulation(space)
    if is_change_possible(trian_i,trian)
      get_cell_isconstrained(space,trian)
    else
      Fill(false,num_cells(trian))
    end
  end
  lazy_map((args...) -> +(args...)>0, data...)
end

function FESpaces.get_cell_is_dirichlet(f::MultiFieldParamFESpace)
  msg = """\n
  This method does not make sense for multi-field
  since each field can be defined on a different triangulation.
  Pass a triangulation in the second argument to get
  the constrain flag for the corresponding cells.
  """
  trians = map(get_triangulation,f.spaces)
  trian = first(trians)
  @check all(map(t->t===trian,trians)) msg
  get_cell_is_dirichlet(f,trian)
end

function FESpaces.get_cell_is_dirichlet(f::MultiFieldParamFESpace,trian::Triangulation)
  data = map(f.spaces) do space
    trian_i = get_triangulation(space)
    if is_change_possible(trian_i,trian)
      get_cell_is_dirichlet(space,trian)
    else
      Fill(false,num_cells(trian))
    end
  end
  lazy_map((args...) -> +(args...)>0, data...)
end

function FESpaces.get_cell_constraints(f::MultiFieldParamFESpace)
  msg = """\n
  This method does not make sense for multi-field
  since each field can be defined on a different triangulation.
  Pass a triangulation in the second argument to get
  the constrains for the corresponding cells.
  """
  trians = map(get_triangulation,f.spaces)
  trian = first(trians)
  @check all(map(t->t===trian,trians)) msg
  get_cell_constraints(f,trian)
end

function FESpaces.get_cell_constraints(f::MultiFieldParamFESpace,trian::Triangulation)
  nfields = length(f.spaces)
  blockmask = [is_change_possible(get_triangulation(Vi),trian) for Vi in f.spaces]
  active_block_ids = findall(blockmask)
  active_block_data = Any[get_cell_constraints(f.spaces[i],trian) for i in active_block_ids]
  blockshape = (nfields,nfields)
  blockindices = [(i,i) for i in active_block_ids]
  lazy_map(BlockMap(blockshape,blockindices),active_block_data...)
end

function FESpaces.get_cell_dof_ids(f::MultiFieldParamFESpace)
  msg = """\n
  This method does not make sense for multi-field
  since each field can be defined on a different triangulation.
  Pass a triangulation in the second argument to get the DOF ids
  on top of the corresponding cells.
  """
  trians = map(get_triangulation,f.spaces)
  trian = first(trians)
  @check all(map(t->t===trian,trians)) msg
  get_cell_dof_ids(f,trian)
end

function FESpaces.get_cell_dof_ids(f::MultiFieldParamFESpace,trian::Triangulation)
  get_cell_dof_ids(f,trian,MultiFieldStyle(f))
end

function FESpaces.get_cell_dof_ids(::MultiFieldParamFESpace,::Triangulation,::MultiFieldStyle)
  @notimplemented
end

function FESpaces.get_cell_dof_ids(
  f::MultiFieldParamFESpace,
  trian::Triangulation,
  ::Union{<:ConsecutiveMultiFieldStyle,<:BlockMultiFieldStyle})

  offsets = compute_field_offsets(f)
  nfields = length(f.spaces)
  blockmask = [is_change_possible(get_triangulation(Vi),trian) for Vi in f.spaces]
  active_block_ids = findall(blockmask)
  active_block_data = Any[]
  for i in active_block_ids
    cell_dofs_i = get_cell_dof_ids(f.spaces[i],trian)
    if i == 1
      push!(active_block_data,cell_dofs_i)
    else
      offset = Int32(offsets[i])
      o = Fill(offset,length(cell_dofs_i))
      cell_dofs_i_b = lazy_map(Broadcasting(MultiField._sum_if_first_positive),cell_dofs_i,o)
      push!(active_block_data,cell_dofs_i_b)
    end
  end
  lazy_map(BlockMap(nfields,active_block_ids),active_block_data...)
end

function MultiField.num_fields(f::MultiFieldParamFESpace)
  length(f.spaces)
end

Base.iterate(m::MultiFieldParamFESpace) = iterate(m.spaces)
Base.iterate(m::MultiFieldParamFESpace,state) = iterate(m.spaces,state)
Base.getindex(m::MultiFieldParamFESpace,::Colon) = m
Base.getindex(m::MultiFieldParamFESpace,field_id::Integer) = m.spaces[field_id]
Base.length(m::MultiFieldParamFESpace) = length(m.spaces)
length_dirichlet_values(m::MultiFieldParamFESpace) = length_dirichlet_values(first(m.spaces))

function FESpaces.FEFunction(fe::MultiFieldParamFESpace,free_values)
  blocks = map(1:length(fe.spaces)) do i
    free_values_i = restrict_to_field(fe,free_values,i)
    FEFunction(fe.spaces[i],free_values_i)
  end
  MultiFieldParamFEFunction(free_values,fe,blocks)
end

function FESpaces.EvaluationFunction(fe::MultiFieldParamFESpace,free_values)
  blocks = map(1:length(fe.spaces)) do i
    free_values_i = restrict_to_field(fe,free_values,i)
    EvaluationFunction(fe.spaces[i],free_values_i)
  end
  MultiFieldParamFEFunction(free_values,fe,blocks)
end

function CellData.CellField(fe::MultiFieldParamFESpace,cell_values)
  single_fields = map(1:length(fe.spaces)) do i
    cell_values_field = lazy_map(a->a[i],cell_values)
    CellField(fe.spaces[i],cell_values_field)
  end
  MultiFieldCellField(single_fields)
end

function FESpaces.interpolate(objects,fe::MultiFieldParamFESpace)
  free_values = zero_free_values(fe)
  interpolate!(objects,free_values,fe)
end

function FESpaces.interpolate!(objects,free_values::AbstractVector,fe::MultiFieldParamFESpace)
  blocks = SingleFieldParamFEFunction[]
  for (field,(U,object)) in enumerate(zip(fe.spaces,objects))
    free_values_i = restrict_to_field(fe,free_values,field)
    uhi = interpolate!(object, free_values_i,U)
    push!(blocks,uhi)
  end
  MultiFieldParamFEFunction(free_values,fe,blocks)
end

function FESpaces.interpolate_everywhere(objects,fe::MultiFieldParamFESpace)
  free_values = zero_free_values(fe)
  blocks = SingleFieldParamFEFunction[]
  for (field,(U,object)) in enumerate(zip(fe.spaces,objects))
    free_values_i = restrict_to_field(fe,free_values,field)
    dirichlet_values_i = zero_dirichlet_values(U)
    uhi = interpolate_everywhere!(object, free_values_i,dirichlet_values_i,U)
    push!(blocks,uhi)
  end
  MultiFieldParamFEFunction(free_values,fe,blocks)
end

function FESpaces.interpolate_dirichlet(objects,fe::MultiFieldParamFESpace)
  free_values = zero_free_values(fe)
  blocks = SingleFieldParamFEFunction[]
  for (field,(U,object)) in enumerate(zip(fe.spaces,objects))
    free_values_i = restrict_to_field(fe,free_values,field)
    dirichlet_values_i = zero_dirichlet_values(U)
    uhi = interpolate_dirichlet!(object,free_values_i,dirichlet_values_i,U)
    push!(blocks,uhi)
  end
  MultiFieldParamFEFunction(free_values,fe,blocks)
end

function FESpaces.test_fe_space(f::ParamFESpace)
  trian = get_triangulation(f)
  @test isa(trian,Triangulation)
  free_values = zero_free_values(f)
  @test length(testitem(free_values)) == num_free_dofs(f)
  V = get_vector_type(f)
  @test typeof(free_values) == V
  fe_function = FEFunction(f,free_values)
  test_fe_function(fe_function)
  fe_basis = get_fe_basis(f)
  @test isa(has_constraints(f),Bool)
  @test isa(has_constraints(typeof(f)),Bool)
  @test length(get_cell_dof_ids(f,trian)) == num_cells(fe_basis)
  @test length(get_cell_constraints(f,trian)) == num_cells(fe_basis)
  @test length(get_cell_isconstrained(f,trian)) == num_cells(fe_basis)
  @test CellField(f,get_cell_dof_ids(f,trian)) != nothing
end
