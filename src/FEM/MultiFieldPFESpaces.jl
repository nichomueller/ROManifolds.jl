struct MultiFieldPFESpace{MS<:MultiFieldStyle,CS<:ConstraintStyle,V} <: FESpace
  vector_type::Type{V}
  spaces::Vector{<:TrialPFESpace}
  multi_field_style::MS
  constraint_style::CS
  function MultiFieldPFESpace(
    ::Type{V},
    spaces::Vector{<:TrialPFESpace},
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

const PFESpace = Union{TrialPFESpace,MultiFieldPFESpace}

function MultiFieldPFESpace(
  spaces::Vector{<:TrialPFESpace};
  style = ConsecutiveMultiFieldStyle())

  Ts = map(get_dof_value_type,spaces)
  T  = typeof(*(map(zero,Ts)...))
  if isa(style,BlockMultiFieldStyle)
    style = BlockMultiFieldStyle(style,spaces)
    V = map(spaces) do space
      zero_free_values(space.space)
    end
    VT = typeof(mortar(V))
  else
    VT = Vector{T}
  end
  MultiFieldPFESpace(VT,spaces,style)
end

function MultiFieldPFESpace(::Type{V},spaces::Vector{<:TrialPFESpace}) where V
  MultiFieldPFESpace(V,spaces,ConsecutiveMultiFieldStyle())
end

function MultiFieldPFESpace(spaces::Vector{<:SingleFieldFESpace};kwargs...)
  @notimplemented
end

function MultiFieldPFESpace(::Type{V},spaces::Vector{<:SingleFieldFESpace}) where V
  @notimplemented
end

MultiField.MultiFieldStyle(::Type{MultiFieldPFESpace{S,B,V}}) where {S,B,V} = S()
MultiField.MultiFieldStyle(f::MultiFieldPFESpace) = MultiFieldStyle(typeof(f))

function FESpaces.get_triangulation(f::MultiFieldPFESpace)
  s1 = first(f.spaces)
  trian = get_triangulation(s1)
  @check all(map(i->trian===get_triangulation(i),f.spaces))
  trian
end

function FESpaces.num_free_dofs(f::MultiFieldPFESpace)
  n = 0
  for U in f.spaces
    n += num_free_dofs(U)
  end
  n
end

function FESpaces.get_free_dof_ids(f::MultiFieldPFESpace)
  get_free_dof_ids(f,MultiFieldStyle(f))
end

function FESpaces.get_free_dof_ids(::MultiFieldPFESpace,::MultiFieldStyle)
  @abstractmethod
end

function FESpaces.get_free_dof_ids(f::MultiFieldPFESpace,::ConsecutiveMultiFieldStyle)
  block_num_dofs = Int[]
  for U in f.spaces
    push!(block_num_dofs,num_free_dofs(U))
  end
  blockedrange(block_num_dofs)
end

function FESpaces.get_free_dof_ids(f::MultiFieldPFESpace,::BlockMultiFieldStyle{NB,SB,P}) where {NB,SB,P}
  block_ranges   = get_block_ranges(NB,SB,P)
  block_num_dofs = map(range->sum(map(num_free_dofs,f.spaces[range])),block_ranges)
  return BlockArrays.blockedrange(block_num_dofs)
end

function block_zero_free_values(f::MultiFieldPFESpace)
  map(f.spaces) do fe
    zero_free_values(fe)
  end
end

function FESpaces.zero_free_values(f::MultiFieldPFESpace)
  vcat(block_zero_free_values(f)...)
end

function FESpaces.zero_free_values(f::MultiFieldPFESpace{<:BlockMultiFieldStyle{NB,SB,P}}) where {NB,SB,P}
  block_ranges   = get_block_ranges(NB,SB,P)
  block_num_dofs = map(range->sum(map(num_free_dofs,f.spaces[range])),block_ranges)
  block_vtypes   = map(range->get_vector_type(first(f.spaces[range])),block_ranges)
  array = map(1:length(first(f.spaces).dirichlet_values)) do i
    mortar(map(allocate_vector,block_vtypes,block_num_dofs))
  end
  return array
end

FESpaces.get_dof_value_type(::MultiFieldPFESpace{MS,CS,V}) where {MS,CS,V} = eltype(V)

FESpaces.get_vector_type(f::MultiFieldPFESpace) = f.vector_type

FESpaces.ConstraintStyle(::Type{MultiFieldPFESpace{S,B,V}}) where {S,B,V} = B()

function FESpaces.get_fe_basis(f::MultiFieldPFESpace)
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

function FESpaces.get_trial_fe_basis(f::MultiFieldPFESpace)
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

function split_fields(fe::Union{MultiFieldPFESpace,MultiFieldFESpace},free_values::PArray)
  offsets = compute_field_offsets(fe)
  fields = map(1:length(fe.spaces)) do field
    pini = offsets[field] + 1
    pend = offsets[field] + num_free_dofs(fe.spaces[field])
    map(x->getindex(x,pini:pend),free_values)
  end
  fields
end

function MultiField.restrict_to_field(f::MultiFieldPFESpace,free_values::PArray,field::Integer)
  MultiField._restrict_to_field(f,MultiFieldStyle(f),free_values,field)
end

function MultiField._restrict_to_field(
  f::MultiFieldPFESpace,
  ::Union{<:ConsecutiveMultiFieldStyle,<:BlockMultiFieldStyle},
  free_values::PArray,
  field::Integer)

  offsets = compute_field_offsets(f)
  U = f.spaces
  pini = offsets[field] + 1
  pend = offsets[field] + num_free_dofs(U[field])
  map(fv -> SubVector(fv,pini,pend),free_values)
end

function MultiField._restrict_to_field(
  f,
  mfs::BlockMultiFieldStyle{NB,SB,P},
  free_values::PArray{<:BlockVector},
  field) where {NB,SB,P}

  U = f.spaces

  block_ranges = get_block_ranges(NB,SB,P)
  block_idx    = findfirst(range -> field ∈ range, block_ranges)

  offsets = compute_field_offsets(f,mfs)
  pini = offsets[field] + 1
  pend = offsets[field] + num_free_dofs(U[field])

  map(free_values) do free_values
    @check blocklength(free_values) == NB
    block_free_values = free_values[Block(block_idx)]
    SubVector(block_free_values,pini,pend)
  end
end

function MultiField.compute_field_offsets(f::MultiFieldPFESpace)
  compute_field_offsets(f,MultiFieldStyle(f))
end

function MultiField.compute_field_offsets(f::MultiFieldPFESpace,::MultiFieldStyle)
  @notimplemented
end

function MultiField.compute_field_offsets(f::MultiFieldPFESpace,::ConsecutiveMultiFieldStyle)
  MultiField._compute_field_offsets(f.spaces)
end

function MultiField.compute_field_offsets(f::MultiFieldPFESpace,::BlockMultiFieldStyle{NB,SB,P}) where {NB,SB,P}
  U = f.spaces
  block_ranges  = get_block_ranges(NB,SB,P)
  block_offsets = vcat(map(range->MultiField._compute_field_offsets(U[range]),block_ranges)...)
  offsets = map(p->block_offsets[p],P)
  return offsets
end

function MultiField._compute_field_offsets(spaces::Vector{<:TrialPFESpace})
  n = length(spaces)
  offsets = zeros(Int,n)
  for i in 1:(n-1)
    Ui = spaces[i]
    offsets[i+1] = offsets[i] + num_free_dofs(Ui)
  end
  return offsets
end

function FESpaces.get_cell_isconstrained(f::MultiFieldPFESpace)
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

function FESpaces.get_cell_isconstrained(f::MultiFieldPFESpace,trian::Triangulation)
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

function FESpaces.get_cell_is_dirichlet(f::MultiFieldPFESpace)
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

function FESpaces.get_cell_is_dirichlet(f::MultiFieldPFESpace,trian::Triangulation)
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

function FESpaces.get_cell_constraints(f::MultiFieldPFESpace)
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

function FESpaces.get_cell_constraints(f::MultiFieldPFESpace,trian::Triangulation)
  nfields = length(f.spaces)
  blockmask = [is_change_possible(get_triangulation(Vi),trian) for Vi in f.spaces]
  active_block_ids = findall(blockmask)
  active_block_data = Any[get_cell_constraints(f.spaces[i],trian) for i in active_block_ids]
  blockshape = (nfields,nfields)
  blockindices = [(i,i) for i in active_block_ids]
  lazy_map(BlockMap(blockshape,blockindices),active_block_data...)
end

function FESpaces.get_cell_dof_ids(f::MultiFieldPFESpace)
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

function FESpaces.get_cell_dof_ids(f::MultiFieldPFESpace,trian::Triangulation)
  get_cell_dof_ids(f,trian,MultiFieldStyle(f))
end

function FESpaces.get_cell_dof_ids(::MultiFieldPFESpace,::Triangulation,::MultiFieldStyle)
  @notimplemented
end

function FESpaces.get_cell_dof_ids(f::MultiFieldPFESpace,trian::Triangulation,::ConsecutiveMultiFieldStyle)
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

function MultiField.num_fields(f::MultiFieldPFESpace)
  length(f.spaces)
end

Base.iterate(m::MultiFieldPFESpace) = iterate(m.spaces)
Base.iterate(m::MultiFieldPFESpace,state) = iterate(m.spaces,state)
Base.getindex(m::MultiFieldPFESpace,::Colon) = m
Base.getindex(m::MultiFieldPFESpace,field_id::Integer) = m.spaces[field_id]
Base.length(m::MultiFieldPFESpace) = length(m.spaces)


struct MultiFieldPFEFunction{T<:MultiFieldCellField} <: PCellField
  single_fe_functions::Vector{<:SingleFieldPFEFunction}
  free_values::AbstractArray
  fe_space::MultiFieldPFESpace
  multi_cell_field::T

  function MultiFieldPFEFunction(
    free_values::AbstractVector,
    space::MultiFieldPFESpace,
    single_fe_functions::Vector{<:SingleFieldPFEFunction})

    multi_cell_field = MultiFieldCellField(map(i->i.cell_field,single_fe_functions))
    T = typeof(multi_cell_field)

    new{T}(
      single_fe_functions,
      free_values,
      space,
      multi_cell_field)
  end
end

Base.length(f::MultiFieldPFEFunction) = length(f.free_values)
CellData.get_data(f::MultiFieldPFEFunction) = get_data(f.multi_cell_field)
FESpaces.get_triangulation(f::MultiFieldPFEFunction) = get_triangulation(f.multi_cell_field)
CellData.DomainStyle(::Type{MultiFieldPFEFunction{T}}) where T = DomainStyle(T)
FESpaces.get_free_dof_values(f::MultiFieldPFEFunction) = f.free_values
FESpaces.get_fe_space(f::MultiFieldPFEFunction) = f.fe_space

function FESpaces.get_cell_dof_values(f::MultiFieldPFEFunction)
  msg = """\n
  This method does not make sense for multi-field
  since each field can be defined on a different triangulation.
  Pass a triangulation in the second argument to get the DOF values
  on top of the corresponding cells.
  """
  trians = map(get_triangulation,f.fe_space.spaces)
  trian = first(trians)
  @check all(map(t->is_change_possible(t,trian),trians)) msg
  get_cell_dof_values(f,trian)
end

function FESpaces.get_cell_dof_values(f::MultiFieldPFEFunction,trian::Triangulation)
  uhs = f.single_fe_functions
  blockmask = [is_change_possible(get_triangulation(uh),trian) for uh in uhs]
  active_block_ids = findall(blockmask)
  active_block_data = Any[ get_cell_dof_values(uhs[i],trian) for i in active_block_ids ]
  nblocks = length(uhs)
  lazy_map(BlockMap(nblocks,active_block_ids),active_block_data...)
end

MultiField.num_fields(m::MultiFieldPFEFunction) = length(m.single_fe_functions)
# Base.iterate(m::MultiFieldPFEFunction) = iterate(m.single_fe_functions)
# Base.iterate(m::MultiFieldPFEFunction,state) = iterate(m.single_fe_functions,state)
Base.getindex(m::MultiFieldPFEFunction,field_id::Integer) = m.single_fe_functions[field_id]

function FESpaces.FEFunction(fe::MultiFieldPFESpace,free_values::PArray)
  blocks = map(1:length(fe.spaces)) do i
    free_values_i = restrict_to_field(fe,free_values,i)
    FEFunction(fe.spaces[i],free_values_i)
  end
  MultiFieldPFEFunction(free_values,fe,blocks)
end

function FESpaces.EvaluationFunction(fe::MultiFieldPFESpace,free_values::PArray)
  blocks = map(1:length(fe.spaces)) do i
    free_values_i = restrict_to_field(fe,free_values,i)
    EvaluationFunction(fe.spaces[i],free_values_i)
  end
  MultiFieldFEFunction(free_values,fe,blocks)
end

function CellData.CellField(fe::MultiFieldPFESpace,cell_values)
  single_fields = map(1:length(fe.spaces)) do i
    cell_values_field = lazy_map(a->a[i],cell_values)
    CellField(fe.spaces[i],cell_values_field)
  end
  MultiFieldCellField(single_fields)
end

function TransientFETools.TransientCellField(multi_field::MultiFieldPFEFunction,derivatives::Tuple)
  transient_single_fields = TransientFETools._to_transient_single_fields(multi_field,derivatives)
  TransientFETools.TransientMultiFieldCellField(multi_field,derivatives,transient_single_fields)
end

function FESpaces.interpolate(objects,fe::MultiFieldPFESpace)
  free_values = zero_free_values(fe)
  interpolate!(objects,free_values,fe)
end

function FESpaces.interpolate!(objects,free_values::PArray,fe::MultiFieldPFESpace)
  block_free_values = block_zero_free_values(fe)
  blocks = SingleFieldPFEFunction[]
  for (free_values_i,U,object) in zip(block_free_values,fe.spaces,objects)
    uhi = interpolate!(object,free_values_i,U)
    push!(blocks,uhi)
  end
  MultiFieldPFEFunction(free_values,fe,blocks)
end

function FESpaces.interpolate_everywhere(objects,fe::MultiFieldPFESpace)
  free_values = zero_free_values(fe)
  block_free_values = block_zero_free_values(fe)
  blocks = SingleFieldPFEFunction[]
  for (free_values_i,U,object) in zip(block_free_values,fe.spaces,objects)
    dirichlet_values_i = zero_dirichlet_values(U)
    uhi = interpolate_everywhere!(object,free_values_i,dirichlet_values_i,U)
    push!(blocks,uhi)
  end
  MultiFieldPFEFunction(free_values,fe,blocks)
end

function FESpaces.interpolate_dirichlet(objects,fe::MultiFieldPFESpace)
  free_values = zero_free_values(fe)
  block_free_values = block_zero_free_values(fe)
  blocks = SingleFieldPFEFunction[]
  for (free_values_i,U,object) in zip(block_free_values,fe.spaces,objects)
    dirichlet_values_i = zero_dirichlet_values(U)
    uhi = interpolate_dirichlet!(object,free_values_i,dirichlet_values_i,U)
    push!(blocks,uhi)
  end
  MultiFieldPFEFunction(free_values,fe,blocks)
end

# function FESpaces.EvaluationFunction(fe::MultiFieldPFESpace,free_values::PArray)
#   free_values,fe_functions = map(eachindex(fe.spaces)) do i
#     free_values_i = restrict_to_field(fe,free_values,i)
#     fe_function_i = EvaluationFunction(fe.spaces[i],free_values_i)
#     free_values_i,fe_function_i
#   end |> tuple_of_arrays
#   free_values = vcat(free_values...)
#   MultiFieldPFEFunction(free_values,fe,fe_functions)
# end

# for visualization purposes
function Base.iterate(f::MultiFieldPFEFunction)
  mfs,states = map(iterate,f.single_fe_functions) |> tuple_of_arrays
  sff,statef = map(iterate,f.fe_space) |> tuple_of_arrays
  index = first(first.(statef))
  fv = f.free_values[index]
  MultiFieldPFEFunction(fv,mfs,sff),(states,statef)
end

function Base.iterate(f::MultiFieldPFEFunction,state)
  states,statef = state
  iters = map(f->iterate(f.single_fe_functions,states))
  iterf = map(f->iterate(f.fe_space,statef))
  if isnothing(iters) && isnothing(iterf)
    return nothing
  end
  mfs,states = iters |> tuple_of_arrays
  sff,statef = iterf |> tuple_of_arrays
  index = first(first.(statef))
  fv = f.free_values[index]
  MultiFieldPFEFunction(fv,mfs,sff),(states,statef)
end

function Arrays.testitem(f::MultiFieldPFESpace)
  error("DEPRECADED")
  MultiFieldFESpace(f.vector_type,map(testitem,f.spaces),f.multi_field_style)
end

function field_offsets(f::Union{MultiFieldFESpace,MultiFieldPFESpace})
  error("DEPRECADED")
  [compute_field_offsets(f)...,num_free_dofs(f)]
end
