struct PTrialFESpace{S} <: SingleFieldFESpace
  dirichlet_values::AbstractVector
  space::S
  function PTrialFESpace(dirichlet_values::AbstractVector,space::SingleFieldFESpace)
    new{typeof(space)}(dirichlet_values,space)
  end
end

function PTrialFESpace(space::SingleFieldFESpace,objects)
  dir_values = compute_dirichlet_values_for_tags(space,objects)
  PTrialFESpace(dir_values,space)
end

function PTrialFESpace!(dir_values::PTArray,space::SingleFieldFESpace,objects)
  dir_values_scratch = zero_dirichlet_values(space)
  dir_values = compute_dirichlet_values_for_tags!(dir_values,dir_values_scratch,space,objects)
  PTrialFESpace!(dir_values,space)
end

function PTrialFESpace!(space::PTrialFESpace,objects)
  dir_values = get_dirichlet_dof_values(space)
  dir_values_scratch = zero_dirichlet_values(space)
  dir_values = compute_dirichlet_values_for_tags!(dir_values,dir_values_scratch,space,objects)
  space
end

function HomogeneousPTrialFESpace(U::SingleFieldFESpace,n::Int)
  dv = zero_dirichlet_values(U)
  array = Vector{typeof(dv)}(undef,n)
  @inbounds for i in eachindex(array)
    array[i] = copy(dv)
  end
  dir_values = PTArray(array)
  PTrialFESpace(dir_values,U)
end

FESpaces.get_free_dof_ids(f::PTrialFESpace) = get_free_dof_ids(f.space)

FESpaces.get_triangulation(f::PTrialFESpace) = get_triangulation(f.space)

FESpaces.get_dof_value_type(f::PTrialFESpace) = get_dof_value_type(f.space)

FESpaces.get_vector_type(f::PTrialFESpace) = get_vector_type(f.space)

FESpaces.get_cell_dof_ids(f::PTrialFESpace) = get_cell_dof_ids(f.space)

FESpaces.get_fe_basis(f::PTrialFESpace) = get_fe_basis(f.space)

FESpaces.get_trial_fe_basis(f::PTrialFESpace) = get_trial_fe_basis(f.space)

FESpaces.get_fe_dof_basis(f::PTrialFESpace) = get_fe_dof_basis(f.space)

FESpaces.ConstraintStyle(::Type{<:PTrialFESpace{B}}) where B = ConstraintStyle(B)

FESpaces.get_cell_isconstrained(f::PTrialFESpace) = get_cell_isconstrained(f.space)

FESpaces.get_cell_constraints(f::PTrialFESpace) = get_cell_constraints(f.space)

FESpaces.get_dirichlet_dof_ids(f::PTrialFESpace) = get_dirichlet_dof_ids(f.space)

FESpaces.get_cell_is_dirichlet(f::PTrialFESpace) = get_cell_is_dirichlet(f.space)

FESpaces.num_dirichlet_tags(f::PTrialFESpace) = num_dirichlet_tags(f.space)

FESpaces.get_dirichlet_dof_tag(f::PTrialFESpace) = get_dirichlet_dof_tag(f.space)

FESpaces.get_dirichlet_dof_values(f::PTrialFESpace) = f.dirichlet_values

FESpaces.scatter_free_and_dirichlet_values(f::PTrialFESpace,fv,dv) = scatter_free_and_dirichlet_values(f.space,fv,dv)

FESpaces.gather_free_and_dirichlet_values(f::PTrialFESpace,cv) = gather_free_and_dirichlet_values(f.space,cv)

FESpaces.gather_free_and_dirichlet_values!(fv,dv,f::PTrialFESpace,cv) = gather_free_and_dirichlet_values!(fv,dv,f.space,cv)

FESpaces.gather_dirichlet_values(f::PTrialFESpace,cv) = gather_dirichlet_values(f.space,cv)

FESpaces.gather_dirichlet_values!(dv,f::PTrialFESpace,cv) = gather_dirichlet_values!(dv,f.space,cv)

FESpaces.gather_free_values(f::PTrialFESpace,cv) = gather_free_values(f.space,cv)

FESpaces.gather_free_values!(fv,f::PTrialFESpace,cv) = gather_free_values!(fv,f.space,cv)

function FESpaces.zero_free_values(f::PTrialFESpace)
  fv = zero_free_values(f.space)
  n = length(f.dirichlet_values)
  array = Vector{typeof(fv)}(undef,n)
  @inbounds for i in eachindex(array)
    array[i] = copy(fv)
  end
  PTArray(array)
end

function FESpaces.zero_dirichlet_values(f::PTrialFESpace)
  zdv = zero_dirichlet_values(f.space)
  n = length(f.dirichlet_values)
  array = Vector{typeof(zdv)}(undef,n)
  @inbounds for i in eachindex(array)
    array[i] = copy(zdv)
  end
  PTArray(array)
end

function FESpaces.compute_dirichlet_values_for_tags!(
  dirichlet_values::PTArray{T},
  dirichlet_values_scratch::PTArray{T},
  f::PTrialFESpace,
  tag_to_object) where T

  dirichlet_dof_to_tag = get_dirichlet_dof_tag(f)
  @inbounds for n in eachindex(dirichlet_values)
    dv = dirichlet_values[n]
    dvs = dirichlet_values_scratch[n]
    _tag_to_object = FESpaces._convert_to_collectable(tag_to_object[n],num_dirichlet_tags(f))
    fill!(dvs,zero(eltype(T)))
    for (tag,object) in enumerate(_tag_to_object)
      cell_vals = FESpaces._cell_vals(f,object)
      gather_dirichlet_values!(dvs,f.space,cell_vals)
      FESpaces._fill_dirichlet_values_for_tag!(dv,dvs,tag,dirichlet_dof_to_tag)
    end
  end
  dirichlet_values
end

# MultiField interface
struct PMultiFieldFESpace{MS<:MultiFieldStyle,CS<:ConstraintStyle,V} <: FESpace
  vector_type::Type{V}
  spaces::Vector{<:PTrialFESpace}
  multi_field_style::MS
  constraint_style::CS
  function PMultiFieldFESpace(
    ::Type{V},
    spaces::Vector{<:PTrialFESpace},
    multi_field_style::MultiFieldStyle) where V
    @assert length(spaces) > 0

    MS = typeof(multi_field_style)
    if any(map(has_constraints,spaces))
      constraint_style = Constrained()
    else
      constraint_style = UnConstrained()
    end
    CS = typeof(constraint_style)
    @assert all([length(fe.dirichlet_values) == length(first(spaces).dirichlet_values) for fe in spaces])
    new{MS,CS,V}(V,spaces,multi_field_style,constraint_style)
  end
end

function PMultiFieldFESpace(spaces::Vector{<:SingleFieldFESpace})
  Ts = map(get_dof_value_type,spaces)
  T = typeof(*(map(zero,Ts)...))
  PMultiFieldFESpace(Vector{T},spaces,ConsecutiveMultiFieldStyle())
end

function PMultiFieldFESpace(::Type{V},spaces::Vector{<:SingleFieldFESpace}) where V
  PMultiFieldFESpace(V,spaces,ConsecutiveMultiFieldStyle())
end

MultiField.MultiFieldStyle(::Type{PMultiFieldFESpace{S,B,V}}) where {S,B,V} = S()
MultiField.MultiFieldStyle(f::PMultiFieldFESpace) = MultiFieldStyle(typeof(f))

function FESpaces.get_triangulation(f::PMultiFieldFESpace)
  s1 = first(f.spaces)
  trian = get_triangulation(s1)
  @check all(map(i->trian===get_triangulation(i),f.spaces))
  trian
end

function FESpaces.num_free_dofs(f::PMultiFieldFESpace)
  n = 0
  for U in f.spaces
    n += num_free_dofs(U)
  end
  n
end

function FESpaces.get_free_dof_ids(f::PMultiFieldFESpace)
  get_free_dof_ids(f,MultiFieldStyle(f))
end

function FESpaces.get_free_dof_ids(::PMultiFieldFESpace,::MultiFieldStyle)
  @abstractmethod
end

function FESpaces.get_free_dof_ids(f::PMultiFieldFESpace,::ConsecutiveMultiFieldStyle)
  block_num_dofs = Int[]
  for U in f.spaces
    push!(block_num_dofs,num_free_dofs(U))
  end
  blockedrange(block_num_dofs)
end

FESpaces.get_dof_value_type(::PMultiFieldFESpace{MS,CS,V}) where {MS,CS,V} = eltype(V)

FESpaces.get_vector_type(f::PMultiFieldFESpace) = f.vector_type

FESpaces.ConstraintStyle(::Type{PMultiFieldFESpace{S,B,V}}) where {S,B,V} = B()

function FESpaces.zero_free_values(f::PMultiFieldFESpace)
  vcat([zero_free_values(fe) for fe in f.spaces]...)
end

function block_zero_free_values(f::PMultiFieldFESpace)
  [zero_free_values(fe) for fe in f.spaces]
end

function FESpaces.get_fe_basis(f::PMultiFieldFESpace)
  nfields = length(f.spaces)
  all_febases = MultiFieldFEBasisComponent[]
  for field_i in 1:nfields
    dv_i = get_fe_basis(f.spaces[field_i])
    @assert BasisStyle(dv_i) == FESpaces.TestBasis()
    dv_i_b = MultiFieldFEBasisComponent(dv_i,field_i,nfields)
    push!(all_febases,dv_i_b)
  end
  MultiFieldCellField(all_febases)
end

function FESpaces.get_trial_fe_basis(f::PMultiFieldFESpace)
  nfields = length(f.spaces)
  all_febases = MultiFieldFEBasisComponent[]
  for field_i in 1:nfields
    du_i = get_trial_fe_basis(f.spaces[field_i])
    @assert BasisStyle(du_i) == FESpaces.TrialBasis()
    du_i_b = MultiFieldFEBasisComponent(du_i,field_i,nfields)
    push!(all_febases,du_i_b)
  end
  MultiFieldCellField(all_febases)
end

function FESpaces.FEFunction(fe::PMultiFieldFESpace,free_values)
  blocks = map(1:length(fe.spaces)) do i
    free_values_i = restrict_to_field(fe,free_values,i)
    FEFunction(fe.spaces[i],free_values_i)
  end
  PTMultiFieldFEFunction(free_values,fe,blocks)
end

function CellData.CellField(fe::PMultiFieldFESpace,cell_values)
  single_fields = map(1:length(fe.spaces)) do i
    cell_values_field = lazy_map(a->a.array[i],cell_values)
    CellField(fe.spaces[i],cell_values_field)
  end
  MultiFieldCellField(single_fields)
end

function split_fields(fe::Union{PMultiFieldFESpace,MultiFieldFESpace},free_values::PTArray)
  offsets = compute_field_offsets(fe)
  fields = map(1:length(fe.spaces)) do field
    pini = offsets[field] + 1
    pend = offsets[field] + num_free_dofs(fe.spaces[field])
    map(x->getindex(x,pini:pend),free_values)
  end
  fields
end

function MultiField.restrict_to_field(f::PMultiFieldFESpace,free_values::PTArray,field::Integer)
  MultiField._restrict_to_field(f,MultiFieldStyle(f),free_values,field)
end

function MultiField._restrict_to_field(
  f::PMultiFieldFESpace,
  ::ConsecutiveMultiFieldStyle,
  free_values::PTArray,
  field::Integer)

  offsets = compute_field_offsets(f)
  U = f.spaces
  pini = offsets[field] + 1
  pend = offsets[field] + num_free_dofs(U[field])
  map(fv -> SubVector(fv,pini,pend),free_values)
end

function MultiField.compute_field_offsets(f::PMultiFieldFESpace)
  @assert MultiFieldStyle(f) == ConsecutiveMultiFieldStyle()
  U = f.spaces
  n = length(U)
  offsets = zeros(Int,n)
  for i in 1:(n-1)
    Ui = U[i]
    offsets[i+1] = offsets[i] + num_free_dofs(Ui)
  end
  offsets
end

function FESpaces.get_cell_isconstrained(f::PMultiFieldFESpace)
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

function FESpaces.get_cell_isconstrained(f::PMultiFieldFESpace,trian::Triangulation)
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

function FESpaces.get_cell_is_dirichlet(f::PMultiFieldFESpace)
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

function FESpaces.get_cell_is_dirichlet(f::PMultiFieldFESpace,trian::Triangulation)
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

function FESpaces.get_cell_constraints(f::PMultiFieldFESpace)
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

function FESpaces.get_cell_constraints(f::PMultiFieldFESpace,trian::Triangulation)
  nfields = length(f.spaces)
  blockmask = [is_change_possible(get_triangulation(Vi),trian) for Vi in f.spaces]
  active_block_ids = findall(blockmask)
  active_block_data = Any[get_cell_constraints(f.spaces[i],trian) for i in active_block_ids]
  blockshape = (nfields,nfields)
  blockindices = [(i,i) for i in active_block_ids]
  lazy_map(BlockMap(blockshape,blockindices),active_block_data...)
end

function FESpaces.get_cell_dof_ids(f::PMultiFieldFESpace)
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

function FESpaces.get_cell_dof_ids(f::PMultiFieldFESpace,trian::Triangulation)
  get_cell_dof_ids(f,trian,MultiFieldStyle(f))
end

function FESpaces.get_cell_dof_ids(::PMultiFieldFESpace,::Triangulation,::MultiFieldStyle)
  @notimplemented
end

function FESpaces.get_cell_dof_ids(f::PMultiFieldFESpace,trian::Triangulation,::ConsecutiveMultiFieldStyle)
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

function MultiField.num_fields(f::PMultiFieldFESpace)
  length(f.spaces)
end

Base.iterate(m::PMultiFieldFESpace) = iterate(m.spaces)
Base.iterate(m::PMultiFieldFESpace,state) = iterate(m.spaces,state)
Base.getindex(m::PMultiFieldFESpace,::Colon) = m
Base.getindex(m::PMultiFieldFESpace,field_id::Integer) = m.spaces[field_id]
Base.length(m::PMultiFieldFESpace) = length(m.spaces)

function FESpaces.interpolate!(objects,free_values::PTArray,fe::PMultiFieldFESpace)
  block_free_values = block_zero_free_values(fe)
  blocks = PTSingleFieldFEFunction[]
  for (free_values_i,U,object) in zip(block_free_values,fe.spaces,objects)
    uhi = interpolate!(object,free_values_i,U)
    push!(blocks,uhi)
  end
  PTMultiFieldFEFunction(free_values,fe,blocks)
end

function FESpaces.interpolate_everywhere(objects,fe::PMultiFieldFESpace)
  free_values = zero_free_values(fe)
  block_free_values = block_zero_free_values(fe)
  blocks = PTSingleFieldFEFunction[]
  for (free_values_i,U,object) in zip(block_free_values,fe.spaces,objects)
    dirichlet_values_i = zero_dirichlet_values(U)
    uhi = interpolate_everywhere!(object,free_values_i,dirichlet_values_i,U)
    push!(blocks,uhi)
  end
  PTMultiFieldFEFunction(free_values,fe,blocks)
end

function FESpaces.interpolate_dirichlet(objects,fe::PMultiFieldFESpace)
  free_values = zero_free_values(fe)
  block_free_values = block_zero_free_values(fe)
  blocks = PTSingleFieldFEFunction[]
  for (free_values_i,U,object) in zip(block_free_values,fe.spaces,objects)
    dirichlet_values_i = zero_dirichlet_values(U)
    uhi = interpolate_dirichlet!(object,free_values_i,dirichlet_values_i,U)
    push!(blocks,uhi)
  end
  PTMultiFieldFEFunction(free_values,fe,blocks)
end

function FESpaces.EvaluationFunction(fe::PMultiFieldFESpace,free_values::PTArray)
  blocks = map(eachindex(fe.spaces)) do i
    free_values_i = restrict_to_field(fe,free_values,i)
    fe_function_i = EvaluationFunction(fe.spaces[i],free_values_i)
    free_values_i,fe_function_i
  end
  free_values = vcat(first.(blocks)...)
  fe_functions = last.(blocks)
  PTMultiFieldFEFunction(free_values,fe,fe_functions)
end

function Arrays.testitem(f::PMultiFieldFESpace)
  MultiFieldFESpace(f.vector_type,map(testitem,f.spaces),f.multi_field_style)
end

function field_offsets(f::Union{MultiFieldFESpace,PMultiFieldFESpace})
  [compute_field_offsets(f)...,num_free_dofs(f)]
end
