struct TrialPFESpace{S} <: SingleFieldFESpace
  dirichlet_values::PTArray
  space::S
  function TrialPFESpace(dirichlet_values::PTArray,space::SingleFieldFESpace)
    new{typeof(space)}(dirichlet_values,space)
  end
end

function TrialPFESpace(U::SingleFieldFESpace,dirichlet_values::PTArray)
  TrialPFESpace(dirichlet_values,U)
end

function HomogeneousTrialPFESpace(U::SingleFieldFESpace,n::Int)
  dv = zero_dirichlet_values(U)
  array = Vector{typeof(dv)}(undef,n)
  @inbounds for i in eachindex(array)
    array[i] = copy(dv)
  end
  dirichlet_values = PTArray(array)
  TrialPFESpace(dirichlet_values,U)
end

function TrialPFESpace(space::SingleFieldFESpace,objects)
  dirichlet_values = compute_dirichlet_values_for_tags(space,objects)
  TrialPFESpace(dirichlet_values,space)
end

function TrialPFESpace!(dir_values::PTArray,space::SingleFieldFESpace,objects)
  dir_values_scratch = zero_dirichlet_values(space)
  dir_values = compute_dirichlet_values_for_tags!(dir_values,dir_values_scratch,space,objects)
  TrialPFESpace!(dir_values,space)
end

function TrialPFESpace!(space::TrialPFESpace,objects)
  dir_values = get_dirichlet_dof_values(space)
  dir_values_scratch = zero_dirichlet_values(space)
  dir_values = compute_dirichlet_values_for_tags!(dir_values,dir_values_scratch,space,objects)
  space
end

FESpaces.get_free_dof_ids(f::TrialPFESpace) = get_free_dof_ids(f.space)

FESpaces.get_triangulation(f::TrialPFESpace) = get_triangulation(f.space)

FESpaces.get_dof_value_type(f::TrialPFESpace) = get_dof_value_type(f.space)

FESpaces.get_vector_type(f::TrialPFESpace) = get_vector_type(f.space)

FESpaces.get_cell_dof_ids(f::TrialPFESpace) = get_cell_dof_ids(f.space)

FESpaces.get_fe_basis(f::TrialPFESpace) = get_fe_basis(f.space)

FESpaces.get_trial_fe_basis(f::TrialPFESpace) = get_trial_fe_basis(f.space)

FESpaces.get_fe_dof_basis(f::TrialPFESpace) = get_fe_dof_basis(f.space)

FESpaces.ConstraintStyle(::Type{<:TrialPFESpace{B}}) where B = ConstraintStyle(B)

FESpaces.get_cell_isconstrained(f::TrialPFESpace) = get_cell_isconstrained(f.space)

FESpaces.get_cell_constraints(f::TrialPFESpace) = get_cell_constraints(f.space)

FESpaces.get_dirichlet_dof_ids(f::TrialPFESpace) = get_dirichlet_dof_ids(f.space)

FESpaces.get_cell_is_dirichlet(f::TrialPFESpace) = get_cell_is_dirichlet(f.space)

FESpaces.num_dirichlet_tags(f::TrialPFESpace) = num_dirichlet_tags(f.space)

FESpaces.get_dirichlet_dof_tag(f::TrialPFESpace) = get_dirichlet_dof_tag(f.space)

FESpaces.get_dirichlet_dof_values(f::TrialPFESpace) = f.dirichlet_values

FESpaces.scatter_free_and_dirichlet_values(f::TrialPFESpace,fv,dv) = scatter_free_and_dirichlet_values(f.space,fv,dv)

# These functions allow us to pass from cell-wise PTArray(s) to global PTArray(s)
function FESpaces.zero_free_values(f::TrialPFESpace)
  fv = zero_free_values(f.space)
  n = length(f.dirichlet_values)
  array = Vector{typeof(fv)}(undef,n)
  @inbounds for i in eachindex(array)
    array[i] = copy(fv)
  end
  PTArray(array)
end

function FESpaces.zero_dirichlet_values(f::TrialPFESpace)
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
  f::TrialPFESpace,
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

function FESpaces.gather_free_and_dirichlet_values(fs::TrialPFESpace,cell_vals)
  free_values = zero_free_values(fs)
  dirichlet_values = zero_dirichlet_values(fs)
  gather_free_and_dirichlet_values!(free_values,dirichlet_values,fs,cell_vals)
end

function FESpaces.gather_dirichlet_values(fs::TrialPFESpace,cell_vals)
  dirichlet_values = zero_dirichlet_values(fs)
  gather_dirichlet_values!(dirichlet_values,fs,cell_vals)
  dirichlet_values
end

function FESpaces.gather_free_values(fs::TrialPFESpace,cell_vals)
  free_values = zero_free_values(fs)
  gather_free_values!(free_values,fs,cell_vals)
  free_values
end

function FESpaces.gather_free_and_dirichlet_values!(
  free_vals,
  dirichlet_vals,
  f::TrialPFESpace,
  cell_vals)

  cell_dofs = get_cell_dof_ids(f)
  cache_vals = array_cache(cell_vals)
  cache_dofs = array_cache(cell_dofs)
  cells = 1:length(cell_vals)

  FESpaces._free_and_dirichlet_values_fill!(
    free_vals,
    dirichlet_vals,
    cache_vals,
    cache_dofs,
    cell_vals,
    cell_dofs,
    cells)

  (free_vals,dirichlet_vals)
end

function FESpaces.gather_free_values!(free_values,f::TrialPFESpace,cell_vals)
  dirichlet_values = zero_dirichlet_values(f)
  gather_free_and_dirichlet_values!(free_values,dirichlet_values,f,cell_vals)
  free_values
end

function FESpaces.gather_dirichlet_values!(
  dirichlet_vals,
  f::TrialPFESpace,
  cell_vals)

  cell_dofs = get_cell_dof_ids(f)
  cache_vals = array_cache(cell_vals)
  cache_dofs = array_cache(cell_dofs)
  free_vals = zero_free_values(f)
  cells = f.dirichlet_cells

  FESpaces._free_and_dirichlet_values_fill!(
    free_vals,
    dirichlet_vals,
    cache_vals,
    cache_dofs,
    cell_vals,
    cell_dofs,
    cells)

  dirichlet_vals
end

function FESpaces._free_and_dirichlet_values_fill!(
  free_vals::PTArray,
  dirichlet_vals::PTArray,
  cache_vals,
  cache_dofs,
  cell_vals::PTArray,
  cell_dofs::PTArray,
  cells)

  for cell in cells
    vals = getindex!(cache_vals,cell_vals,cell)
    dofs = getindex!(cache_dofs,cell_dofs,cell)
    for (i,dof) in enumerate(dofs)
      for k in eachindex(vals)
        val = vals[k][i]
        if dof > 0
          free_vals[dof] = val
        elseif dof < 0
          dirichlet_vals[-dof] = val
        else
          @unreachable "dof ids either positive or negative, not zero"
        end
      end
    end
  end

end

function FESpaces.interpolate!(
  object::AbstractPTFunction,
  free_values::PTArray,
  fs::TrialPFESpace)

  for k in eachindex(object)
    cell_vals = FESpaces._cell_vals(fs,object[k])
    gather_free_values!(free_values[k],fs,cell_vals)
  end
  FEFunction(fs,free_values)
end

function FESpaces.interpolate_everywhere!(
  object::AbstractPTFunction,
  free_values::PTArray,
  dirichlet_values::PTArray,
  fs::TrialPFESpace)

  for k in eachindex(object)
    cell_vals = FESpaces._cell_vals(fs,object[k])
    gather_free_and_dirichlet_values!(free_values[k],dirichlet_values[k],fs,cell_vals)
  end
  FEFunction(fs,free_values,dirichlet_values)
end

function FESpaces.interpolate_dirichlet!(
  object::AbstractPTFunction,
  free_values::PTArray,
  dirichlet_values::PTArray,
  fs::TrialPFESpace)

  for k in eachindex(object)
    cell_vals = FESpaces._cell_vals(fs,object[k])
    gather_dirichlet_values!(dirichlet_values[k],fs,cell_vals)
    fill!(free_values[k],zero(eltype(free_values[k])))
  end
  FEFunction(fs,free_values,dirichlet_values)
end

# MultiField interface
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
    @assert all([length(fe.dirichlet_values) == length(first(spaces).dirichlet_values) for fe in spaces])
    new{MS,CS,V}(V,spaces,multi_field_style,constraint_style)
  end
end

function MultiFieldPFESpace(spaces::Vector{<:SingleFieldFESpace})
  Ts = map(get_dof_value_type,spaces)
  T = typeof(*(map(zero,Ts)...))
  MultiFieldPFESpace(Vector{T},spaces,ConsecutiveMultiFieldStyle())
end

function MultiFieldPFESpace(::Type{V},spaces::Vector{<:SingleFieldFESpace}) where V
  MultiFieldPFESpace(V,spaces,ConsecutiveMultiFieldStyle())
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

FESpaces.get_dof_value_type(::MultiFieldPFESpace{MS,CS,V}) where {MS,CS,V} = eltype(V)

FESpaces.get_vector_type(f::MultiFieldPFESpace) = f.vector_type

FESpaces.ConstraintStyle(::Type{MultiFieldPFESpace{S,B,V}}) where {S,B,V} = B()

function FESpaces.zero_free_values(f::MultiFieldPFESpace)
  vcat([zero_free_values(fe) for fe in f.spaces]...)
end

function block_zero_free_values(f::MultiFieldPFESpace)
  [zero_free_values(fe) for fe in f.spaces]
end

function FESpaces.get_fe_basis(f::MultiFieldPFESpace)
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

function FESpaces.get_trial_fe_basis(f::MultiFieldPFESpace)
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

function split_fields(fe::Union{MultiFieldPFESpace,MultiFieldFESpace},free_values::PTArray)
  offsets = compute_field_offsets(fe)
  fields = map(1:length(fe.spaces)) do field
    pini = offsets[field] + 1
    pend = offsets[field] + num_free_dofs(fe.spaces[field])
    map(x->getindex(x,pini:pend),free_values)
  end
  fields
end

function MultiField.restrict_to_field(f::MultiFieldPFESpace,free_values::PTArray,field::Integer)
  MultiField._restrict_to_field(f,MultiFieldStyle(f),free_values,field)
end

function MultiField._restrict_to_field(
  f::MultiFieldPFESpace,
  ::ConsecutiveMultiFieldStyle,
  free_values::PTArray,
  field::Integer)

  offsets = compute_field_offsets(f)
  U = f.spaces
  pini = offsets[field] + 1
  pend = offsets[field] + num_free_dofs(U[field])
  map(fv -> SubVector(fv,pini,pend),free_values)
end

function MultiField.compute_field_offsets(f::MultiFieldPFESpace)
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

function FESpaces.interpolate!(objects,free_values::PTArray,fe::MultiFieldPFESpace)
  block_free_values = block_zero_free_values(fe)
  blocks = SingleFieldPTFEFunction[]
  for (free_values_i,U,object) in zip(block_free_values,fe.spaces,objects)
    uhi = interpolate!(object,free_values_i,U)
    push!(blocks,uhi)
  end
  PTMultiFieldFEFunction(free_values,fe,blocks)
end

function FESpaces.interpolate_everywhere(objects,fe::MultiFieldPFESpace)
  free_values = zero_free_values(fe)
  block_free_values = block_zero_free_values(fe)
  blocks = SingleFieldPTFEFunction[]
  for (free_values_i,U,object) in zip(block_free_values,fe.spaces,objects)
    dirichlet_values_i = zero_dirichlet_values(U)
    uhi = interpolate_everywhere!(object,free_values_i,dirichlet_values_i,U)
    push!(blocks,uhi)
  end
  PTMultiFieldFEFunction(free_values,fe,blocks)
end

function FESpaces.interpolate_dirichlet(objects,fe::MultiFieldPFESpace)
  free_values = zero_free_values(fe)
  block_free_values = block_zero_free_values(fe)
  blocks = SingleFieldPTFEFunction[]
  for (free_values_i,U,object) in zip(block_free_values,fe.spaces,objects)
    dirichlet_values_i = zero_dirichlet_values(U)
    uhi = interpolate_dirichlet!(object,free_values_i,dirichlet_values_i,U)
    push!(blocks,uhi)
  end
  PTMultiFieldFEFunction(free_values,fe,blocks)
end

function FESpaces.EvaluationFunction(fe::MultiFieldPFESpace,free_values::PTArray)
  blocks = map(eachindex(fe.spaces)) do i
    free_values_i = restrict_to_field(fe,free_values,i)
    fe_function_i = EvaluationFunction(fe.spaces[i],free_values_i)
    free_values_i,fe_function_i
  end
  free_values = vcat(first.(blocks)...)
  fe_functions = last.(blocks)
  PTMultiFieldFEFunction(free_values,fe,fe_functions)
end

function Arrays.testitem(f::MultiFieldPFESpace)
  MultiFieldFESpace(f.vector_type,map(testitem,f.spaces),f.multi_field_style)
end

function field_offsets(f::Union{MultiFieldFESpace,MultiFieldPFESpace})
  [compute_field_offsets(f)...,num_free_dofs(f)]
end
