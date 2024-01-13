abstract type PCellField <: CellField end

function CellData.CellField(f::AbstractPTFunction,trian::Triangulation,::DomainStyle)
  s = size(get_cell_map(trian))
  cell_field = Fill(PGenericField(f),s)
  GenericCellField(cell_field,trian,PhysicalDomain())
end

function CellData.CellField(
  f::AbstractPTFunction{<:AbstractVector{<:Number},<:Union{Real,Nothing}},
  trian::Triangulation,
  ::DomainStyle)

  s = size(get_cell_map(trian))
  cell_field = Fill(get_fields(f),s)
  GenericCellField(cell_field,trian,PhysicalDomain())
end

struct SingleFieldPFEFunction{T<:CellField} <: PCellField
  cell_field::T
  cell_dof_values::AbstractArray{<:PArray{<:AbstractVector{<:Number}}}
  free_values::PArray{<:AbstractVector{<:Number}}
  dirichlet_values::PArray{<:AbstractVector{<:Number}}
  fe_space::SingleFieldFESpace
end

CellData.get_data(f::SingleFieldPFEFunction) = get_data(f.cell_field)
FESpaces.get_triangulation(f::SingleFieldPFEFunction) = get_triangulation(f.cell_field)
CellData.DomainStyle(::Type{SingleFieldPFEFunction{T}}) where T = DomainStyle(T)

FESpaces.get_free_dof_values(f::SingleFieldPFEFunction) = f.free_values
FESpaces.get_cell_dof_values(f::SingleFieldPFEFunction) = f.cell_dof_values
FESpaces.get_fe_space(f::SingleFieldPFEFunction) = f.fe_space

function FESpaces.FEFunction(
  fs::SingleFieldFESpace,free_values::PArray,dirichlet_values::PArray)
  cell_vals = scatter_free_and_dirichlet_values(fs,free_values,dirichlet_values)
  cell_field = CellField(fs,cell_vals)
  SingleFieldPFEFunction(cell_field,cell_vals,free_values,dirichlet_values,fs)
end

function Base.iterate(f::SingleFieldPFEFunction)
  state = 1

  data = getindex.(get_data(f),state)
  cell_field = GenericCellField(data,get_triangulation(f),DomainStyle(f))
  cell_dof_values = getindex.(f.cell_dof_values,state)
  free_values = f.free_values[state]
  dirichlet_values = f.dirichlet_values[state]
  fe_space = TrialFESpace(dirichlet_values,f.fe_space.space)
  sf = SingleFieldFEFunction(cell_field,cell_dof_values,free_values,dirichlet_values,fe_space)

  (sf,state),state
end

function Base.iterate(f::SingleFieldPFEFunction,state)
  if state >= length(f.free_values)
    return nothing
  end
  state += 1

  data = getindex.(get_data(f),state)
  cell_field = GenericCellField(data,get_triangulation(f),DomainStyle(f))
  cell_dof_values = getindex.(f.cell_dof_values,state)
  free_values = f.free_values[state]
  dirichlet_values = f.dirichlet_values[state]
  fe_space = TrialFESpace(dirichlet_values,f.fe_space.space)
  sf = SingleFieldFEFunction(cell_field,cell_dof_values,free_values,dirichlet_values,fe_space)

  (sf,state),state
end

function TransientFETools.TransientCellField(single_field::SingleFieldPFEFunction,derivatives::Tuple)
  TransientSingleFieldCellField(single_field,derivatives)
end

struct MultiFieldPFEFunction{T<:MultiFieldCellField} <: PCellField
  single_fe_functions::Vector{<:SingleFieldPFEFunction}
  free_values::AbstractArray
  fe_space::MultiFieldPFESpace
  multi_cell_field::T

  function MultiFieldPFEFunction(
    free_values::AbstractVector,
    space::MultiFieldFESpace,
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
Base.iterate(m::MultiFieldPFEFunction) = iterate(m.single_fe_functions)
Base.iterate(m::MultiFieldPFEFunction,state) = iterate(m.single_fe_functions,state)
Base.getindex(m::MultiFieldPFEFunction,field_id::Integer) = m.single_fe_functions[field_id]

function FESpaces.FEFunction(fe::MultiFieldPFESpace,free_values::PArray)
  blocks = map(1:length(fe.spaces)) do i
    free_values_i = restrict_to_field(fe,free_values,i)
    FEFunction(fe.spaces[i],free_values_i)
  end
  MultiFieldPFEFunction(free_values,fe,blocks)
end

function CellData.CellField(fe::MultiFieldPFESpace,cell_values)
  single_fields = map(1:length(fe.spaces)) do i
    cell_values_field = lazy_map(a->a.array[i],cell_values)
    CellField(fe.spaces[i],cell_values_field)
  end
  MultiFieldCellField(single_fields)
end

function TransientFETools.TransientCellField(multi_field::MultiFieldPFEFunction,derivatives::Tuple)
  transient_single_fields = TransientFETools._to_transient_single_fields(multi_field,derivatives)
  TransientMultiFieldCellField(multi_field,derivatives,transient_single_fields)
end
