abstract type PTCellField <: CellField end

function CellData.CellField(f::AbstractPTFunction,trian::Triangulation,::DomainStyle)
  s = size(get_cell_map(trian))
  cell_field = Fill(PTGenericField(f),s)
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

abstract type PTFEFunction <: PTCellField end

struct SingleFieldPTFEFunction{T<:CellField} <: PTFEFunction
  cell_field::T
  cell_dof_values::AbstractArray{<:PTArray{<:AbstractVector{<:Number}}}
  free_values::PTArray{<:AbstractVector{<:Number}}
  dirichlet_values::PTArray{<:AbstractVector{<:Number}}
  fe_space::SingleFieldFESpace
end

CellData.get_data(f::SingleFieldPTFEFunction) = get_data(f.cell_field)
FESpaces.get_triangulation(f::SingleFieldPTFEFunction) = get_triangulation(f.cell_field)
CellData.DomainStyle(::Type{SingleFieldPTFEFunction{T}}) where T = DomainStyle(T)

FESpaces.get_free_dof_values(f::SingleFieldPTFEFunction) = f.free_values
FESpaces.get_cell_dof_values(f::SingleFieldPTFEFunction) = f.cell_dof_values
FESpaces.get_fe_space(f::SingleFieldPTFEFunction) = f.fe_space

function FESpaces.FEFunction(
  fs::SingleFieldFESpace,free_values::PTArray,dirichlet_values::PTArray)
  cell_vals = scatter_free_and_dirichlet_values(fs,free_values,dirichlet_values)
  cell_field = CellField(fs,cell_vals)
  SingleFieldPTFEFunction(cell_field,cell_vals,free_values,dirichlet_values,fs)
end

function TransientFETools.TransientCellField(single_field::SingleFieldPTFEFunction,derivatives::Tuple)
  TransientSingleFieldCellField(single_field,derivatives)
end

struct MultiFieldPTFEFunction{T<:MultiFieldCellField} <: PTCellField
  single_fe_functions::Vector{<:SingleFieldPTFEFunction}
  free_values::AbstractArray
  fe_space::MultiFieldFESpace
  multi_cell_field::T

  function MultiFieldPTFEFunction(
    free_values::AbstractVector,
    space::MultiFieldFESpace,
    single_fe_functions::Vector{<:SingleFieldPTFEFunction})

    multi_cell_field = MultiFieldCellField(map(i->i.cell_field,single_fe_functions))
    T = typeof(multi_cell_field)

    new{T}(
      single_fe_functions,
      free_values,
      space,
      multi_cell_field)
  end
end

CellData.get_data(f::MultiFieldPTFEFunction) = get_data(f.multi_cell_field)
FESpaces.get_triangulation(f::MultiFieldPTFEFunction) = get_triangulation(f.multi_cell_field)
CellData.DomainStyle(::Type{MultiFieldPTFEFunction{T}}) where T = DomainStyle(T)
FESpaces.get_free_dof_values(f::MultiFieldPTFEFunction) = f.free_values
FESpaces.get_fe_space(f::MultiFieldPTFEFunction) = f.fe_space

function FESpaces.get_cell_dof_values(f::MultiFieldPTFEFunction)
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

function FESpaces.get_cell_dof_values(f::MultiFieldPTFEFunction,trian::Triangulation)
  uhs = f.single_fe_functions
  blockmask = [is_change_possible(get_triangulation(uh),trian) for uh in uhs]
  active_block_ids = findall(blockmask)
  active_block_data = Any[ get_cell_dof_values(uhs[i],trian) for i in active_block_ids ]
  nblocks = length(uhs)
  lazy_map(BlockMap(nblocks,active_block_ids),active_block_data...)
end

MultiField.num_fields(m::MultiFieldPTFEFunction) = length(m.single_fe_functions)
Base.iterate(m::MultiFieldPTFEFunction) = iterate(m.single_fe_functions)
Base.iterate(m::MultiFieldPTFEFunction,state) = iterate(m.single_fe_functions,state)
Base.getindex(m::MultiFieldPTFEFunction,field_id::Integer) = m.single_fe_functions[field_id]

function FESpaces.FEFunction(fe::MultiFieldPFESpace,free_values::PTArray)
  blocks = map(1:length(fe.spaces)) do i
    free_values_i = restrict_to_field(fe,free_values,i)
    FEFunction(fe.spaces[i],free_values_i)
  end
  MultiFieldPTFEFunction(free_values,fe,blocks)
end

function CellData.CellField(fe::MultiFieldPFESpace,cell_values)
  single_fields = map(1:length(fe.spaces)) do i
    cell_values_field = lazy_map(a->a.array[i],cell_values)
    CellField(fe.spaces[i],cell_values_field)
  end
  MultiFieldCellField(single_fields)
end

function TransientFETools.TransientCellField(multi_field::MultiFieldPTFEFunction,derivatives::Tuple)
  transient_single_fields = TransientFETools._to_transient_single_fields(multi_field,derivatives)
  TransientMultiFieldCellField(multi_field,derivatives,transient_single_fields)
end
