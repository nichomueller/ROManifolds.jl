abstract type PCellField <: CellField end

function CellData.CellField(f::AbstractPFunction,trian::Triangulation,::DomainStyle)
  s = size(get_cell_map(trian))
  cell_field = Fill(PGenericField(f),s)
  GenericCellField(cell_field,trian,PhysicalDomain())
end

struct SingleFieldPFEFunction{T<:CellField} <: PCellField
  cell_field::T
  cell_dof_values::AbstractArray{<:PArray{<:AbstractVector{<:Number}}}
  free_values::PArray{<:AbstractVector{<:Number}}
  dirichlet_values::PArray{<:AbstractVector{<:Number}}
  fe_space::SingleFieldFESpace
end

Base.length(f::SingleFieldPFEFunction) = length(f.free_values)
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

# for visualization purposes
function _get_at_index(f::TrialPFESpace,i::Integer)
  dv = f.dirichlet_values[i]
  TrialFESpace(dv,f.space)
end

function _get_at_index(f::GenericCellField,i::Integer)
  data = get_data(f)
  trian = get_triangulation(f)
  DS = DomainStyle(f)
  di = getindex.(data,i)
  GenericCellField(di,trian,DS)
end

function _get_at_index(f::SingleFieldPFEFunction,i::Integer)
  cf = _get_at_index(f.cell_field,i)
  fs = _get_at_index(f.fe_space,i)
  cv = f.cell_dof_values[i]
  fv = f.free_values[i]
  dv = f.dirichlet_values[i]
  SingleFieldFEFunction(cf,cv,fv,dv,fs)
end

function _get_at_index(f::MultiFieldPFEFunction,i::Integer)
  fv = f.free_values[i]
  mfs,sff = map(f.single_fe_functions,f.fe_space) do ff,fs
    _get_at_index(fs,i),_get_at_index(ff,i)
  end |> tuple_of_arrays
  MultiFieldPFEFunction(fv,mfs,sff)
end

function _to_vector_cellfields(f)
  map(1:length(f)) do i
    _get_at_index(f,i)
  end
end
