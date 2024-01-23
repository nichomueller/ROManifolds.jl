abstract type FEPFunction <: FEFunction end

struct SingleFieldFEPFunction{T<:CellField} <: FEPFunction
  cell_field::T
  cell_dof_values::AbstractArray{<:PArray{<:AbstractVector{<:Number}}}
  free_values::PArray{<:AbstractVector{<:Number}}
  dirichlet_values::PArray{<:AbstractVector{<:Number}}
  fe_space::SingleFieldFESpace
end

Base.length(f::SingleFieldFEPFunction) = length(f.free_values)
CellData.get_data(f::SingleFieldFEPFunction) = get_data(f.cell_field)
FESpaces.get_triangulation(f::SingleFieldFEPFunction) = get_triangulation(f.cell_field)
CellData.DomainStyle(::Type{SingleFieldFEPFunction{T}}) where T = DomainStyle(T)
FESpaces.get_free_dof_values(f::SingleFieldFEPFunction) = f.free_values
FESpaces.get_cell_dof_values(f::SingleFieldFEPFunction) = f.cell_dof_values
FESpaces.get_fe_space(f::SingleFieldFEPFunction) = f.fe_space

function FESpaces.FEFunction(
  fs::SingleFieldFESpace,free_values::PArray,dirichlet_values::PArray)
  cell_vals = scatter_free_and_dirichlet_values(fs,free_values,dirichlet_values)
  cell_field = CellField(fs,cell_vals)
  SingleFieldFEPFunction(cell_field,cell_vals,free_values,dirichlet_values,fs)
end

function TransientFETools.TransientCellField(single_field::SingleFieldFEPFunction,derivatives::Tuple)
  TransientFETools.TransientSingleFieldCellField(single_field,derivatives)
end

# for visualization/testing purposes

function FESpaces.test_fe_function(f::SingleFieldFEPFunction)
  lazy_getter(a,i=1) = lazy_map(x->getindex(x.array,i),a)
  trian = get_triangulation(f)
  free_values = get_free_dof_values(f)
  fe_space = get_fe_space(f)
  cell_values = get_cell_dof_values(f,trian)
  dirichlet_values = f.dirichlet_values
  for i in 1:length_dirichlet_values(fe_space)
    fe_space_i = _getindex(fe_space,i)
    fi = FEFunction(fe_space_i,free_values[i])
    test_fe_function(fi)
    @test free_values[i] == get_free_dof_values(fi)
    @test lazy_getter(cell_values,i) == get_cell_dof_values(fi,trian)
    @test dirichlet_values[i] == fi.dirichlet_values
  end
end

function _getindex(f::GenericCellField,index)
  data = get_data(f)
  trian = get_triangulation(f)
  DS = DomainStyle(f)
  di = getindex.(data,index)
  GenericCellField(di,trian,DS)
end

function _length(f::SingleFieldFEPFunction)
  @assert length_dirichlet_values(f.fe_space) == length(f.dirichlet_values)
  length(f.dirichlet_values)
end

function _getindex(f::SingleFieldFEPFunction,index)
  cf = _getindex(f.cell_field,index)
  fs = _getindex(f.fe_space,index)
  cv = f.cell_dof_values[index]
  fv = f.free_values[index]
  dv = f.dirichlet_values[index]
  SingleFieldFEFunction(cf,cv,fv,dv,fs)
end

struct MultiFieldFEPFunction{T<:MultiFieldCellField} <: FEPFunction
  single_fe_functions::Vector{<:SingleFieldFEPFunction}
  free_values::AbstractArray
  fe_space::MultiFieldPFESpace
  multi_cell_field::T

  function MultiFieldFEPFunction(
    free_values::AbstractVector,
    space::MultiFieldPFESpace,
    single_fe_functions::Vector{<:SingleFieldFEPFunction})

    multi_cell_field = MultiFieldCellField(map(i->i.cell_field,single_fe_functions))
    T = typeof(multi_cell_field)

    new{T}(
      single_fe_functions,
      free_values,
      space,
      multi_cell_field)
  end
end

Base.length(f::MultiFieldFEPFunction) = length(f.free_values)
CellData.get_data(f::MultiFieldFEPFunction) = get_data(f.multi_cell_field)
FESpaces.get_triangulation(f::MultiFieldFEPFunction) = get_triangulation(f.multi_cell_field)
CellData.DomainStyle(::Type{MultiFieldFEPFunction{T}}) where T = DomainStyle(T)
FESpaces.get_free_dof_values(f::MultiFieldFEPFunction) = f.free_values
FESpaces.get_fe_space(f::MultiFieldFEPFunction) = f.fe_space

function FESpaces.get_cell_dof_values(f::MultiFieldFEPFunction)
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

function FESpaces.get_cell_dof_values(f::MultiFieldFEPFunction,trian::Triangulation)
  uhs = f.single_fe_functions
  blockmask = [is_change_possible(get_triangulation(uh),trian) for uh in uhs]
  active_block_ids = findall(blockmask)
  active_block_data = Any[get_cell_dof_values(uhs[i],trian) for i in active_block_ids]
  nblocks = length(uhs)
  lazy_map(BlockMap(nblocks,active_block_ids),active_block_data...)
end

MultiField.num_fields(m::MultiFieldFEPFunction) = length(m.single_fe_functions)
Base.iterate(m::MultiFieldFEPFunction) = iterate(m.single_fe_functions)
Base.iterate(m::MultiFieldFEPFunction,state) = iterate(m.single_fe_functions,state)
Base.getindex(m::MultiFieldFEPFunction,field_id::Integer) = m.single_fe_functions[field_id]

function TransientFETools.TransientCellField(multi_field::MultiFieldFEPFunction,derivatives::Tuple)
  transient_single_fields = TransientFETools._to_transient_single_fields(multi_field,derivatives)
  TransientFETools.TransientMultiFieldCellField(multi_field,derivatives,transient_single_fields)
end

# for visualization/testing purposes

function FESpaces.test_fe_function(f::MultiFieldFEPFunction)
  map(test_fe_function,f.single_fe_functions)
end

_length(f::MultiFieldFEPFunction) = _length(first(f.single_fe_functions))

function _getindex(f::MultiFieldFEPFunction,index)
  mfs = map(_getindex,f.single_fe_functions)
  sff = map(_getindex,f.fe_space)
  fv = f.free_values[index]
  MultiFieldFEPFunction(fv,mfs,sff)
end
