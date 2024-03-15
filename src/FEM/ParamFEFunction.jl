abstract type ParamFEFunction <: FEFunction end

struct SingleFieldParamFEFunction{T<:CellField} <: ParamFEFunction
  cell_field::T
  cell_dof_values::AbstractArray{<:AbstractVector{<:Number}}
  free_values::AbstractVector{<:Number}
  dirichlet_values::AbstractVector{<:Number}
  fe_space::SingleFieldFESpace
end

Base.length(f::SingleFieldParamFEFunction) = length(f.free_values)
CellData.get_data(f::SingleFieldParamFEFunction) = get_data(f.cell_field)
FESpaces.get_triangulation(f::SingleFieldParamFEFunction) = get_triangulation(f.cell_field)
CellData.DomainStyle(::Type{SingleFieldParamFEFunction{T}}) where T = DomainStyle(T)
FESpaces.get_free_dof_values(f::SingleFieldParamFEFunction) = f.free_values
FESpaces.get_cell_dof_values(f::SingleFieldParamFEFunction) = f.cell_dof_values
FESpaces.get_fe_space(f::SingleFieldParamFEFunction) = f.fe_space

function TransientFETools.TransientCellField(f::SingleFieldParamFEFunction,derivatives::Tuple)
  TransientFETools.TransientSingleFieldCellField(f,derivatives)
end

# for visualization/testing purposes

function FESpaces.test_fe_function(f::SingleFieldParamFEFunction)
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

function _length(f::SingleFieldParamFEFunction)
  @assert length_dirichlet_values(f.fe_space) == length(f.dirichlet_values)
  length(f.dirichlet_values)
end

function _getindex(f::SingleFieldParamFEFunction,index)
  cf = _getindex(f.cell_field,index)
  fs = _getindex(f.fe_space,index)
  cv = lazy_map(x->getindex(x,index),f.cell_dof_values)
  fv = f.free_values[index]
  dv = f.dirichlet_values[index]
  SingleFieldFEFunction(cf,cv,fv,dv,fs)
end

struct MultiFieldParamFEFunction{T<:MultiFieldCellField} <: ParamFEFunction
  single_fe_functions::Vector{<:SingleFieldParamFEFunction}
  free_values::AbstractArray
  fe_space::MultiFieldParamFESpace
  multi_cell_field::T

  function MultiFieldParamFEFunction(
    free_values::AbstractVector,
    space::MultiFieldParamFESpace,
    single_fe_functions::Vector{<:SingleFieldParamFEFunction})

    multi_cell_field = MultiFieldCellField(map(i->i.cell_field,single_fe_functions))
    T = typeof(multi_cell_field)

    new{T}(
      single_fe_functions,
      free_values,
      space,
      multi_cell_field)
  end
end

Base.length(f::MultiFieldParamFEFunction) = length(f.free_values)
CellData.get_data(f::MultiFieldParamFEFunction) = get_data(f.multi_cell_field)
FESpaces.get_triangulation(f::MultiFieldParamFEFunction) = get_triangulation(f.multi_cell_field)
CellData.DomainStyle(::Type{MultiFieldParamFEFunction{T}}) where T = DomainStyle(T)
FESpaces.get_free_dof_values(f::MultiFieldParamFEFunction) = f.free_values
FESpaces.get_fe_space(f::MultiFieldParamFEFunction) = f.fe_space

function FESpaces.get_cell_dof_values(f::MultiFieldParamFEFunction)
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

function FESpaces.get_cell_dof_values(f::MultiFieldParamFEFunction,trian::Triangulation)
  uhs = f.single_fe_functions
  blockmask = [is_change_possible(get_triangulation(uh),trian) for uh in uhs]
  active_block_ids = findall(blockmask)
  active_block_data = Any[get_cell_dof_values(uhs[i],trian) for i in active_block_ids]
  nblocks = length(uhs)
  lazy_map(BlockMap(nblocks,active_block_ids),active_block_data...)
end

MultiField.num_fields(m::MultiFieldParamFEFunction) = length(m.single_fe_functions)
Base.iterate(m::MultiFieldParamFEFunction) = iterate(m.single_fe_functions)
Base.iterate(m::MultiFieldParamFEFunction,state) = iterate(m.single_fe_functions,state)
Base.getindex(m::MultiFieldParamFEFunction,field_id::Integer) = m.single_fe_functions[field_id]

function TransientFETools.TransientCellField(multi_field::MultiFieldParamFEFunction,derivatives::Tuple)
  transient_single_fields = TransientFETools._to_transient_single_fields(multi_field,derivatives)
  TransientFETools.TransientMultiFieldCellField(multi_field,derivatives,transient_single_fields)
end

# for visualization/testing purposes

function FESpaces.test_fe_function(f::MultiFieldParamFEFunction)
  map(test_fe_function,f.single_fe_functions)
end

_length(f::MultiFieldParamFEFunction) = _length(first(f.single_fe_functions))

function _getindex(f::MultiFieldParamFEFunction,index)
  mfs = map(_getindex,f.single_fe_functions)
  sff = map(_getindex,f.fe_space)
  fv = f.free_values[index]
  MultiFieldParamFEFunction(fv,mfs,sff)
end
