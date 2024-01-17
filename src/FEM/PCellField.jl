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

function FESpaces.test_fe_function(f::SingleFieldPFEFunction)
  trian = get_triangulation(f)
  free_values = get_free_dof_values(f)
  fe_space = get_fe_space(f)
  cell_values = get_cell_dof_values(f,trian)
  dirichlet_values = f.dirichlet_values
  map(1:length(f)) do i
    fe_space_i = _get_at_index(fe_space,i)
    free_values_i = free_values[i]
    fi = FEFunction(fe_space_i,free_values_i)
    test_fe_function(fi)
    @test free_values[i] == get_free_dof_values(fi)
    @test cell_values[i] == get_cell_dof_values(fi,trian)
    @test dirichlet_values[i] == fi.dirichlet_values
  end
end

function TransientFETools.TransientCellField(single_field::SingleFieldPFEFunction,derivatives::Tuple)
  TransientFETools.TransientSingleFieldCellField(single_field,derivatives)
end

# for visualization purposes

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

function _to_vector_cellfields(f)
  map(1:length(f)) do i
    _get_at_index(f,i)
  end
end
