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
  for fe_space_i in fe_space
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

# for visualization/testing purposes

function Base.iterate(f::GenericCellField)
  data = get_data(f)
  trian = get_triangulation(f)
  DS = DomainStyle(f)
  index = 1
  final_index = length(first.(data))
  di = getindex.(data,index)
  state = (index,final_index,data,trian,DS)
  GenericCellField(di,trian,DS),state
end

function Base.iterate(f::GenericCellField,state)
  index,final_index,data,trian,DS = state
  index += 1
  if index > final_index
    return nothing
  end
  di = getindex.(data,index)
  state = (index,final_index,data,trian,DS)
  GenericCellField(di,trian,DS),state
end

function Base.iterate(f::SingleFieldPFEFunction,state...)
  citer = iterate(f.cell_field,state...)
  fiter = iterate(f.fe_space,state...)
  if isnothing(citer) && isnothing(fiter)
    return nothing
  end
  cf,cstate = citer
  fs,fstate = fiter
  index, = fstate
  cv = f.cell_dof_values[index]
  fv = f.free_values[index]
  dv = f.dirichlet_values[index]
  SingleFieldFEFunction(cf,cv,fv,dv,fs),(cstate,fstate)
end
