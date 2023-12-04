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
