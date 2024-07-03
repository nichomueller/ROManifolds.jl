function FESpaces.get_cell_fe_data(fun,f,ttrian::Geometry.TriangulationView)
  parent_vals = FESpaces.get_cell_fe_data(fun,f,ttrian.parent)
  return view(parent_vals,ttrian.cell_to_parent_cell)
end

struct TableView{T,Vd<:AbstractVector{T},Vp<:AbstractVector,Vi<:AbstractVector} <: AbstractVector{Vector{T}}
  table::Table{T,Vd,Vp}
  ids::Vi
end

Base.size(a::TableView) = (length(a.ids),)
Base.IndexStyle(::Type{<:TableView}) = IndexLinear()
Arrays.array_cache(a::TableView) = Arrays.array_cache(a.table)
Arrays.getindex!(c,a::TableView,i::Integer) = Arrays.getindex!(c,a.table,a.ids[i])

function Base.getindex(a::TableView,i::Integer)
  cache = array_cache(a)
  getindex!(cache,a,i)
end

Base.view(a::Arrays.Table,ids::AbstractArray) = TableView(a,ids)

@inline function Geometry.is_change_possible(strian::Geometry.TriangulationView,ttrian::Triangulation)
  return false
end

@inline function Geometry.is_change_possible(strian::Triangulation,ttrian::Geometry.TriangulationView)
  return Geometry.is_change_possible(strian,ttrian.parent)
end

function CellData.change_domain(a::CellField,strian::Triangulation,::ReferenceDomain,ttrian::Geometry.TriangulationView,::ReferenceDomain)
  if strian === ttrian
    return a
  end
  parent = CellData.change_domain(a,strian,ReferenceDomain(),ttrian.parent,ReferenceDomain())
  cell_data = view(CellData.get_data(parent),ttrian.cell_to_parent_cell)
  return CellData.similar_cell_field(a,cell_data,ttrian,ReferenceDomain())
end

function CellData.change_domain(a::CellField,strian::Triangulation,::PhysicalDomain,ttrian::Geometry.TriangulationView,::PhysicalDomain)
  if strian === ttrian
    return a
  end
  parent = CellData.change_domain(a,strian,PhysicalDomain(),ttrian.parent,PhysicalDomain())
  cell_data = view(CellData.get_data(parent),ttrian.cell_to_parent_cell)
  return CellData.similar_cell_field(a,cell_data,ttrian,PhysicalDomain())
end
