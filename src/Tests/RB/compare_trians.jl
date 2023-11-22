abstract type MyGrid{Dc,Dp} <: Grid{Dc,Dp} end

struct MyGridView{Dc,Dp,A,B} <: MyGrid{Dc,Dp}
  parent::A
  cell_to_parent_cell::B
  function MyGridView(parent::Grid,cell_to_parent_cell::AbstractArray)
    Dc = num_cell_dims(parent)
    Dp = num_point_dims(parent)
    A = typeof(parent)
    B = typeof(cell_to_parent_cell)
    new{Dc,Dp,A,B}(parent,cell_to_parent_cell)
  end
end

struct ValsAtMask{T,N,M} <: AbstractArray{T,N}
  values::AbstractArray{T,N}
  mask::M
end

Base.size(k::ValsAtMask) = size(k.values)
Base.length(k::ValsAtMask) = length(k.mask)
Base.getindex(k::ValsAtMask,i...) = k.values[i...]
Arrays.testitem(k::ValsAtMask) = first(k.values)
Arrays.return_cache(k::ValsAtMask,i...) = array_cache(k.values)
Arrays.evaluate!(cache,k::ValsAtMask,i...) = getindex!(cache,k.values,i...)
Arrays.evaluate(k::ValsAtMask,i...) = k[i...]

# function Arrays.CompressedArray()

# end

function Arrays.lazy_map(f,k::ValsAtMask,args::AbstractArray...)
  ValsAtMask(lazy_map(f,k.values,args...),k.mask)
end

myview(a::Grid,b::AbstractArray) = MyGridView(a,b)

function myview(t::BodyFittedTriangulation,ids::AbstractArray)
  model = t.model
  grid = myview(t.grid,ids)
  tface_to_mface = ValsAtMask(t.tface_to_mface,ids)
  BodyFittedTriangulation(model,grid,tface_to_mface)
end

function MyGridView(parent::Grid,parent_cell_to_mask::AbstractArray{Bool})
  cell_to_parent_cell = findall(collect1d(parent_cell_to_mask))
  MyGridView(parent,cell_to_parent_cell)
end

function MyGridView(parent::Grid,parent_cell_to_mask::AbstractVector{Bool})
  cell_to_parent_cell = findall(parent_cell_to_mask)
  MyGridView(parent,cell_to_parent_cell)
end

function Geometry.OrientationStyle(::Type{MyGridView{Dc,Dp,G}}) where {Dc,Dp,G}
  OrientationStyle(G)
end

function Geometry.RegularityStyle(::Type{MyGridView{Dc,Dp,G}}) where {Dc,Dp,G}
  RegularityStyle(G)
end

function Geometry.get_node_coordinates(grid::MyGridView)
  get_node_coordinates(grid.parent)
end

function Geometry.get_cell_node_ids(grid::MyGridView)
  ValsAtMask(get_cell_node_ids(grid.parent),grid.cell_to_parent_cell)
end

function Geometry.get_reffes(grid::MyGridView)
  get_reffes(grid.parent)
end

function Geometry.get_cell_type(grid::MyGridView)
  ValsAtMask(get_cell_type(grid.parent),grid.cell_to_parent_cell)
end

function Geometry.get_cell_map(grid::MyGridView)
  ValsAtMask(get_cell_map(grid.parent),grid.cell_to_parent_cell)
end

function Geometry.get_facet_normal(grid::MyGridView)
  ValsAtMask(get_facet_normal(grid.parent),grid.cell_to_parent_cell)
end

function CellData.change_domain(
  a::CellField,
  strian::BodyFittedTriangulation{Dt,Dp,A,<:Grid,C},::ReferenceDomain,
  ttrian::BodyFittedTriangulation{Dt,Dp,A,<:MyGrid,<:ValsAtMask},::ReferenceDomain
  ) where {Dt,Dp,A,C}

  @assert strian === ttrian || is_parent(strian,ttrian)
  return a
end

function CellData.change_domain(
  a::CellField,
  strian::BodyFittedTriangulation{Dt,Dp,A,<:Grid,C},::PhysicalDomain,
  ttrian::BodyFittedTriangulation{Dt,Dp,A,<:MyGrid,<:ValsAtMask},::PhysicalDomain
  ) where {Dt,Dp,A,C}

  @assert strian === ttrian || is_parent(strian,ttrian)
  return a
end

function CellData.change_domain(
  a::CellField,
  strian::BoundaryTriangulation,::ReferenceDomain,
  ttrian::TriangulationView,::ReferenceDomain
  )

  @assert strian === ttrian || is_parent(strian,ttrian)
  return a
end

function CellData.change_domain(
  a::CellField,
  strian::BoundaryTriangulation,::PhysicalDomain,
  ttrian::TriangulationView,::PhysicalDomain)

  @assert strian === ttrian || is_parent(strian,ttrian)
  return a
end

function CellData.is_change_possible(
  strian::BodyFittedTriangulation{Dt,Dp,A,<:Grid,C},
  ttrian::BodyFittedTriangulation{Dt,Dp,A,<:MyGrid,<:ValsAtMask}
  ) where {Dt,Dp,A,C}
  @assert strian === ttrian || is_parent(strian,ttrian)
  return true
end

function CellData.is_change_possible(strian::BoundaryTriangulation,ttrian::TriangulationView)
  @assert strian === ttrian || is_parent(strian,ttrian)
  return true
end

dv = get_fe_basis(test)
du = get_trial_fe_basis(trial(nothing,nothing))
idx = [1,10,20]
Ξ = myview(Ω,idx)
dΞ = Measure(Ξ,2)

μ = rand(3)
t = dt
ff = a(μ,t)*∇(dv)⋅∇(du)
trian_f = get_triangulation(ff)

quad = dΩ.quad
@time b = change_domain(ff,quad.trian,quad.data_domain_style)
@time cell_map = get_cell_map(quad.trian)
@time cell_Jt = lazy_map(∇,cell_map)
@time cell_Jtx = lazy_map(evaluate,cell_Jt,quad.cell_point)
x = get_cell_points(quad)
@time bx = b(x)
@time integral = lazy_map(IntegrationMap(),bx,quad.cell_weight,cell_Jtx)

_quad = dΞ.quad
@time _b = change_domain(ff,_quad.trian,_quad.data_domain_style)
@time _cell_map = get_cell_map(_quad.trian)
@time _cell_Jt = lazy_map(∇,_cell_map)
@time _cell_Jtx = lazy_map(evaluate,_cell_Jt,_quad.cell_point)
_x = get_cell_points(_quad)
@time _bx = _b(_x)
@time _integral = lazy_map(IntegrationMap(),_bx,_quad.cell_weight,_cell_Jtx)

@time dc = ∫(a(μ,t)*∇(dv)⋅∇(du))dΩ
@time _dchat = ∫(a(μ,t)*∇(dv)⋅∇(du))dΞ
