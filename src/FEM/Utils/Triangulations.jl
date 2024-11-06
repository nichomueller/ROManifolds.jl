"""
    is_parent(tparent::Triangulation,tchild::Triangulation) -> Bool

Returns true if `tchild` is a triangulation view of `tparent`, false otherwise

"""
function is_parent(tparent::Triangulation,tchild::Triangulation)
  false
end

function is_parent(
  tparent::BodyFittedTriangulation,
  tchild::BodyFittedTriangulation{Dt,Dp,A,<:Geometry.GridView}) where {Dt,Dp,A}
  tparent.grid === tchild.grid.parent
end

function is_parent(tparent::BoundaryTriangulation,tchild::Geometry.TriangulationView)
  tparent === tchild.parent
end

function is_parent(tparent::Interfaces.SubFacetTriangulation,tchild::Geometry.TriangulationView)
  tparent === tchild.parent
end

function get_parent(t::Geometry.Grid)
  @abstractmethod
end

function get_parent(gv::Geometry.GridView)
  gv.parent
end

function get_parent(t::Geometry.TriangulationView)
  t.parent
end

function get_parent(t::BodyFittedTriangulation)
  grid = get_parent(get_grid(t))
  model = get_background_model(t)
  tface_to_mface = IdentityVector(num_cells(grid))
  BodyFittedTriangulation(model,grid,tface_to_mface)
end

function get_parent(t::AbstractVector{<:Triangulation})
  get_parent(first(t))
end

"""
    is_parent(t::Triangulation) -> Triangulation

When `t` is a triangulation view, returns its parent; throws an error when `t`
is not a triangulation view

"""
function get_parent(t::Triangulation)
  @abstractmethod
end

function Base.isapprox(t::Grid,s::Grid)
  false
end

function Base.isapprox(t::T,s::T) where {T<:UnstructuredGrid}
  (
    t.cell_node_ids == s.cell_node_ids &&
    t.cell_types == s.cell_types &&
    t.node_coordinates == s.node_coordinates
  )
end

function Base.isapprox(t::T,s::T) where {T<:Geometry.GridView}
  t.parent ≈ s.parent
end

get_tface_to_mface(t::Triangulation) = @abstractmethod
get_tface_to_mface(t::BodyFittedTriangulation) = t.tface_to_mface
get_tface_to_mface(t::BoundaryTriangulation) = get_tface_to_mface(t.trian)

function Base.isapprox(t::T,s::S) where {T<:Triangulation,S<:Triangulation}
  false
end

function Base.isapprox(t::T,s::T) where {T<:Triangulation}
  get_tface_to_mface(t) == get_tface_to_mface(s) && get_grid(t) ≈ get_grid(s)
end

function Base.isapprox(t::T,s::T) where {T<:Geometry.TriangulationView}
  get_view_indices(t) == get_view_indices(s) && get_parent(t) ≈ get_parent(s)
end

function Base.isapprox(t::T,s::T) where {T<:BoundaryTriangulation}
  (
    t.trian.tface_to_mface == s.trian.tface_to_mface &&
    t.trian.grid.parent.cell_node_ids == s.trian.grid.parent.cell_node_ids &&
    t.trian.grid.parent.cell_types == s.trian.grid.parent.cell_types &&
    t.trian.grid.parent.node_coordinates == s.trian.grid.parent.node_coordinates
  )
end

function isapprox_parent(tparent::Triangulation,tchild::Triangulation)
  tparent ≈ get_parent(tchild)
end

function get_view_indices(t::BodyFittedTriangulation)
  grid = get_grid(t)
  grid.cell_to_parent_cell
end

function get_view_indices(t::Geometry.TriangulationView)
  t.cell_to_parent_cell
end

function get_union_indices(trians)
  indices = map(get_view_indices,trians)
  union(indices...) |> unique
end

"""
    merge_triangulations(trians::Tuple{Vararg{Triangulation}}) -> Triangulation

Given a tuple of triangulation views `trians`, returns the triangulation view
on the union of the viewed cells. In other words, the minimum common integration
domain is found

"""
function merge_triangulations(trians)
  parent = get_parent(trians)
  uindices = get_union_indices(trians)
  view(parent,uindices)
end

function find_trian_permutation(a,b,cmp::Function)
  map(a -> findfirst(b -> cmp(a,b),b),a)
end

function find_trian_permutation(a,b)
  cmp = (a,b) -> a == b || is_parent(a,b)
  map(a -> findfirst(b -> cmp(a,b),b),a)
end

"""
    order_triangulations(tparents::Tuple{Vararg{Triangulation}},
      tchildren::Tuple{Vararg{Triangulation}}) -> Tuple{Vararg{Triangulation}}

Orders the triangulation children in the same way as the triangulation parents

"""
function order_triangulations(tparents,tchildren)
  @check length(tparents) == length(tchildren)
  iperm = find_trian_permutation(tparents,tchildren)
  map(iperm->tchildren[iperm],iperm)
end

"""
    find_closest_view(tparents::Tuple{Vararg{Triangulation}},
      tchild::Triangulation) -> Integer, Triangulation

Finds the approximate parent of `tchild`; it returns the parent's index and its
view in the same indices as `tchild` (which should be a triangulation view)

"""
function find_closest_view(tparents,tchild::Triangulation)
  cmp(a,b) = isapprox_parent(b,a)
  iperm::Tuple{Vararg{Int}} = find_trian_permutation((tchild,),tparents,cmp)
  @check length(iperm) == 1
  indices = get_view_indices(tchild)
  return iperm,view(tparents[iperm...],indices)
end

# triangulation views

function FESpaces.get_cell_fe_data(fun,f,ttrian::Geometry.TriangulationView)
  parent_vals = FESpaces.get_cell_fe_data(fun,f,ttrian.parent)
  return lazy_map(Reindex(parent_vals),ttrian.cell_to_parent_cell)
end

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
  cell_data = lazy_map(Reindex(CellData.get_data(parent)),ttrian.cell_to_parent_cell)
  return CellData.similar_cell_field(a,cell_data,ttrian,ReferenceDomain())
end

function CellData.change_domain(a::CellField,strian::Triangulation,::PhysicalDomain,ttrian::Geometry.TriangulationView,::PhysicalDomain)
  if strian === ttrian
    return a
  end
  parent = CellData.change_domain(a,strian,PhysicalDomain(),ttrian.parent,PhysicalDomain())
  cell_data = lazy_map(Reindex(CellData.get_data(parent)),ttrian.cell_to_parent_cell)
  return CellData.similar_cell_field(a,cell_data,ttrian,PhysicalDomain())
end
