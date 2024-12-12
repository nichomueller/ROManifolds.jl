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

function is_parent(tparent::Triangulation,tchild::Geometry.TriangulationView)
  tparent === tchild.parent
end

for T in (:Triangulation,:(BodyFittedTriangulation{Dt,Dp,A,<:Geometry.GridView} where {Dt,Dp,A}),:(Geometry.TriangulationView))
  @eval begin
    function is_parent(tparent::Geometry.AppendedTriangulation,tchild::$T)
      is_parent(tparent.a,tchild) || is_parent(tparent.b,tchild)
    end
  end
end

function is_parent(tparent::Geometry.AppendedTriangulation,tchild::Geometry.AppendedTriangulation)
  is_parent(tparent.a,tchild.a) && is_parent(tparent.b,tchild.b)
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
  tface_to_mface = get_parent(t.tface_to_mface)
  BodyFittedTriangulation(model,grid,tface_to_mface)
end

function get_parent(t::Geometry.AppendedTriangulation)
  a = get_parent(t.a)
  b = get_parent(t.b)
  lazy_append(a,b)
end

"""
    get_parent(t::Triangulation) -> Triangulation

When `t` is a triangulation view, returns its parent; throws an error when `t`
is not a triangulation view

"""
function get_parent(t::Triangulation)
  @abstractmethod
end

get_parent(i::LazyArray{<:Fill{<:Reindex}}) = i.maps.value.values

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

get_tface_to_mface(t::Geometry.BodyFittedTriangulation) = t.tface_to_mface
get_tface_to_mface(t::Geometry.BoundaryTriangulation) = t.glue.face_to_cell
get_tface_to_mface(t::Geometry.TriangulationView) = get_tface_to_mface(t.parent)
get_tface_to_mface(t::Interfaces.SubFacetTriangulation) = t.subfacets.facet_to_bgcell
get_tface_to_mface(t::Interfaces.SubCellTriangulation) = unique(t.subcells.cell_to_bgcell)
function get_tface_to_mface(t::Geometry.AppendedTriangulation)
  lazy_append(get_tface_to_mface(t.a),get_tface_to_mface(t.b))
end

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

function isincluded(tchild::Triangulation,tparent::Triangulation)
  isparent(tparent,tchild) && return true
  tface_to_mface_child = get_tface_to_mface(tchild)
  tface_to_mface_parent = get_tface_to_mface(tparent)
  child_to_parent = indexin(tface_to_mface_child,tface_to_mface_parent)
  isa(child_to_parent,Vector{Int})
end

function get_view_indices(t::BodyFittedTriangulation)
  grid = get_grid(t)
  grid.cell_to_parent_cell
end

function get_view_indices(t::Geometry.TriangulationView)
  t.cell_to_parent_cell
end

function get_view_indices(t::Geometry.AppendedTriangulation)
  a = get_view_indices(t.a)
  b = get_view_indices(t.b)
  lazy_append(a,b)
end

function get_union_indices(parent::Triangulation,trians)
  indices = map(get_view_indices,trians)
  union(indices...) |> unique
end

function get_union_indices(parent::AppendedTriangulation,trians)
  indices = ()
  for t in trians
    indst = get_view_indices(t)
    if !isa(t,AppendedTriangulation)
      _shift_inds!(indst,t,parent)
    end
    indices = (indices...,indst...)
  end
  union(indices...) |> unique
end

function _shift_inds!(i::AbstractVector,child::Triangulation,parent::AppendedTriangulation)
  isappended(t) = isa(t,AppendedTriangulation)
  @notimplementedif isappended(parent.a) || isappended(parent.b)
  cmp = (a,b) -> a == b || is_parent(a,b)
  j = findfirst(a -> cmp(a,child),(parent.a,parent.b))
  if j == 2
    ncells = num_cells(parent.a)
    i .+= ncells
  end
  i
end

"""
    merge_triangulations(trians::AbstractVector{<:Triangulation}) -> Triangulation

Given a tuple of triangulation views `trians`, returns the triangulation view
on the union of the viewed cells. In other words, the minimum common integration
domain is found

"""
function merge_triangulations(trians::AbstractVector{<:Triangulation})
  trian = first(trians)
  x = get_node_coordinates(trian)
  for t in trians[2:end]
    if get_node_coordinates(t) !== x
      trian = lazy_append(trian,t)
    end
  end
  parent = get_parent(trian)
  uindices = get_union_indices(parent,trians)
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
    order_domains(tparents::Tuple{Vararg{Triangulation}},
      tchildren::Tuple{Vararg{Triangulation}}) -> Tuple{Vararg{Triangulation}}

Orders the triangulation children in the same way as the triangulation parents

"""
function order_domains(tparents,tchildren)
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

# correct bug

function Base.view(trian::Geometry.AppendedTriangulation,ids::AbstractArray)
  Ti = eltype(ids)
  ids1 = Ti[]
  ids2 = Ti[]
  n1 = num_cells(trian.a)
  for i in ids
    if i <= n1
      push!(ids1,i)
    else
      push!(ids2,i-n1)
    end
  end
  trian1 = view(trian.a,ids1)
  trian2 = view(trian.b,ids2)
  isempty(ids1) ? trian2 : (isempty(ids2) ? trian1 : lazy_append(trian1,trian2))
end
