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

function get_parent(t::Triangulation)
  @abstractmethod
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

function merge_triangulations(trians)
  parent = get_parent(trians)
  uindices = get_union_indices(trians)
  view(parent,uindices)
end

function find_permutation(a,b)
  compare(a,b) = a == b || is_parent(a,b)
  map(a -> findfirst(b -> compare(a,b),b),a)
end

function order_triangulations(tparents,tchildren)
  @check length(tparents) == length(tchildren)
  iperm = find_permutation(tparents,tchildren)
  map(iperm->tchildren[iperm],iperm)
end
