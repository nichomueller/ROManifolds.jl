# function Geometry.UnstructuredGridTopology(grid::RBUnstructuredGrid)
#   cell_to_vertices,vertex_to_node, = Geometry._generate_cell_to_vertices_from_grid(grid)
#   Geometry._generate_grid_topology_from_grid(grid,cell_to_vertices,vertex_to_node)
# end

# function Geometry._generate_grid_topology_from_grid(grid::RBUnstructuredGrid,cell_to_vertices,vertex_to_node)

#   @notimplementedif (! is_regular(grid)) "Extrtacting the GridTopology form a Grid only implemented for the regular case"

#   node_to_coords = get_node_coordinates(grid)
#   if vertex_to_node == 1:num_nodes(grid)
#     vertex_to_coords = node_to_coords
#   else
#     vertex_to_coords = node_to_coords[vertex_to_node]
#   end

#   cell_to_type = get_cell_type(grid)
#   polytopes = map(get_polytope, get_reffes(grid))

#   UnstructuredGridTopology(
#     vertex_to_coords,
#     cell_to_vertices,
#     cell_to_type,
#     polytopes,
#     OrientationStyle(grid))
# end

# function Geometry._generate_cell_to_vertices_from_grid(grid::RBUnstructuredGrid)
#   if is_first_order(grid)
#     cell_to_vertices = Table(get_cell_node_ids(grid))
#     vertex_to_node = collect(1:num_nodes(grid))
#     node_to_vertex = vertex_to_node
#   else
#     cell_to_nodes = get_cell_node_ids(grid)
#     cell_to_cell_type = get_cell_type(grid)
#     reffes = get_reffes(grid)
#     cell_type_to_lvertex_to_lnode = map(get_vertex_node, reffes)
#     cell_to_vertices, vertex_to_node, node_to_vertex = _generate_cell_to_vertices(
#       cell_to_nodes,
#       cell_to_cell_type,
#       cell_type_to_lvertex_to_lnode,
#       num_nodes(grid))
#   end
#   (cell_to_vertices, vertex_to_node, node_to_vertex)
# end

# function reduce_background_model(trian::Triangulation,rgrid::Grid)
#   model = get_background_model(trian)
#   rtopo_grid = UnstructuredGridTopology(rgrid)
#   labels = get_face_labeling(model)
#   GenericDiscreteModel(rgrid,rtopo_grid,labels)
# end

# struct RBUnstructuredGrid{Dc,Dp,Tp,O,Tn} <: Grid{Dc,Dp}
#   node_coordinates::Vector{Point{Dp,Tp}}
#   cell_node_ids::Table{Int32,Vector{Int32},Vector{Int32}}
#   reffes::Vector{LagrangianRefFE{Dc}}
#   cell_types::Vector{Int8}
#   orientation_style::O
#   facet_normal::Tn
#   cell_map

#   function RBUnstructuredGrid(
#     node_coordinates::Vector{Point{Dp,Tp}},
#     cell_node_ids::Table{Ti},
#     reffes::Vector{<:LagrangianRefFE{Dc}},
#     cell_types::Vector,
#     cell_map::AbstractArray,
#     orientation_style::B=NonOriented(),
#     facet_normal::Tn=nothing) where {Dc,Dp,Tp,Ti,B,Tn}

#     new{Dc,Dp,Tp,B,Tn}(
#       node_coordinates,
#       cell_node_ids,
#       reffes,
#       cell_types,
#       orientation_style,
#       facet_normal,
#       cell_map)
#   end
# end

# function RBUnstructuredGrid(grid::Grid,ids::AbstractArray)
#   node_coordinates = collect1d(get_node_coordinates(grid))
#   node_ids = get_cell_node_ids(grid)
#   active_nodes, = Geometry._find_active_nodes(node_ids,ids,num_nodes(grid))
#   ids_node_coordinates = node_coordinates[active_nodes]
#   cell_node_ids = Table(map(Reindex(node_ids),ids))
#   reffes = get_reffes(grid)
#   cell_types = collect1d(map(Reindex(get_cell_type(grid)),ids))
#   cell_map = lazy_map(Reindex(get_cell_map(grid)),ids)
#   orien = OrientationStyle(grid)
#   RBUnstructuredGrid(ids_node_coordinates,cell_node_ids,reffes,cell_types,cell_map,orien)
# end

# Geometry.get_reffes(g::RBUnstructuredGrid) = g.reffes
# Geometry.get_cell_type(g::RBUnstructuredGrid) = g.cell_types
# Geometry.get_node_coordinates(g::RBUnstructuredGrid) = g.node_coordinates
# Geometry.get_cell_node_ids(g::RBUnstructuredGrid) = g.cell_node_ids
# Geometry.get_cell_map(g::RBUnstructuredGrid) = g.cell_map

# function Geometry.get_facet_normal(g::RBUnstructuredGrid)
#   @assert !isnothing(g.facet_normal) "This Grid does not have information about normals."
#   g.facet_normal
# end

# function Geometry.to_dict(grid::UnstructuredGrid)
#   dict = Dict{Symbol,Any}()
#   x = get_node_coordinates(grid)
#   dict[:node_coordinates] = reinterpret(eltype(eltype(x)),x)
#   dict[:Dp] = num_point_dims(grid)
#   dict[:cell_node_ids] = to_dict(get_cell_node_ids(grid))
#   dict[:reffes] = map(to_dict, get_reffes(grid))
#   dict[:cell_type] = get_cell_type(grid)
#   dict[:orientation] = is_oriented(grid)
#   dict
# end

# struct RBBodyFittedTriangulation{Dt,Dp,A,B,C} <: Triangulation{Dt,Dp}
#   model::A
#   grid::B
#   tface_to_mface::C
#   function RBBodyFittedTriangulation(model::DiscreteModel,grid::Grid,tface_to_mface)
#     Dp = num_point_dims(model)
#     @assert Dp == num_point_dims(grid)
#     Dt = num_cell_dims(grid)
#     A = typeof(model)
#     B = typeof(grid)
#     C = typeof(tface_to_mface)
#     new{Dt,Dp,A,B,C}(model,grid,tface_to_mface)
#   end
# end

# Geometry.get_background_model(trian::RBBodyFittedTriangulation) = trian.model
# Geometry.get_grid(trian::RBBodyFittedTriangulation) = trian.grid

# function Geometry.get_glue(trian::RBBodyFittedTriangulation{Dt},::Val{Dt}) where Dt
#   tface_to_mface = trian.tface_to_mface
#   tface_to_mface_map = Fill(GenericField(identity),num_cells(trian))
#   nmfaces = num_faces(trian.model,Dt)
#   mface_to_tface = PosNegPartition(tface_to_mface,Int32(nmfaces))
#   FaceToFaceGlue(tface_to_mface,tface_to_mface_map,mface_to_tface)
# end

# function RBTriangulation(trian::BodyFittedTriangulation,ids::AbstractArray)
#   grid = RBUnstructuredGrid(trian.grid,ids)
#   model = reduce_background_model(trian,grid)
#   tface_to_mface = lazy_map(Reindex(trian.tface_to_mface),ids)
#   RBBodyFittedTriangulation(model,grid,tface_to_mface)
# end

###############
###############
struct MyGridPortion{Dc,Dp,Tp,O,Tn} <: Grid{Dc,Dp}
  cell_to_parent_cell::Vector{Int32}
  node_coordinates::Vector{Point{Dp,Tp}}
  cell_node_ids::Table{Int32,Vector{Int32},Vector{Int32}}
  reffes::Vector{LagrangianRefFE{Dc}}
  cell_types::Vector{Int8}
  orientation_style::O
  facet_normal::Tn
  cell_map

  function MyGridPortion(
    cell_to_parent_cell::Vector{Int32},
    node_coordinates::Vector{Point{Dp,Tp}},
    cell_node_ids::Table{Ti},
    reffes::Vector{<:LagrangianRefFE{Dc}},
    cell_types::Vector,
    cell_map::AbstractArray,
    orientation_style::B=NonOriented(),
    facet_normal::Tn=nothing) where {Dc,Dp,Tp,Ti,B,Tn}

    new{Dc,Dp,Tp,B,Tn}(
      cell_to_parent_cell,
      node_coordinates,
      cell_node_ids,
      reffes,
      cell_types,
      orientation_style,
      facet_normal,
      cell_map)
  end
end

function MyGridPortion(grid::Grid,ids::AbstractArray)
  node_coordinates = collect1d(get_node_coordinates(grid))
  cell_node = get_cell_node_ids(grid)[ids]
  reffes = get_reffes(grid)
  cell_types = get_cell_type(grid)[ids]
  # cell_map = lazy_map(Reindex(get_cell_map(grid)),ids)
  cell_map = map(Reindex(get_cell_map(grid)),ids)
  orien = OrientationStyle(grid)
  MyGridPortion(ids,node_coordinates,cell_node,reffes,cell_types,cell_map,orien)
end

Geometry.get_reffes(g::MyGridPortion) = g.reffes
Geometry.get_cell_type(g::MyGridPortion) = g.cell_types
Geometry.get_node_coordinates(g::MyGridPortion) = g.node_coordinates
Geometry.get_cell_node_ids(g::MyGridPortion) = g.cell_node_ids
Geometry.get_cell_map(g::MyGridPortion) = g.cell_map

function Geometry.get_facet_normal(g::MyGridPortion)
  @assert !isnothing(g.facet_normal) "This Grid does not have information about normals."
  g.facet_normal
end

function Geometry.to_dict(grid::UnstructuredGrid)
  dict = Dict{Symbol,Any}()
  x = get_node_coordinates(grid)
  dict[:node_coordinates] = reinterpret(eltype(eltype(x)),x)
  dict[:Dp] = num_point_dims(grid)
  dict[:cell_node_ids] = to_dict(get_cell_node_ids(grid))
  dict[:reffes] = map(to_dict, get_reffes(grid))
  dict[:cell_type] = get_cell_type(grid)
  dict[:orientation] = is_oriented(grid)
  dict
end

struct MyDiscreteModelPortion{Dc,Dp} <: DiscreteModel{Dc,Dp}
  model::DiscreteModel{Dc,Dp}
  parent_model::DiscreteModel{Dc,Dp}
  d_to_dface_to_parent_dface::Vector{Vector{Int}}
end

Geometry.get_grid(model::MyDiscreteModelPortion) = get_grid(model.model)
Geometry.get_grid_topology(model::MyDiscreteModelPortion) = get_grid_topology(model.model)
Geometry.get_face_labeling(model::MyDiscreteModelPortion) = get_face_labeling(model.model)
Geometry.get_face_to_parent_face(model::MyDiscreteModelPortion,d::Integer) = model.d_to_dface_to_parent_dface[d+1]
Geometry.get_cell_to_parent_cell(model::MyDiscreteModelPortion) = get_face_to_parent_face(model,num_cell_dims(model))
Geometry.get_parent_model(model::MyDiscreteModelPortion) = model.parent_model

function MyDiscreteModelPortion(model::DiscreteModel,cell_to_parent_cell::AbstractVector{<:Integer})
  topo = get_grid_topology(model)
  labels = get_face_labeling(model)
  grid_p =  MyGridPortion(get_grid(model),cell_to_parent_cell)
  topo_p,d_to_dface_to_parent_dface = Geometry._grid_topology_portion(topo,cell_to_parent_cell)
  labels_p = Geometry._setup_labels_p(labels,d_to_dface_to_parent_dface)
  model_p = DiscreteModel(grid_p,topo_p,labels_p)
  MyDiscreteModelPortion(model_p,model,d_to_dface_to_parent_dface)
end

function MyDiscreteModelPortion(model::DiscreteModel,cell_to_is_in::AbstractVector{Bool})
  cell_to_parent_cell = findall(cell_to_is_in)
  MyDiscreteModelPortion(model,cell_to_parent_cell)
end

function MyDiscreteModelPortion(model::DiscreteModel,grid_p::MyGridPortion)
  topo = get_grid_topology(model)
  labels = get_face_labeling(model)
  cell_to_parent_cell = grid_p.cell_to_parent_cell
  topo_p,d_to_dface_to_parent_dface = Geometry._grid_topology_portion(topo,cell_to_parent_cell)
  labels_p = Geometry._setup_labels_p(labels,d_to_dface_to_parent_dface)
  model_p = DiscreteModel(grid_p,topo_p,labels_p)
  MyDiscreteModelPortion(model_p,model,d_to_dface_to_parent_dface)
end

# function Arrays.lazy_map(
#   k::Reindex{<:LazyArray{<:Fill,M,N}},
#   ::Type{T},
#   j_to_i::AbstractArray
#   )::LazyArray{<:Fill,M,N} where {M,N,T}

#   i_to_maps = k.values.maps
#   i_to_args = k.values.args
#   j_to_maps = lazy_map(Reindex(i_to_maps),eltype(i_to_maps),j_to_i)
#   j_to_args = map(i_to_fk->lazy_map(Reindex(i_to_fk),eltype(i_to_fk),j_to_i),i_to_args)
#   LazyArray(T,j_to_maps,j_to_args...)
# end
