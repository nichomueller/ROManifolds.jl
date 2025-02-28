module ParamGeometry

using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.Geometry
using Gridap.Helpers
using Gridap.ReferenceFEs

using GridapEmbedded
using GridapEmbedded.Interfaces
using GridapEmbedded.LevelSetCutters

using ROManifolds.Utils
using ROManifolds.ParamDataStructures

import FillArrays: Fill

export ParamMappedGrid
export ParamMappedDiscreteModel
export ParamUnstructuredGrid

function Geometry.pos_neg_data(
  ipos_to_val::AbstractArray{<:AbstractParamArray{<:Number}},
  i_to_iposneg::PosNegPartition)
  nineg = length(i_to_iposneg.ineg_to_i)
  val = testitem(ipos_to_val)
  zs = 0 .* size(val)
  void = similar(val,eltype(val),zs)
  ineg_to_val = Fill(void,nineg)
  ipos_to_val,ineg_to_val
end

function Geometry._setsize_compress!(a::AbstractParamArray,b::AbstractParamArray)
  @check param_length(a) == param_length(b)
  for i in param_eachindex(a)
    ai = param_getindex(a,i)
    bi = param_getindex(b,i)
    Geometry._setsize_compress!(ai,bi)
  end
end

function Geometry._uncached_compress!(a::AbstractParamArray,b::AbstractParamArray)
  @check param_length(a) == param_length(b)
  for i in param_eachindex(a)
    ai = param_getindex(a,i)
    bi = param_getindex(b,i)
    Geometry._uncached_compress!(ai,bi)
  end
  a
end

function Geometry._uncached_compress!(a::AbstractParamArray,b)
  for i in param_eachindex(a)
    ai = param_getindex(a,i)
    Geometry._uncached_compress!(ai,b)
  end
  a
end

function Geometry._copyto_compress!(a::AbstractParamArray,b::AbstractParamArray)
  @check param_length(a) == param_length(b)
  for i in param_eachindex(a)
    ai = param_getindex(a,i)
    bi = param_getindex(b,i)
    Geometry._copyto_compress!(ai,bi)
  end
  a
end

function Geometry._addto_compress!(a::AbstractParamArray,b::AbstractParamArray)
  @check param_length(a) == param_length(b)
  for i in param_eachindex(a)
    ai = param_getindex(a,i)
    bi = param_getindex(b,i)
    Geometry._addto_compress!(ai,bi)
  end
  a
end

function _node_ids_to_coords!(node_ids_to_coords::ArrayOfArrays,cell_node_ids,cell_to_coords)
  cache_node_ids = array_cache(cell_node_ids)
  cache_coords = array_cache(cell_to_coords)
  for k = 1:length(cell_node_ids)
    node_ids = getindex!(cache_node_ids,cell_node_ids,k)
    coords = getindex!(cache_coords,cell_to_coords,k)
    for (i,id) in enumerate(node_ids)
      for j in param_eachindex(node_ids_to_coords)
        node_ids_to_coords[j][node_ids[i]] = coords[j][i]
      end
    end
  end
end

struct ParamMappedGrid{Dc,Dp,T,M,L} <: Grid{Dc,Dp}
  grid::Grid{Dc,Dp}
  geo_map::T
  phys_map::M
  node_coords::L
end

function Geometry.MappedGrid(grid::Grid,phys_map::AbstractVector{<:GenericParamField})

  @assert length(phys_map) == num_cells(grid)

  function _compute_node_coordinates(grid,phys_map)
    cell_node_ids = get_cell_node_ids(grid)
    old_nodes = get_node_coordinates(grid)
    node_coordinates = Vector{eltype(old_nodes)}(undef,length(old_nodes))
    plength = param_length(testitem(phys_map))
    pnode_coordinates = parameterize(node_coordinates,plength)
    cell_to_coords = get_cell_coordinates(grid)
    cell_coords_map = lazy_map(evaluate,phys_map,cell_to_coords)
    _node_ids_to_coords!(pnode_coordinates,cell_node_ids,cell_coords_map)
    return pnode_coordinates
  end

  model_map = get_cell_map(grid)
  geo_map = lazy_map(∘,phys_map,model_map)
  node_coords = _compute_node_coordinates(grid,phys_map)
  ParamMappedGrid(grid,geo_map,phys_map,node_coords)
end

Geometry.get_node_coordinates(grid::ParamMappedGrid) = grid.node_coords
Geometry.get_cell_node_ids(grid::ParamMappedGrid) = get_cell_node_ids(grid.grid)
Geometry.get_reffes(grid::ParamMappedGrid) = get_reffes(grid.grid)
Geometry.get_cell_type(grid::ParamMappedGrid) = get_cell_type(grid.grid)

"""
MappedDiscreteModel

Represent a model with a `MappedGrid` grid.
See also [`MappedGrid`](@ref).
"""
struct ParamMappedDiscreteModel{Dc,Dp} <: DiscreteModel{Dc,Dp}
  model::DiscreteModel{Dc,Dp}
  mapped_grid::ParamMappedGrid{Dc,Dp}
end

function Geometry.MappedDiscreteModel(model::DiscreteModel,mapped_grid::ParamMappedGrid)
  ParamMappedDiscreteModel(model,mapped_grid)
end

function Geometry.MappedDiscreteModel(model::DiscreteModel,phys_map::AbstractParamFunction)
  mapped_grid = MappedGrid(get_grid(model),phys_map)
  MappedDiscreteModel(model,mapped_grid)
end

Geometry.get_grid(model::ParamMappedDiscreteModel) = model.mapped_grid
Geometry.get_cell_map(model::ParamMappedDiscreteModel) = get_cell_map(model.mapped_grid)
Geometry.get_grid_topology(model::ParamMappedDiscreteModel) = get_grid_topology(model.model)
Geometry.get_face_labeling(model::ParamMappedDiscreteModel) = get_face_labeling(model.model)

function Geometry.Grid(::Type{ReferenceFE{d}},model::ParamMappedDiscreteModel) where d
  node_coordinates = get_node_coordinates(model)
  cell_to_nodes = Table(get_face_nodes(model,d))
  cell_to_type = collect1d(get_face_type(model,d))
  reffes = get_reffaces(ReferenceFE{d},model)
  UnstructuredGrid(node_coordinates,cell_to_nodes,reffes,cell_to_type)
end

struct ParamUnstructuredGrid{Dc,Dp,Tp,O,Tn} <: Grid{Dc,Dp}
  node_coordinates::ParamVector{Point{Dp,Tp}}
  cell_node_ids::Table{Int32,Vector{Int32},Vector{Int32}}
  reffes::Vector{LagrangianRefFE{Dc}}
  cell_types::Vector{Int8}
  orientation_style::O
  facet_normal::Tn
  cell_map

  function ParamUnstructuredGrid(
    node_coordinates::ParamVector{Point{Dp,Tp}},
    cell_node_ids::Table{Ti},
    reffes::Vector{<:LagrangianRefFE{Dc}},
    cell_types::Vector,
    orientation_style::OrientationStyle=NonOriented(),
    facet_normal=nothing;
    has_affine_map=nothing) where {Dc,Dp,Tp,Ti}

    if has_affine_map === nothing
      _has_affine_map = Geometry.get_has_affine_map(reffes)
    else
      _has_affine_map = has_affine_map
    end
    cell_map = Geometry._compute_cell_map(node_coordinates,cell_node_ids,reffes,cell_types,_has_affine_map)
    B = typeof(orientation_style)
    Tn = typeof(facet_normal)
    new{Dc,Dp,Tp,B,Tn}(
      node_coordinates,
      cell_node_ids,
      reffes,
      cell_types,
      orientation_style,
      facet_normal,
      cell_map)
  end
end

function Geometry.UnstructuredGrid(node_coordinates::ParamVector{<:Point},args...;kwargs...)
  ParamUnstructuredGrid(node_coordinates,args...;kwargs...)
end

function Geometry.UnstructuredGrid(grid::ParamUnstructuredGrid)
  grid
end

Geometry.OrientationStyle(
  ::Type{<:ParamUnstructuredGrid{Dc,Dp,Tp,B}}) where {Dc,Dp,Tp,B} = B()

Geometry.get_reffes(g::ParamUnstructuredGrid) = g.reffes
Geometry.get_cell_type(g::ParamUnstructuredGrid) = g.cell_types
Geometry.get_node_coordinates(g::ParamUnstructuredGrid) = g.node_coordinates
Geometry.get_cell_node_ids(g::ParamUnstructuredGrid) = g.cell_node_ids
Geometry.get_cell_map(g::ParamUnstructuredGrid) = g.cell_map

function Geometry.get_facet_normal(g::ParamUnstructuredGrid)
  @assert g.facet_normal != nothing "This Grid does not have information about normals."
  g.facet_normal
end

function Geometry.simplexify(grid::ParamUnstructuredGrid;kwargs...)
  reffes = get_reffes(grid)
  @notimplementedif length(reffes) != 1
  reffe = first(reffes)
  order = 1
  @notimplementedif get_order(reffe) != order
  p = get_polytope(reffe)
  ltcell_to_lpoints, simplex = simplexify(p;kwargs...)
  cell_to_points = get_cell_node_ids(grid)
  tcell_to_points = Geometry._refine_grid_connectivity(cell_to_points,ltcell_to_lpoints)
  ctype_to_reffe = [LagrangianRefFE(Float64,simplex,order),]
  tcell_to_ctype = fill(Int8(1),length(tcell_to_points))
  point_to_coords = get_node_coordinates(grid)
  ParamUnstructuredGrid(
    point_to_coords,
    tcell_to_points,
    ctype_to_reffe,
    tcell_to_ctype,
    Oriented())
end

end
