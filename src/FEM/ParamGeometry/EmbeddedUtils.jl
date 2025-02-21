struct NodesInOrOut
  in_or_out::Int
end

const NODES_IN = NodesInOrOut(IN)
const NODES_OUT = NodesInOrOut(OUT)

function get_nodes_to_cut_mask(cut::EmbeddedDiscretization)
  filter = NODES_IN
  get_nodes_to_cut_mask(filter,cut)
end

function get_nodes_to_cut_mask(filter::NodesInOrOut,cut::EmbeddedDiscretization)
  get_nodes_to_cut_mask(filter,cut,cut.geo)
end

function get_nodes_to_cut_mask(filter::NodesInOrOut,cut::EmbeddedDiscretization,geo::Interfaces.CSG.Geometry)
  model = cut.bgmodel
  D = num_cell_dims(model)
  grid = Grid(ReferenceFE{D-1},model)
  cutter = LevelSetCutter()
  facet_to_inoutcut = LevelSetCutters._compute_bgcell_to_inoutcut(grid,geo)
  _get_nodes_to_cut_mask(filter,grid,facet_to_inoutcut)
end

function _get_nodes_to_cut_mask(filter::NodesInOrOut,grid::Grid,facet_to_inoutcut::AbstractVector)
  node_ids = get_cell_node_ids(grid)
  nodes_to_mask = ones(Bool,num_nodes(grid))
  for (facet,inoutcut) in enumerate(facet_to_inoutcut)
    if inoutcut == filter.in_or_out
      nodes = node_ids[facet]
      for node in nodes
        nodes_to_mask[node] = false
      end
    end
  end
  nodes_to_mask
end

struct TriangulationFilter{Dt,Dp} <: Triangulation{Dt,Dp}
  trian::Triangulation
  nodes_to_mask::AbstractVector
end

function Geometry.Triangulation(
  cut::EmbeddedDiscretization,
  in_or_out::Interfaces.ActiveInOrOut,
  filter::NodesInOrOut,
  geo::Interfaces.CSG.Geometry)

  cell_to_inoutcut = compute_bgcell_to_inoutcut(cut,geo)
  cell_to_mask = lazy_map(i-> i==CUT || i==in_or_out.in_or_out,cell_to_inoutcut)
  trian = Triangulation(cut.bgmodel,cell_to_mask)
  nodes_to_mask = get_nodes_to_cut_mask(filter,cut,geo)
  TriangulationFilter(trian,nodes_to_mask)
end

function Geometry.Triangulation(
  cut::EmbeddedDiscretization,
  in_or_out::Interfaces.ActiveInOrOut,
  filter::NodesInOrOut)

  Triangulation(cut,in_or_out,filter,cut.geo)
end
