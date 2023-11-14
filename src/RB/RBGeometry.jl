struct RBGridPortion{Dc,Dp,Tp,O,Tn} <: Grid{Dc,Dp}
  cell_to_parent_cell::Vector{Int32}
  node_coordinates::Vector{Point{Dp,Tp}}
  cell_node_ids::Table{Int32,Vector{Int32},Vector{Int32}}
  reffes::Vector{LagrangianRefFE{Dc}}
  cell_types::Vector{Int8}
  orientation_style::O
  facet_normal::Tn
  cell_map

  function RBGridPortion(
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

function RBGridPortion(grid::Grid,cell_to_parent_cell::AbstractArray)
  node_coordinates = collect1d(get_node_coordinates(grid))
  cell_node_ids = get_cell_node_ids(grid)[cell_to_parent_cell]
  reffes = get_reffes(grid)
  cell_types = get_cell_type(grid)[cell_to_parent_cell]
  _has_affine_map = Geometry.get_has_affine_map(reffes)
  cell_map = Geometry._compute_cell_map(node_coordinates,cell_node_ids,reffes,cell_types,_has_affine_map)
  orien = OrientationStyle(grid)
  RBGridPortion(cell_to_parent_cell,node_coordinates,cell_node_ids,reffes,cell_types,cell_map,orien)
end

Geometry.get_reffes(g::RBGridPortion) = g.reffes
Geometry.get_cell_type(g::RBGridPortion) = g.cell_types
Geometry.get_node_coordinates(g::RBGridPortion) = g.node_coordinates
Geometry.get_cell_node_ids(g::RBGridPortion) = g.cell_node_ids
Geometry.get_cell_map(g::RBGridPortion) = g.cell_map
get_cell_to_parent_cell(g::RBGridPortion) = g.cell_to_parent_cell

function Geometry.get_facet_normal(g::RBGridPortion)
  @assert !isnothing(g.facet_normal) "This Grid does not have information about normals."
  g.facet_normal
end

struct RBDiscreteModelPortion{Dc,Dp} <: DiscreteModel{Dc,Dp}
  model::DiscreteModel{Dc,Dp}
  parent_model::DiscreteModel{Dc,Dp}
  d_to_dface_to_parent_dface::Vector{Vector{Int}}
end

Geometry.get_grid(model::RBDiscreteModelPortion) = get_grid(model.model)
Geometry.get_grid_topology(model::RBDiscreteModelPortion) = get_grid_topology(model.model)
Geometry.get_face_labeling(model::RBDiscreteModelPortion) = get_face_labeling(model.model)
Geometry.get_face_to_parent_face(model::RBDiscreteModelPortion,d::Integer) = model.d_to_dface_to_parent_dface[d+1]
Geometry.get_cell_to_parent_cell(model::RBDiscreteModelPortion) = get_face_to_parent_face(model,num_cell_dims(model))
Geometry.get_parent_model(model::RBDiscreteModelPortion) = model.parent_model
get_cell_to_parent_cell(model::RBDiscreteModelPortion) = get_cell_to_parent_cell(get_grid(model.model))

function RBDiscreteModelPortion(model::DiscreteModel,cell_to_parent_cell::AbstractVector)
  grid_p = RBGridPortion(get_grid(model),cell_to_parent_cell)
  RBDiscreteModelPortion(model,grid_p)
end

function RBDiscreteModelPortion(model::DiscreteModel,grid_p::RBGridPortion)
  topo = get_grid_topology(model)
  labels = get_face_labeling(model)
  cell_to_parent_cell = grid_p.cell_to_parent_cell
  topo_p,d_to_dface_to_parent_dface = Geometry._grid_topology_portion(topo,cell_to_parent_cell)
  labels_p = Geometry._setup_labels_p(labels,d_to_dface_to_parent_dface)
  model_p = DiscreteModel(grid_p,topo_p,labels_p)
  RBDiscreteModelPortion(model_p,model,d_to_dface_to_parent_dface)
end

function get_reduced_dirichlet_dof_ids(test,reduced_dirichlet_cells)
  if isempty(reduced_dirichlet_cells)
    return Int32[]
  end
  reduced_dirichlet_cell_dofs_ids = map(i->test.cell_dofs_ids[i],reduced_dirichlet_cells)
  reduced_dirichlet_dofs = vcat(filter.(x->x<0,reduced_dirichlet_cell_dofs_ids)...)
  @. reduced_dirichlet_dofs *= -one(Int32)
  reduced_sorted_dirichlet_dofs = sort(reduced_dirichlet_dofs)
  unique(reduced_sorted_dirichlet_dofs)
end

function get_reduced_free_dof_ids(reduced_cell_dofs_ids)
  reduced_dofs = vcat(filter.(x->x>0,reduced_cell_dofs_ids)...)
  reduced_sorted_dofs = sort(reduced_dofs)
  unique(reduced_sorted_dofs)
end

function get_reduced_node_and_comp_to_dof(node_and_comp_to_dof,reduced_free_dofs,reduced_dirichlet_dofs)
  reduced_dofs = union(reduced_free_dofs,-reduced_dirichlet_dofs)
  nreddofs = length(reduced_dofs)
  reduced_node_and_comp_to_dof = Vector{Int32}(undef,nreddofs)
  count = 1
  for node_comp in node_and_comp_to_dof
    if node_comp ∈ reduced_dofs
      reduced_node_and_comp_to_dof[count] = node_comp
      count += 1
    end
  end
  reduced_node_and_comp_to_dof
end

function reduce_test(test::UnconstrainedFESpace,trian::Triangulation)
  model = get_background_model(trian)
  cell_to_parent_cell = get_cell_to_parent_cell(model)
  cell_dofs_ids = test.cell_dofs_ids[cell_to_parent_cell]
  cell_is_dirichlet = test.cell_is_dirichlet[cell_to_parent_cell]
  dirichlet_cells = intersect(test.dirichlet_cells,cell_to_parent_cell)
  cell_basis = test.fe_basis.cell_basis[cell_to_parent_cell]
  fe_basis = FESpaces.SingleFieldFEBasis(cell_basis,trian,FESpaces.TestBasis(),test.fe_basis.domain_style)
  cell_dof_basis = test.fe_dof_basis.cell_dof[cell_to_parent_cell]
  fe_dof_basis = CellDof(cell_dof_basis,trian,test.fe_dof_basis.domain_style)
  dirichlet_dofs = get_reduced_dirichlet_dof_ids(test,dirichlet_cells)
  free_dofs = get_reduced_free_dof_ids(cell_dofs_ids)
  ndirichlet = length(dirichlet_dofs)
  nfree = length(free_dofs)
  ntags = test.ntags
  vector_type = test.vector_type
  dirichlet_dof_tag = test.dirichlet_dof_tag[dirichlet_dofs]
  glue = test.metadata
  if isnothing(glue)
    UnconstrainedFESpace(vector_type,nfree,ndirichlet,cell_dofs_ids,fe_basis,
      fe_dof_basis,cell_is_dirichlet,dirichlet_dof_tag,dirichlet_cells,ntags)
  else
    dirichlet_dof_to_comp = glue.dirichlet_dof_to_comp[dirichlet_dofs]
    dirichlet_dof_to_node = glue.dirichlet_dof_to_node[dirichlet_dofs]
    free_dof_to_comp = glue.free_dof_to_comp[free_dofs]
    free_dof_to_node = glue.free_dof_to_node[free_dofs]
    node_and_comp_to_dof = get_reduced_node_and_comp_to_dof(glue.node_and_comp_to_dof,free_dofs,dirichlet_dofs)
    metadata = FESpaces.NodeToDofGlue(free_dof_to_node,free_dof_to_comp,
      dirichlet_dof_to_node,dirichlet_dof_to_comp,node_and_comp_to_dof)
    UnconstrainedFESpace(vector_type,nfree,ndirichlet,cell_dofs_ids,fe_basis,
      fe_dof_basis,cell_is_dirichlet,dirichlet_dof_tag,dirichlet_cells,ntags,metadata)
  end
end

function reduce_trial(trial::PTTrialFESpace,test::FESpace)
  PTTrialFESpace(test,trial.dirichlet_μt)
end

function reduce_feoperator(feop::PTFEOperator{Affine},trian::Triangulation)
  red_test = reduce_test(get_test(feop),trian)
  red_trial = reduce_trial(get_trial(feop),red_test)
  AffinePTFEOperator(feop.res,feop.jacs[1],feop.jacs[2],feop.pspace,red_trial,red_test)
end

function reduce_feoperator(feop::PTFEOperator,trian::Triangulation)
  red_test = reduce_test(get_test(feop),trian)
  red_trial = reduce_trial(get_trial(feop),red_test)
  PTFEOperator(feop.res,feop.jacs[1],feop.jacs[2],feop.nl,feop.pspace,red_trial,red_test)
end
