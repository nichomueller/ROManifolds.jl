abstract type OrderedFESpace{S} <: SingleFieldFESpace end

# FESpace interface

FESpaces.get_fe_space(f::OrderedFESpace) = @abstractmethod

get_ordered_dof_ids(f::OrderedFESpace) = @abstractmethod

FESpaces.ConstraintStyle(::Type{<:OrderedFESpace{S}}) where S = ConstraintStyle(S)

FESpaces.get_free_dof_ids(f::OrderedFESpace) = get_free_dof_ids(get_fe_space(f))

FESpaces.get_triangulation(f::OrderedFESpace) = get_triangulation(get_fe_space(f))

FESpaces.get_dof_value_type(f::OrderedFESpace) = get_dof_value_type(get_fe_space(f))

FESpaces.get_cell_dof_ids(f::OrderedFESpace) = get_ordered_dof_ids(f)

FESpaces.get_fe_basis(f::OrderedFESpace) = get_fe_basis(get_fe_space(f))

FESpaces.get_trial_fe_basis(f::OrderedFESpace) = get_trial_fe_basis(get_fe_space(f))

FESpaces.get_fe_dof_basis(f::OrderedFESpace) = get_fe_dof_basis(get_fe_space(f))

FESpaces.get_cell_isconstrained(f::OrderedFESpace) = get_cell_isconstrained(get_fe_space(f))

FESpaces.get_cell_constraints(f::OrderedFESpace) = get_cell_constraints(get_fe_space(f))

FESpaces.get_dirichlet_dof_ids(f::OrderedFESpace) = get_dirichlet_dof_ids(get_fe_space(f))

FESpaces.get_cell_is_dirichlet(f::OrderedFESpace) = get_cell_is_dirichlet(get_fe_space(f))

FESpaces.num_dirichlet_dofs(f::OrderedFESpace) = num_dirichlet_dofs(get_fe_space(f))

FESpaces.num_dirichlet_tags(f::OrderedFESpace) = num_dirichlet_tags(get_fe_space(f))

FESpaces.get_dirichlet_dof_tag(f::OrderedFESpace) = get_dirichlet_dof_tag(get_fe_space(f))

FESpaces.get_vector_type(f::OrderedFESpace) = get_vector_type(get_fe_space(f))

function FESpaces.scatter_free_and_dirichlet_values(f::OrderedFESpace,fv,dv)
  cell_dof_ids = get_cell_dof_ids(f)
  lazy_map(Broadcasting(PosNegReindex(fv,dv)),cell_dof_ids)
end

function FESpaces.gather_free_and_dirichlet_values(f::OrderedFESpace,cv)
  gather_free_and_dirichlet_values(get_fe_space(f),cv)
end

function FESpaces.gather_free_and_dirichlet_values!(fv,dv,f::OrderedFESpace,cv)
  gather_free_and_dirichlet_values!(fv,dv,get_fe_space(f),cv)
end

# Local functions

get_dof_to_mask(f::OrderedFESpace) = get_dof_to_mask(get_fe_space(f))

function get_cell_dof_ids_with_zeros(f::OrderedFESpace)
  cellids = get_cell_dof_ids(f)
  dof_mask = get_dof_to_mask(f)
  cellids_mask = lazy_map(MaskEntryMap(dof_mask),cellids)
  bg_cell_to_mask = _get_bg_cell_to_mask(f)
  bg_cells = LinearIndices(bg_cell_to_mask)
  lazy_map(AddZeroCellIdsMap(cellids_mask,bg_cell_to_mask),bg_cells)
end

get_bg_dofs_to_act_dofs(f::OrderedFESpace) = get_free_dof_ids(f)

function get_dof_map(f::OrderedFESpace)
  bg_dof_to_mask = get_dof_to_mask(f)
  bg_ndofs = length(bg_dof_to_mask)
  return VectorDofMap(bg_ndofs,bg_dof_to_mask)
end

# Constructors

function OrderedFESpace(model::CartesianDiscreteModel,args...;kwargs...)
  CartesianFESpace(model,args...;kwargs...)
end

function OrderedFESpace(trian::Triangulation,args...;kwargs...)
  act_model = get_active_model(trian)
  bg_model = get_background_model(trian)
  OrderedFESpace(act_model,bg_model,args...;kwargs...)
end

function OrderedFESpace(
  act_model::DiscreteModel,
  bg_model::CartesianDiscreteModel,
  args...;kwargs...)

  act_f = CartesianFESpace(act_model,args...;kwargs...)
  bg_f = CartesianFESpace(bg_model,args...;kwargs...)
  act_cell_ids = get_cell_dof_ids_with_zeros(act_f)
  act_dofs = _get_bg_dofs_to_act_dofs(bg_f,act_f)
  CartesianFESpacePortion(act_f,act_cell_ids,act_dofs)
end

function OrderedFESpace(::DiscreteModel,::DiscreteModel,args...;kwargs...)
  @notimplemented "Background model must be cartesian for the selected dof reordering"
end

struct CartesianFESpace{S<:SingleFieldFESpace} <: OrderedFESpace{S}
  space::S
  ordered_cell_dofs_ids::AbstractArray
end

function CartesianFESpace(f::SingleFieldFESpace)
  ordered_cell_dofs_ids = get_ordered_cell_dof_ids(f)
  CartesianFESpace(f,ordered_cell_dofs_ids)
end

function CartesianFESpace(model::DiscreteModel,args...;kwargs...)
  f = FESpace(model,args...;kwargs...)
  CartesianFESpace(f)
end

function CartesianFESpace(trian::Triangulation,args...;kwargs...)
  OrderedFESpace(trian,args...;kwargs...)
end

FESpaces.get_fe_space(f::CartesianFESpace) = f.space

get_ordered_dof_ids(f::CartesianFESpace) = f.ordered_cell_dofs_ids

struct CartesianFESpacePortion{S<:SingleFieldFESpace} <: OrderedFESpace{S}
  active_space::CartesianFESpace{S}
  bg_cell_ids_to_act_cell_ids::AbstractArray
  bg_dofs_to_act_dofs::AbstractVector
end

FESpaces.get_fe_space(f::CartesianFESpacePortion) = get_fe_space(f.active_space)

get_ordered_dof_ids(f::CartesianFESpacePortion) = get_ordered_dof_ids(f.active_space)

get_cell_dof_ids_with_zeros(f::CartesianFESpacePortion) = f.bg_cell_ids_to_act_cell_ids

get_bg_dofs_to_act_dofs(f::CartesianFESpacePortion) = f.bg_dofs_to_act_dofs

function get_dof_to_mask(f::CartesianFESpacePortion)
  act_dof_to_mask = get_dof_to_mask(f.active_space)
  bg_dof_to_act_dofs = get_bg_dofs_to_act_dofs(f)
  bg_dof_to_mask = zeros(Bool,length(bg_dof_to_act_dofs))
  for (bg_dof,act_dof) in enumerate(bg_dof_to_act_dofs)
    if iszero(act_dof)
      bg_dof_to_mask[bg_dof] = true
    else
      bg_dof_to_mask[bg_dof] = act_dof_to_mask[act_dof]
    end
  end
  return bg_dof_to_mask
end

# dof reordering

function get_ordered_cell_dof_ids(space::SingleFieldFESpace)
  cell_dofs_ids = get_cell_dof_ids(space)
  fe_dof_basis = get_data(get_fe_dof_basis(space))
  orders = get_polynomial_orders(space)
  cell_to_parent_cell = _get_bg_cell_to_act_cell(space)
  trian = get_triangulation(space)
  model = get_background_model(trian)
  get_ordered_cell_dof_ids(model,cell_dofs_ids,fe_dof_basis,cell_to_parent_cell,orders)
end

function get_ordered_cell_dof_ids(model::DiscreteModel,args...)
  parent = _get_parent_model(model)
  get_ordered_cell_dof_ids(parent,args...)
  # try parent = _get_parent_model(model)
  #   get_ordered_cell_dof_ids(parent,args...)
  # catch
  #   @notimplemented "Background model must be cartesian the selected dof reordering"
  # end
end

function get_ordered_cell_dof_ids(
  model::CartesianDiscreteModel,
  cell_dofs_ids::AbstractArray,
  fe_dof_basis::AbstractArray,
  cell_to_parent_cell::AbstractVector,
  orders::Tuple)

  desc = get_cartesian_descriptor(model)
  periodic = desc.isperiodic
  ncells = desc.partition
  cells = CartesianIndices(ncells)
  pcells = view(cells,cell_to_parent_cell)
  onodes = LinearIndices(orders .* ncells .+ 1 .- periodic)

  node_and_comps_to_odof = get_ordered_dof_ids(cell_dofs_ids,fe_dof_basis,pcells,onodes,orders)
  k = DofsToODofs(fe_dof_basis,node_and_comps_to_odof,orders)
  lazy_map(k,pcells)
end

# utils

cubic_polytope(::Val{d}) where d = @abstractmethod
cubic_polytope(::Val{1}) = SEGMENT
cubic_polytope(::Val{2}) = QUAD
cubic_polytope(::Val{3}) = HEX

function get_ordered_dof_ids(args...)
  @notimplemented "This function is only implemented for Lagrangian dof bases"
end

function get_ordered_dof_ids(
  cell_dofs_ids::AbstractArray,
  fe_dof_basis::Fill{<:LagrangianDofBasis},
  args...)

  get_ordered_dof_ids(cell_dofs_ids,testitem(fe_dof_basis),args...)
end

function get_ordered_dof_ids(
  cell_dofs_ids::AbstractArray,
  fe_dof_basis::LagrangianDofBasis{P,V},
  cells::AbstractVector{CartesianIndex{D}},
  onodes::LinearIndices{D},
  orders::NTuple{D,Int}
  ) where {D,P,V}

  cache = array_cache(cell_dofs_ids)
  p = cubic_polytope(Val(D))
  node_to_i_onode = _local_node_to_pnode(p,orders)
  nnodes = length(onodes)
  ncomps = num_components(V)
  ndofs = nnodes*ncomps
  vec_odofs = zeros(eltype(V),ndofs)
  for (icell,cell) in enumerate(cells)
    first_new_node = orders .* (Tuple(cell) .- 1) .+ 1
    onodes_range = map(enumerate(first_new_node)) do (i,ni)
      ni:ni+orders[i]
    end
    onodes_cell = view(onodes,onodes_range...)
    cell_dofs = getindex!(cache,cell_dofs_ids,icell)
    for node in 1:length(onodes_cell)
      comp_to_dof = fe_dof_basis.node_and_comp_to_dof[node]
      i_onode = node_to_i_onode[node]
      onode = onodes_cell[i_onode]
      for comp in 1:ncomps
        dof = cell_dofs[comp_to_dof[comp]]
        odof = onode + (comp-1)*nnodes
        vec_odofs[odof] = sign(dof)
      end
    end
  end

  nfree = 0
  ndiri = 0
  for (i,odof) in enumerate(vec_odofs)
    if odof > 0
      nfree += 1
      vec_odofs[i] = nfree
    else
      ndiri -= 1
      vec_odofs[i] = ndiri
    end
  end

  _get_node_and_comps_to_odof(fe_dof_basis,vec_odofs,onodes)
end

function get_ordered_dof_ids(
  cell_dofs_ids::AbstractArray,
  fe_basis::AbstractVector{<:Dof},
  args...)

  @notimplemented "This function is only implemented for Lagrangian dof bases"
end

function _get_node_and_comps_to_odof(
  ::LagrangianDofBasis{P,V},
  vec_odofs,
  onodes
  ) where {P,V}

  reshape(vec_odofs,size(onodes))
end

function _get_node_and_comps_to_odof(
  ::LagrangianDofBasis{P,V},
  vec_odofs,
  onodes
  ) where {P,V<:MultiValue}

  nnodes = length(onodes)
  ncomps = num_components(V)
  odofs = zeros(V,size(onodes))
  m = zero(Mutable(V))
  for onode in 1:nnodes
    for comp in 1:ncomps
      odof = onode + (comp-1)*nnodes
      m[comp] = vec_odofs[odof]
    end
    odofs[onode] = m
  end
  return odofs
end

function _local_pdof_to_dof(fe_dof_basis::AbstractVector{<:Dof},args...)
  @notimplemented "This function is only implemented for Lagrangian dof bases"
end

function _local_pdof_to_dof(fe_dof_basis::Fill{<:LagrangianDofBasis},orders::NTuple{D,Int}) where D
  _local_pdof_to_dof(testitem(fe_dof_basis),orders)
end

function _local_pdof_to_dof(b::LagrangianDofBasis,orders::NTuple{D,Int}) where D
  nnodes = length(b.node_and_comp_to_dof)
  ndofs = length(b.dof_to_comp)

  p = cubic_polytope(Val(D))
  _nodes, = Gridap.ReferenceFEs._compute_nodes(p,orders)
  node_to_pnode = Gridap.ReferenceFEs._coords_to_terms(_nodes,orders)
  node_to_pnode_linear = LinearIndices(orders.+1)[node_to_pnode]

  pdof_to_dof = zeros(Int32,ndofs)
  for (inode,ipnode) in enumerate(node_to_pnode_linear)
    for icomp in b.dof_to_comp
      local_shift = (icomp-1)*nnodes
      pdof_to_dof[local_shift+ipnode] = local_shift + inode
    end
  end

  return pdof_to_dof
end

function _local_node_to_pnode(p::Polytope,orders)
  _nodes, = Gridap.ReferenceFEs._compute_nodes(p,orders)
  pnodes = Gridap.ReferenceFEs._coords_to_terms(_nodes,orders)
  return pnodes
end

for f in (:_get_bg_cell_to_mask,:_get_bg_cell_to_act_cell)
  @eval begin
    function $f(f::SingleFieldFESpace)
      trian = get_triangulation(f)
      model = get_background_model(trian)
      grid = get_grid(model)
      $f(grid)
    end

    function $f(grid::Grid)
      @abstractmethod
    end
  end
end

function _get_bg_cell_to_mask(grid::CartesianGrid)
  fill(true,num_cells(grid))
end

function _get_bg_cell_to_act_cell(grid::CartesianGrid)
  IdentityVector(num_cells(grid))
end

function _get_bg_cell_to_mask(grid::GridPortion)
  bg_grid = grid.parent
  mask = ones(Bool,num_cells(bg_grid))
  for bg_cell in grid.cell_to_parent_cell
    mask[bg_cell] = false
  end
  return mask
end

function _get_bg_cell_to_act_cell(grid::GridPortion)
  grid.cell_to_parent_cell
end

function _get_bg_dofs_to_act_dofs(bg_f::CartesianFESpace,act_f::CartesianFESpace)
  bg_dof_to_act_dof = zeros(Int,num_free_dofs(bg_f))
  bg_cell_ids = get_cell_dof_ids(bg_f)
  act_cell_ids = get_cell_dof_ids(act_f)
  bg_cache = array_cache(bg_cell_ids)
  act_cache = array_cache(act_cell_ids)
  act_to_bg_cell = _get_bg_cell_to_act_cell(act_f)
  for (act_cell,bg_cell) in enumerate(act_to_bg_cell)
    bg_dofs = getindex!(bg_cache,bg_cell_ids,bg_cell)
    act_dofs = getindex!(act_cache,act_cell_ids,act_cell)
    for (bg_dof,act_dof) in zip(bg_dofs,act_dofs)
      bg_dof_to_act_dof[bg_dof] = act_dof
    end
  end
  return bg_dof_to_act_dof
end
