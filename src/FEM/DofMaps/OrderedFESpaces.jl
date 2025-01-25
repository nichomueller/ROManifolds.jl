abstract type OrderedFESpace{S} <: SingleFieldFESpace end

# FESpace interface

FESpaces.get_fe_space(f::OrderedFESpace) = @abstractmethod

get_cell_odof_ids(f::OrderedFESpace) = @abstractmethod

FESpaces.ConstraintStyle(::Type{<:OrderedFESpace{S}}) where S = ConstraintStyle(S)

FESpaces.get_free_dof_ids(f::OrderedFESpace) = get_free_dof_ids(get_fe_space(f))

FESpaces.get_triangulation(f::OrderedFESpace) = get_triangulation(get_fe_space(f))

FESpaces.get_dof_value_type(f::OrderedFESpace) = get_dof_value_type(get_fe_space(f))

FESpaces.get_cell_dof_ids(f::OrderedFESpace) = get_cell_odof_ids(f)

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

function get_dof_map(f::OrderedFESpace)
  bg_dof_to_mask = get_bg_dof_to_mask(f)
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
  OrderedFESpace(act_model,bg_model,trian,args...;kwargs...)
end

function OrderedFESpace(
  act_model::DiscreteModel,
  bg_model::CartesianDiscreteModel,
  act_trian::Triangulation,
  args...;kwargs...)

  act_f = CartesianFESpace(act_model,args...;trian=act_trian,kwargs...)
  bg_f = CartesianFESpace(bg_model,args...;kwargs...)
  act_dofs = _get_bg_odof_to_act_odof(bg_f,act_f)
  CartesianFESpace(act_f.space,act_f.cell_odofs_ids,act_dofs)
end

function OrderedFESpace(::DiscreteModel,args...;kwargs...)
  @notimplemented "Background model must be cartesian for the selected dof reordering"
end

function OrderedFESpace(::DiscreteModel,::DiscreteModel,args...;kwargs...)
  @notimplemented "Background model must be cartesian for the selected dof reordering"
end

struct CartesianFESpace{S<:SingleFieldFESpace} <: OrderedFESpace{S}
  space::S
  cell_odofs_ids::AbstractArray
  bg_odofs_to_act_odofs::AbstractVector
end

function CartesianFESpace(f::SingleFieldFESpace)
  cell_odofs_ids,bg_odofs_to_act_odofs = _get_cell_odof_info(f)
  CartesianFESpace(f,cell_odofs_ids,bg_odofs_to_act_odofs)
end

function CartesianFESpace(model::DiscreteModel,args...;kwargs...)
  f = FESpace(model,args...;kwargs...)
  CartesianFESpace(f)
end

function CartesianFESpace(trian::Triangulation,args...;kwargs...)
  OrderedFESpace(trian,args...;kwargs...)
end

FESpaces.get_fe_space(f::CartesianFESpace) = f.space

get_cell_odof_ids(f::CartesianFESpace) = f.cell_odofs_ids

get_bg_dof_to_act_dof(f::CartesianFESpace) = f.bg_odofs_to_act_odofs

# this works because the dofs are sorted lexicographically
get_act_dof_to_bg_dof(f::CartesianFESpace) = findall(!iszero,f.bg_odofs_to_act_odofs)

function get_sparsity(f::CartesianFESpace,g::CartesianFESpace,args...)
  sparsity = SparsityPattern(f,g,args...)
  CartesianSparsity(sparsity,get_bg_dof_to_act_dof(f),get_bg_dof_to_act_dof(g))
end

function get_sparsity(f::CartesianFESpace,g::SingleFieldFESpace,args...)
  sparsity = get_sparsity(get_fe_space(f),g,args...)
  CartesianSparsity(sparsity,get_bg_dof_to_act_dof(f),get_bg_dof_to_act_dof(g))
end

function get_sparsity(f::SingleFieldFESpace,g::CartesianFESpace,args...)
  sparsity = get_sparsity(f,get_fe_space(g),args...)
  CartesianSparsity(sparsity,get_bg_dof_to_act_dof(f),get_bg_dof_to_act_dof(g))
end

# dof reordering

function get_cell_odof_ids(space::SingleFieldFESpace)
  cell_odof_ids, = _get_cell_odof_info(space)
  return cell_odof_ids
end

# utils

function _get_cell_odof_info(space::SingleFieldFESpace)
  bg_dof_to_mask = get_bg_dof_to_mask(space)
  cell_dofs_ids = get_cell_dof_ids(space)
  cell_to_parent_cell = _get_bg_cell_to_act_cell(space)
  fe_dof_basis = get_data(get_fe_dof_basis(space))
  orders = get_polynomial_orders(space)
  trian = get_triangulation(space)
  model = get_background_model(trian)
  _get_cell_odof_info(model,fe_dof_basis,bg_dof_to_mask,cell_dofs_ids,cell_to_parent_cell,orders)
end

function _get_cell_odof_info(model::DiscreteModel,args...)
  @notimplemented "Background model must be cartesian the selected dof reordering"
end

function _get_cell_odof_info(
  model::CartesianDiscreteModel,
  fe_dof_basis::AbstractArray,
  bg_dof_to_mask::AbstractVector,
  cell_dofs_ids::AbstractArray,
  cell_to_parent_cell::AbstractVector,
  orders::Tuple)

  desc = get_cartesian_descriptor(model)
  periodic = desc.isperiodic
  ncells = desc.partition
  cells = CartesianIndices(ncells)
  pcells = view(cells,cell_to_parent_cell)
  onodes = LinearIndices(orders .* ncells .+ 1 .- periodic)

  node_and_comps_to_odof,bg_odofs_to_act_odofs = _get_odof_maps(
    fe_dof_basis,bg_dof_to_mask,cell_dofs_ids,pcells,onodes,orders)
  cell_odof_ids = lazy_map(DofsToODofs(fe_dof_basis,node_and_comps_to_odof,orders),pcells)
  return cell_odof_ids,bg_odofs_to_act_odofs
end

function _get_odof_maps(fe_basis::AbstractVector{<:Dof},args...)
  @notimplemented "This function is only implemented for Lagrangian dof bases"
end

function _get_odof_maps(fe_dof_basis::Fill{<:LagrangianDofBasis},args...)
  _get_odof_maps(testitem(fe_dof_basis),args...)
end

function _get_odof_maps(
  fe_dof_basis::LagrangianDofBasis{P,V},
  bg_dof_to_mask::AbstractVector,
  cell_dofs_ids::AbstractArray,
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
  odofs = zeros(eltype(V),ndofs)
  bg_odofs_to_act_odofs = collect(1:ndofs)
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
        bg_dof_to_mask[dof] && (bg_odofs_to_act_odofs[odof] = 0)
        odofs[odof] = sign(dof)
      end
    end
  end

  nfree = 0
  ndiri = 0
  for (i,odof) in enumerate(odofs)
    if odof > 0
      nfree += 1
      odofs[i] = nfree
    else
      ndiri -= 1
      odofs[i] = ndiri
    end
  end

  node_and_comps_to_odof = _get_node_and_comps_to_odof(fe_dof_basis,odofs,onodes)
  return node_and_comps_to_odof,bg_odofs_to_act_odofs
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

cubic_polytope(::Val{d}) where d = @abstractmethod
cubic_polytope(::Val{1}) = SEGMENT
cubic_polytope(::Val{2}) = QUAD
cubic_polytope(::Val{3}) = HEX

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
      model = get_active_model(trian)
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

function _get_bg_odof_to_act_odof(bg_f::CartesianFESpace,act_f::CartesianFESpace)
  bg_dof_to_bg_dof_to_act_dof = get_bg_dof_to_act_dof(bg_f) # potential underlying constraints
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
      if bg_dof > 0
        if !iszero(bg_dof_to_bg_dof_to_act_dof[bg_dof])
          bg_dof_to_act_dof[bg_dof] = act_dof
        end
      end
    end
  end
  return bg_dof_to_act_dof
end

function _get_bg_odof_to_act_odof(
  bg_f::SingleFieldFESpace,
  cell_odofs_ids::LazyArray{<:Fill{<:DofsToODofs}})

  bg_dofs_to_act_dofs = get_bg_dof_to_act_dof(bg_f)
  bg_odof_to_act_odof = collect(1:length(bg_dofs_to_act_dofs))
  k = testitem(cell_odofs_ids.maps)
  for (bg_dof,act_dof) in enumerate(bg_dofs_to_act_dofs)
    if iszero(act_dof)
      bg_odof_to_act_odof[get_odof(k,bg_dof)] = 0
    end
  end
  return bg_odof_to_act_odof
end

_get_parent_model(model::DiscreteModel) = @abstractmethod
_get_parent_model(model::MappedDiscreteModel) = model.model
_get_parent_model(model::DiscreteModelPortion) = model.parent_model
_get_parent_model(model::CartesianDiscreteModel) = model

function _constrained_dofs_locations(bg_space::SingleFieldFESpace)
  Int[],Int[]
end

function _constrained_dofs_locations(act_space::FESpaceWithLinearConstraints)
  bg_space = act_space.space
  _constrained_dofs_locations(bg_space,act_space)
end

function _constrained_dofs_locations(act_space::FESpaceWithConstantFixed)
  bg_space = f.space
  _constrained_dofs_locations(bg_space,act_space)
end

function _constrained_dofs_locations(act_space::ZeroMeanFESpace)
  _constrained_dofs_locations(act_space.space)
end

function _constrained_dofs_locations(fs::SingleFieldFESpace,cs::SingleFieldFESpace)
  @check ConstraintStyle(fs) == UnConstrained()

  fcellids = get_cell_dof_ids(fs)
  ccellids = get_cell_dof_ids(cs)
  fcache = array_cache(fcellids)
  ccache = array_cache(ccellids)
  cells = Int[]
  ldofs = Int[]

  for cell = 1:length(fcache)
    fdofs = getindex(fcache,fcellids,cell)
    cdofs = getindex(ccache,ccellids,cell)
    for (i,(fdof,cdof)) in enumerate(zip(fdofs,cdofs))
      if fdof > 0 && cdof < 0
        push!(cells,cell)
        push!(ldofs,i)
      end
    end
  end

  return cells,ldofs
end
