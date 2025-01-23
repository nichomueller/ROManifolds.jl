abstract type OrderedFESpace{S} <: SingleFieldFESpace end

# FESpace interface

get_ordered_dof_ids(f::OrderedFESpace) = @abstractmethod

FESpaces.get_fe_space(f::OrderedFESpace) = @abstractmethod

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

function get_dof_map(trial::OrderedFESpace)
  get_dof_map(get_fe_space(f))
end

function get_sparse_dof_map(f::OrderedFESpace,test::SingleFieldFESpace,args...)
  get_sparse_dof_map(get_fe_space(f),test,args...)
end

# Constructors

function OrderedFESpace(args...;kwargs...)
  f = FESpace(args...;kwargs...)
  OrderedFESpace(f)
end

function OrderedFESpace(trian::Grid,args...;kwargs...)
  act_model = get_active_model(trian)
  bg_model = get_background_model(trian)
  OrderedFESpace(act_model,bg_model,args...;kwargs...)
end

function OrderedFESpace(
  act_model::CartesianDiscreteModel,
  bg_model::CartesianDiscreteModel,
  args...;kwargs...)

  f = FESpace(act_model,args...;kwargs...)
  CartesianFESpace(f)
end

function OrderedFESpace(
  act_model::DiscreteModel,
  bg_model::CartesianDiscreteModel,
  args...;kwargs...)

  f_bg = OrderedFESpace(bg_model,args...;kwargs...)
  f_act = FESpace(act_model,args...;kwargs...)
  node_map = d_act_dof_to_bg_odof(f_act,f_bg)
  GenericOrderedFESpace(f_act,node_map)
end

function OrderedFESpace(::DiscreteModel,::DiscreteModel,args...;kwargs...)
  @notimplemented "Background model must be cartesian for dof reordering. If the
  Triangulation on which the FE space is defined "
end

struct CartesianFESpace{S<:SingleFieldFESpace} <: SingleFieldFESpace
  space::S
  ordered_cell_dofs_ids::AbstractArray
end

function OrderedFESpace(f::SingleFieldFESpace)
  ordered_cell_dofs_ids = get_ordered_cell_dof_ids(f)
  CartesianFESpace(f,ordered_cell_dofs_ids)
end

FESpaces.get_fe_space(f::OrderedFESpace) = f.space

get_ordered_cell_dof_ids(f::OrderedFESpace) = f.ordered_cell_dofs_ids

# dof reordering

function get_ordered_cell_dof_ids(space::SingleFieldFESpace)
  cell_dofs_ids = get_cell_dof_ids(space)
  fe_dof_basis = get_data(get_fe_dof_basis(space))
  orders = get_polynomial_orders(space)
  trian = get_triangulation(space)
  model = get_background_model(trian)
  tface_to_mface = get_tface_to_mface(trian)
  get_ordered_cell_dof_ids(model,cell_dofs_ids,fe_dof_basis,tface_to_mface,orders)
end

function get_ordered_cell_dof_ids(model::DiscreteModel,args...)
  @notimplemented "Background model must be cartesian for dof reordering"
end

function get_ordered_cell_dof_ids(
  model::CartesianDiscreteModel,
  cell_dofs_ids::AbstractArray,
  fe_dof_basis::AbstractArray,
  tface_to_mface::AbstractVector,
  orders::Tuple)

  desc = get_cartesian_descriptor(model)
  periodic = desc.isperiodic
  ncells = desc.partition
  cells = CartesianIndices(ncells)
  tcells = view(cells,tface_to_mface)
  onodes = LinearIndices(orders .* ncells .+ 1 .- periodic)

  node_and_comps_to_odof = get_ordered_dof_ids(cell_dofs_ids,fe_dof_basis,cells,onodes,orders)
  k = DofsToODofs(fe_dof_basis,node_and_comps_to_odof,orders)
  lazy_map(k,tcells)
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
  cells::CartesianIndices{D},
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
