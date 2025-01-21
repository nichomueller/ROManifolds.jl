struct OrderedFESpace{S<:SingleFieldFESpace} <: SingleFieldFESpace
  space::S
  cell_dofs_ids::AbstractArray
end

function OrderedFESpace(f::SingleFieldFESpace)
  cell_dofs_ids = get_ordered_cell_dof_ids(space)
  return OrderedFESpace(space,cell_dofs_ids)
end

# FESpace interface

FESpaces.ConstraintStyle(::Type{<:OrderedFESpace{S}}) where S = ConstraintStyle(S)

FESpaces.get_free_dof_ids(f::OrderedFESpace) = get_free_dof_ids(f.space)

FESpaces.get_triangulation(f::OrderedFESpace) = get_triangulation(f.space)

FESpaces.get_dof_value_type(f::OrderedFESpace) = get_dof_value_type(f.space)

FESpaces.get_cell_dof_ids(f::OrderedFESpace) = f.cell_dofs_ids

FESpaces.get_fe_basis(f::OrderedFESpace) = get_fe_basis(f.space)

FESpaces.get_trial_fe_basis(f::OrderedFESpace) = get_trial_fe_basis(f.space)

FESpaces.get_fe_dof_basis(f::OrderedFESpace) = get_fe_dof_basis(f.space)

FESpaces.get_cell_isconstrained(f::OrderedFESpace) = get_cell_isconstrained(f.space)

FESpaces.get_cell_constraints(f::OrderedFESpace) = get_cell_constraints(f.space)

FESpaces.get_dirichlet_dof_ids(f::OrderedFESpace) = get_dirichlet_dof_ids(f.space)

FESpaces.get_cell_is_dirichlet(f::OrderedFESpace) = get_cell_is_dirichlet(f.space)

FESpaces.num_dirichlet_dofs(f::OrderedFESpace) = num_dirichlet_dofs(f.space)

FESpaces.num_dirichlet_tags(f::OrderedFESpace) = num_dirichlet_tags(f.space)

FESpaces.get_dirichlet_dof_tag(f::OrderedFESpace) = get_dirichlet_dof_tag(f.space)

FESpaces.get_vector_type(f::OrderedFESpace) = get_vector_type(f.space)

function FESpaces.scatter_free_and_dirichlet_values(f::OrderedFESpace,fv,dv)
  scatter_free_and_dirichlet_values(f.space,fv,dv)
end

function FESpaces.gather_free_and_dirichlet_values(f::OrderedFESpace,cv)
  gather_free_and_dirichlet_values(remove_layer(f),cv)
end

function FESpaces.gather_free_and_dirichlet_values!(fv,dv,f::OrderedFESpace,cv)
  gather_free_and_dirichlet_values!(fv,dv,remove_layer(f),cv)
end

struct OIdsToIds{T,A<:AbstractVector{<:Integer}} <: AbstractVector{T}
  array::Vector{T}
  terms::A
end

Base.size(a::OIdsToIds) = size(a.array)
Base.getindex(a::OIdsToIds,i::Integer) = getindex(a.array,i)
Base.setindex!(a::OIdsToIds,v,i::Integer) = setindex!(a.array,v,i)

struct DofsToODofs{D,Ti,A} <: Map
  fe_dof_basis::A
  ordered_node_ids::Array{Ti,D}
  orders::NTuple{D,Int}
end

ReferenceFEs.get_polytope(k::DofsToODofs{D}) where D = @notimplemented
ReferenceFEs.get_polytope(k::DofsToODofs{1}) = SEGMENT
ReferenceFEs.get_polytope(k::DofsToODofs{2}) = QUAD
ReferenceFEs.get_polytope(k::DofsToODofs{3}) = HEX

function Arrays.return_cache(k::DofsToODofs{D},cell::CartesianIndex{D},icell::Int) where D
  ibasis = k.fe_dof_basis[icell]
  local_nnodes = length(ibasis.node_and_comp_to_dof)
  global_nfnodes = length(findall(k.ordered_node_ids .> 0))
  global_ndnodes = length(k.ordered_node_ids) - global_nfnodes
  pdof_to_dof = _local_pdof_to_dof(k)
  local_ndofs = length(ibasis)
  odofs = OIdsToIds(zeros(Int32,local_ndofs),pdof_to_dof)
  return odofs,ibasis,local_nnodes,global_nfnodes,global_ndnodes
end

function Arrays.evaluate!(cache,k::DofsToODofs{D},cell::CartesianIndex{D},icell::Int) where D
  odofs,ibasis,local_nnodes,global_nfnodes,global_ndnodes = cache
  first_new_nodes = k.orders .* (Tuple(cell) .- 1) .+ 1
  ordered_nodes_range = map(enumerate(first_new_nodes)) do (i,ni)
    ni:ni+k.orders[i]
  end
  ordered_nodes = view(k.ordered_node_ids,ordered_nodes_range...)
  for inode in ibasis.dof_to_node
    oinode = ordered_nodes[inode]
    for icomp in ibasis.dof_to_comp
      local_shift = (icomp-1)*local_nnodes
      if oinode > 0
        global_fshift = (icomp-1)*global_nfnodes
        odofs[inode+local_shift] = oinode + global_fshift
      else
        global_dshift = (icomp-1)*global_ndnodes
        odofs[inode+local_shift] = oinode - global_dshift
      end
    end
  end
  return odofs
end

# Assembly-related functions

@inline function Algebra.add_entries!(combine::Function,A,vs,is::OIdsToIds,js::OIdsToIds)
  add_ordered_entries!(combine,A,vs,is,js)
end

for T in (:Any,:(Algebra.ArrayCounter))
  @eval begin
    @inline function Algebra.add_entries!(combine::Function,A::$T,vs,is::OIdsToIds)
      add_ordered_entries!(combine,A,vs,is)
    end
  end
end

@inline function add_ordered_entries!(combine::Function,A,vs::Nothing,is::OIdsToIds,js::OIdsToIds)
  Algebra._add_entries!(combine,A,vs,is.array,js.array)
end

@inline function add_ordered_entries!(combine::Function,A,vs,is::OIdsToIds,js::OIdsToIds)
  for (lj,j) in enumerate(js)
    if j>0
      ljp = js.terms[lj]
      for (li,i) in enumerate(is)
        if i>0
          lip = is.terms[li]
          vij = vs[lip,ljp]
          add_entry!(combine,A,vij,i,j)
        end
      end
    end
  end
  A
end

@inline function add_ordered_entries!(combine::Function,A,vs::Nothing,is::OIdsToIds)
  Algebra._add_entries!(combine,A,vs,is.array)
end

@inline function add_ordered_entries!(combine::Function,A,vs,is::OIdsToIds)
  for (li,i) in enumerate(is)
    if i>0
      lip = is.terms[li]
      vi = vs[lip]
      add_entry!(A,vi,i)
    end
  end
  A
end

# utils

function get_ordered_cell_dof_ids(space::SingleFieldFESpace)
  cell_dof_ids = get_cell_dof_ids(space)
  fe_dof_basis = get_data(get_fe_dof_basis(space))
  orders = get_polynomial_orders(space)
  trian = get_triangulation(space)
  model = get_background_model(trian)
  tface_to_mface = get_tface_to_mface(trian)
  get_ordered_cell_dof_ids(model,cell_dof_ids,fe_dof_basis,tface_to_mface,orders)
end

function get_ordered_cell_dof_ids(
  model::CartesianDiscreteModel,
  cell_dof_ids::AbstractArray,
  fe_dof_basis::AbstractArray,
  tface_to_mface::AbstractVector,
  orders::Tuple)

  desc = get_cartesian_descriptor(model)
  periodic = desc.isperiodic
  ncells = desc.partition

  inodes = LinearIndices(orders .* ncells .+ 1 .- periodic)

  cells = vec(CartesianIndices(ncells))
  tcells = view(cells,tface_to_mface)
  ordered_node_ids = get_ordered_node_ids(cell_dof_ids,inodes)

  k = DofsToODofs(fe_dof_basis,ordered_node_ids,orders)
  lazy_map(k,tcells,tface_to_mface)
end

function get_ordered_cell_dof_ids(model::DiscreteModel,args...)
  @notimplemented "Background model must be cartesian for dof reordering"
end

function get_ordered_dof_ids(cell_dofs_ids::AbstractArray,s::Dims)
  onodes = zeros(Int32,s)
  cache = array_cache(cell_dofs_ids)
  nfree = 0
  ndiri = 0
  for cell in 1:length(cell_dofs_ids)
    dofs = getindex!(cache,cell_dofs_ids,cell)
    for dof in dofs
      comp = slow_index(dof,ncomps)
      node = fast_index(dof,ncomps)
      if !touched[dof]
        touched[dof] = true
        ionodes += 1
        if dof > 0
          nfree += 1
          onodes[ionodes] = nfree
        else
          ndiri += 1
          onodes[ionodes] = -ndiri
        end
      end
    end
  end

  for i in CartesianIndices(onodes)
    if _is_free_index(i,fv_ranges)
      nfree += 1
      onodes[i] = nfree
    else
      ndiri += 1
      onodes[i] = -ndiri
    end
  end
  return onodes
end

function _local_pdof_to_dof(k::DofsToODofs)
  ibasis = testitem(k.fe_dof_basis)
  nnodes = length(ibasis.node_and_comp_to_dof)
  ndofs = length(ibasis.dof_to_comp)

  p = get_polytope(k)
  _nodes, = Gridap.ReferenceFEs._compute_nodes(p,k.orders)
  node_to_pnode = Gridap.ReferenceFEs._coords_to_terms(_nodes,k.orders)
  node_to_pnode_linear = LinearIndices(k.orders.+1)[node_to_pnode]

  pdof_to_dof = zeros(Int32,ndofs)
  for (inode,ipnode) in enumerate(node_to_pnode_linear)
    for icomp in ibasis.dof_to_comp
      local_shift = (icomp-1)*nnodes
      pdof_to_dof[local_shift+ipnode] = local_shift + inode
    end
  end

  return pdof_to_dof
end
