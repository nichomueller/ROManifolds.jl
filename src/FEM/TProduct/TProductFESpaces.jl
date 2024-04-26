function _minimum_dir_d(i::AbstractVector{CartesianIndex{D}},d::Integer) where D
  mind = Inf
  for ii in i
    if ii.I[d] < mind
      mind = ii.I[d]
    end
  end
  return mind
end

function _maximum_dir_d(i::AbstractVector{CartesianIndex{D}},d::Integer) where D
  maxd = 0
  for ii in i
    if ii.I[d] > maxd
      maxd = ii.I[d]
    end
  end
  return maxd
end

function _shape_per_dir(i::AbstractVector{CartesianIndex{D}}) where D
  function _admissible_shape(d::Int)
    mind = _minimum_dir_d(i,d)
    maxd = _maximum_dir_d(i,d)
    @assert all([ii.I[d] ≥ mind for ii in i]) && all([ii.I[d] ≤ maxd for ii in i])
    return maxd - mind + 1
  end
  ntuple(d -> _admissible_shape(d),D)
end

function _shape_per_dir(i::AbstractVector{<:Integer})
  min1 = minimum(i)
  max1 = maximum(i)
  (max1 - min1 + 1,)
end

function comp_to_free_dofs(::Type{T},space::FESpace,args...;kwargs...) where T
  @abstractmethod
end

function comp_to_free_dofs(::Type{T},space::UnconstrainedFESpace;kwargs...) where T
  glue = space.metadata
  ncomps = num_components(T)
  free_dof_to_comp = if isnothing(glue)
    _free_dof_to_comp(space,ncomps;kwargs...)
  else
    glue.free_dof_to_comp
  end
  comp_to_free_dofs(free_dof_to_comp,ncomps)
end

function _free_dof_to_comp(space,ncomps;kwargs...)
  @notimplemented
end

function comp_to_free_dofs(dof2comp,ncomps)
  comp2dof = Vector{typeof(dof2comp)}(undef,ncomps)
  for comp in 1:ncomps
    comp2dof[comp] = findall(dof2comp.==comp)
  end
  return comp2dof
end

function _get_cell_dof_comp_ids(cell_dof_ids,dofs)
  T = eltype(cell_dof_ids)
  ncells = length(cell_dof_ids)
  new_cell_ids = Vector{T}(undef,ncells)
  cache_cell_dof_ids = array_cache(cell_dof_ids)
  @inbounds for icell in 1:ncells
    cell_dofs = getindex!(cache_cell_dof_ids,cell_dof_ids,icell)
    ids_comp = findall(map(cd->cd ∈ dofs,abs.(cell_dofs)))
    new_cell_ids[icell] = cell_dofs[ids_comp]
  end
  return Table(new_cell_ids)
end

function _dof_perm_from_dof_perms(dof_perms::Vector{Matrix{Ti}}) where Ti
  @check all(size.(dof_perms) .== [size(first(dof_perms))])
  s = size(first(dof_perms))
  Dc = length(dof_perms)
  dof_perm = zeros(VectorValue{Dc,Ti},s)
  for ij in LinearIndices(s)
    perms_ij = getindex.(dof_perms,ij)
    dof_perm[ij] = Point(perms_ij)
  end
  return dof_perm
end

function get_dof_permutation(
  ::Type{T},
  model::CartesianDiscreteModel,
  space::UnconstrainedFESpace,
  order::Integer;
  kwargs...) where T

  cell_dof_ids = get_cell_dof_ids(space)
  _get_dof_permutation(model,cell_dof_ids,order)
end

function get_dof_permutation(
  ::Type{T},
  model::CartesianDiscreteModel,
  space::UnconstrainedFESpace,
  order::Integer;
  kwargs...) where T<:MultiValue

  cell_dof_ids = get_cell_dof_ids(space)
  comp2dofs = comp_to_free_dofs(T,space;kwargs...)
  Ti = eltype(eltype(cell_dof_ids))
  dof_perms = Matrix{Ti}[]
  for dofs in comp2dofs
    cell_dof_comp_ids = _get_cell_dof_comp_ids(cell_dof_ids,dofs)
    dof_perm_comp = _get_dof_permutation(model,cell_dof_comp_ids,order)
    push!(dof_perms,dof_perm_comp)
  end
  dof_perm = _dof_perm_from_dof_perms(dof_perms)
  return dof_perm
end

function _get_terms(p::Polytope,orders)
  _nodes, = Gridap.ReferenceFEs._compute_nodes(p,orders)
  terms = Gridap.ReferenceFEs._coords_to_terms(_nodes,orders)
  return terms
end

function _get_dof_permutation(
  model::CartesianDiscreteModel{Dc},
  cell_dof_ids::Table,
  order::Integer) where Dc

  desc = get_cartesian_descriptor(model)

  periodic = desc.isperiodic
  ncells = desc.partition
  ndofs = order .* ncells .+ 1 .- periodic

  new_dof_ids = copy(LinearIndices(ndofs))

  terms = _get_terms(first(get_polytopes(model)),fill(order,Dc))
  cache_cell_dof_ids = array_cache(cell_dof_ids)

  for (icell,cell) in enumerate(CartesianIndices(ncells))
    first_new_dof  = order .* (Tuple(cell) .- 1) .+ 1
    new_dofs_range = map(i -> i:i+order,first_new_dof)
    new_dofs = view(new_dof_ids,new_dofs_range...)

    cell_dofs = getindex!(cache_cell_dof_ids,cell_dof_ids,icell)
    for (idof,dof) in enumerate(cell_dofs)
      t = terms[idof]
      new_dofs[t] < 0 && continue
      if dof < 0
        new_dofs[t] *= -1
      end
    end
  end

  pos_ids = findall(new_dof_ids.>0)
  neg_ids = findall(new_dof_ids.<0)
  new_dof_ids[pos_ids] .= LinearIndices(pos_ids)
  new_dof_ids[neg_ids] .= -1 .* LinearIndices(neg_ids)

  free_vals_shape = _shape_per_dir(pos_ids)
  n2o_dof_map = fill(-1,free_vals_shape)

  for (icell,cell) in enumerate(CartesianIndices(ncells))
    first_new_dof  = order .* (Tuple(cell) .- 1) .+ 1
    new_dofs_range = map(i -> i:i+order,first_new_dof)
    new_dofs = view(new_dof_ids,new_dofs_range...)

    cell_dofs = getindex!(cache_cell_dof_ids,cell_dof_ids,icell)
    for (idof,dof) in enumerate(cell_dofs)
      t = terms[idof]
      new_dofs[t] < 0 && continue
      n2o_dof_map[new_dofs[t]] = dof
    end
  end

  return n2o_dof_map
end

function _scan_dimension_d(
  model::CartesianDiscreteModel{Dc},
  cell_dof_ids::Table,
  order::Integer,
  d::Integer) where Dc

  desc = get_cartesian_descriptor(model)
  periodic = desc.isperiodic
  ncells = desc.partition[d]
  ndofs = order * ncells + 1 - periodic

  new_dof_ids = copy(LinearIndices(ndofs))
end

function get_tp_connectivity(
  ::Type{T},
  model::CartesianDiscreteModel,
  space::FESpace,
  spaces::AbstractVector{<:FESpace},
  order::Integer) where T

  cell_dof_ids = get_cell_dof_ids(space)
  cell_dof_ids_1d = _setup_1d_connectivities(spaces)
  _get_tp_connectivity(cell_dof_ids,cell_dof_ids_1d)
end

# function get_tp_connectivity(
#   ::Type{T},
#   model::CartesianDiscreteModel,
#   spaces::AbstractVector{<:FESpace},
#   order::Integer) where T<:MultiValue


# end

function _setup_1d_connectivities(spaces)
  Dc = length(spaces)
  T = typeof(get_cell_dof_ids(first(spaces)))
  vcell_dof_ids = Vector{T}(undef,Dc)
  offset = 0
  for dim in 1:Dc
    space = spaces[dim]
    cell_dof_ids = collect(get_cell_dof_ids(space))
    map!(x->x.+offset,cell_dof_ids,cell_dof_ids)
    vcell_dof_ids[dim] = Table(cell_dof_ids)
    offset += length(cell_dof_ids)
  end
  return vcell_dof_ids
end

function _get_tp_connectivity(cell_dof_ids,cell_dof_ids_1d)
  new_cell_dof_ids = copy(cell_dof_ids)

  desc = get_cartesian_descriptor(model)
  ncells = desc.partition

  terms = get_terms(first(get_polytopes(model)),fill(order,Dc))
  cache_cell_dof_ids = array_cache(cell_dof_ids)
  new_cache_cell_dof_ids = array_cache(new_cell_dof_ids)
  caches_cell_dof_ids_1d = map(array_cache,cell_dof_ids_1d)

  @inbounds for (icell,cell) in enumerate(CartesianIndices(ncells))
    cell_dofs = getindex!(cache_cell_dof_ids,cell_dof_ids,icell)
    new_cell_dofs = getindex!(new_cache_cell_dof_ids,new_cell_dof_ids,icell)
    cell_dofs_1d = map(getindex!,caches_cell_dof_ids_1d,Tuple(cell))

    for idof in eachindex(cell_dofs)
      t = terms[idof]
      # new_cell_dofs[t] =
    end
  end
end

struct TProductFESpace{S,Ti} <: SingleFieldFESpace
  space::S
  dof_permutation::Matrix{Ti}
end

function TProductFESpace(
  model::CartesianDiscreteModel,
  reffe::Tuple{<:ReferenceFEName,Any,Any};
  kwargs...)

  basis,reffe_args,reffe_kwargs = reffe
  T,order = reffe_args
  cell_reffe = ReferenceFE(model,basis,T,order;reffe_kwargs...)
  space = FESpace(model,cell_reffe;kwargs...)
  get_dof_permutation(T,model,space,order;kwargs...)
end
