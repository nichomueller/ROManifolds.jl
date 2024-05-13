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

function _get_tp_dof_permutation(models::AbstractVector,spaces::AbstractVector,order::Integer)
  @assert length(models) == length(spaces)
  D = length(models)
  function _tensor_product(aprev::AbstractArray{Tp,M},a::AbstractVector{Td}) where {Tp,Td,M}
    T = promote_type(Tp,Td)
    N = M+1
    s = (size(aprev)...,length(a))
    atp = zeros(T,s)
    slicesN = eachslice(atp,dims=N)
    @inbounds for (iN,sliceN) in enumerate(slicesN)
      sliceN .= aprev .+ a[iN]
    end
    return atp
  end
  function _d_dof_permutation(::Val{1},::Val{d′}) where d′
    @assert d′ == D
    model_d = models[1]
    space_d = spaces[1]
    ndofs_d = num_free_dofs(space_d) + num_dirichlet_dofs(space_d)
    ndofs = ndofs_d
    cell_ids_d = get_cell_dof_ids(space_d)
    dof_permutations_1d = _get_dof_permutation(model_d,cell_ids_d,order)
    return _d_dof_permutation(dof_permutations_1d,ndofs,Val(2),Val(d′-1))
  end
  function _d_dof_permutation(node2dof_prev,ndofs_prev,::Val{d},::Val{d′}) where {d,d′}
    model_d = models[d]
    space_d = spaces[d]
    ndofs_d = num_free_dofs(space_d) + num_dirichlet_dofs(space_d)
    ndofs = ndofs_prev*ndofs_d
    cell_ids_d = get_cell_dof_ids(space_d)

    dof_permutations_1d = TProduct._get_dof_permutation(model_d,cell_ids_d,order)

    add_dim = ndofs_prev .* collect(0:ndofs_d)
    add_dim_reorder = add_dim[dof_permutations_1d]
    node2dof_d = _tensor_product(node2dof_prev,add_dim_reorder)

    _d_dof_permutation(node2dof_d,ndofs,Val(d+1),Val(d′-1))
  end
  function _d_dof_permutation(node2dof,ndofs,::Val{d},::Val{0}) where d
    @assert d == D+1
    return node2dof
  end
  return _d_dof_permutation(Val(1),Val(D))
end

function get_tp_dof_permutation(
  ::Type{T},
  models::AbstractVector,
  spaces::AbstractVector,
  order::Integer;
  kwargs...) where T

  _get_tp_dof_permutation(models,spaces,order)
end

function get_tp_dof_permutation(
  ::Type{T},
  models::AbstractVector,
  spaces::AbstractVector,
  order::Integer;
  kwargs...) where T<:MultiValue

  @notimplemented
end

# function _scan_dimension_d(
#   model::CartesianDiscreteModel{Dc},
#   cell_dof_ids::Table,
#   order::Integer,
#   d::Integer) where Dc

#   desc = get_cartesian_descriptor(model)
#   periodic = desc.isperiodic
#   ncells = desc.partition[d]
#   ndofs = order * ncells + 1 - periodic

#   new_dof_ids = copy(LinearIndices(ndofs))
# end

# function get_tp_connectivity(
#   ::Type{T},
#   model::CartesianDiscreteModel,
#   space::FESpace,
#   spaces::AbstractVector{<:FESpace},
#   order::Integer) where T

#   cell_dof_ids = get_cell_dof_ids(space)
#   cell_dof_ids_1d = _setup_1d_connectivities(spaces)
#   _get_tp_connectivity(cell_dof_ids,cell_dof_ids_1d)
# end

# function get_tp_connectivity(
#   ::Type{T},
#   model::CartesianDiscreteModel,
#   spaces::AbstractVector{<:FESpace},
#   order::Integer) where T<:MultiValue


# end

# function _setup_1d_connectivities(spaces)
#   Dc = length(spaces)
#   T = typeof(get_cell_dof_ids(first(spaces)))
#   vcell_dof_ids = Vector{T}(undef,Dc)
#   offset = 0
#   for dim in 1:Dc
#     space = spaces[dim]
#     cell_dof_ids = collect(get_cell_dof_ids(space))
#     map!(x->x.+offset,cell_dof_ids,cell_dof_ids)
#     vcell_dof_ids[dim] = Table(cell_dof_ids)
#     offset += length(cell_dof_ids)
#   end
#   return vcell_dof_ids
# end

# function _get_tp_connectivity(cell_dof_ids,cell_dof_ids_1d)
#   new_cell_dof_ids = copy(cell_dof_ids)

#   desc = get_cartesian_descriptor(model)
#   ncells = desc.partition

#   terms = get_terms(first(get_polytopes(model)),fill(order,Dc))
#   cache_cell_dof_ids = array_cache(cell_dof_ids)
#   new_cache_cell_dof_ids = array_cache(new_cell_dof_ids)
#   caches_cell_dof_ids_1d = map(array_cache,cell_dof_ids_1d)

#   @inbounds for (icell,cell) in enumerate(CartesianIndices(ncells))
#     cell_dofs = getindex!(cache_cell_dof_ids,cell_dof_ids,icell)
#     new_cell_dofs = getindex!(new_cache_cell_dof_ids,new_cell_dof_ids,icell)
#     cell_dofs_1d = map(getindex!,caches_cell_dof_ids_1d,Tuple(cell))

#     for idof in eachindex(cell_dofs)
#       t = terms[idof]
#       # new_cell_dofs[t] =
#     end
#   end
# end

struct TProductModel{D,A,B} <: DiscreteModel{D,D}
  model::A
  models_1d::B
  function TProductModel(
    model::A,
    models_1d::B
    ) where {D,A<:CartesianDiscreteModel{D},B<:AbstractVector{<:CartesianDiscreteModel}}
    new{D,A,B}(model,models_1d)
  end
end

Geometry.get_grid(model::TProductModel) = get_grid(model.model)
Geometry.get_grid_topology(model::TProductModel) = get_grid_topology(model.model)
Geometry.get_face_labeling(model::TProductModel) = get_face_labeling(model.model)

get_model(model::TProductModel) = model.model
get_1d_models(model::TProductModel) = model.models_1d

function _split_cartesian_descriptor(desc::CartesianDescriptor{D}) where D
  origin,sizes,partition,cmap,isperiodic = desc.origin,desc.sizes,desc.partition,desc.map,desc.isperiodic
  function _compute_1d_desc(
    o=first(origin.data),s=first(sizes),p=first(partition),m=cmap,i=first(isperiodic))
    CartesianDescriptor(Point(o),(s,),(p,);map=m,isperiodic=(i,))
  end
  isotropy = all([sizes[d] == sizes[1] && partition[d] == partition[1] for d = 1:D])
  factors = isotropy ? Fill(_compute_1d_desc(),D) : map(_compute_1d_desc,origin.data,sizes,partition,Fill(cmap,D),Fill(isperiodic,D))
  return factors
end

function TProductModel(args...;kwargs...)
  desc = CartesianDescriptor(args...;kwargs...)
  descs_1d = _split_cartesian_descriptor(desc)
  model = CartesianDiscreteModel(desc)
  models_1d = map(CartesianDiscreteModel,descs_1d)
  TProductModel(model,models_1d)
end

struct TProductFESpace{A,B,C} <: SingleFieldFESpace
  space::A
  spaces_1d::B
  dof_permutation::C
end

function TProductFESpace(
  model::TProductModel,
  reffe::Tuple{<:ReferenceFEName,Any,Any};
  kwargs...)

  basis,reffe_args,reffe_kwargs = reffe
  T,order = reffe_args
  cell_reffe = ReferenceFE(model.model,basis,T,order;reffe_kwargs...)
  cell_reffes_1d = map(model->ReferenceFE(model,basis,T,order;reffe_kwargs...),model.models_1d)
  space = FESpace(model,cell_reffe;kwargs...)
  spaces_1d = map(FESpace,model.models_1d,cell_reffes_1d) # is it ok to eliminate the kwargs?
  perm = get_tp_dof_permutation(T,model.models_1d,spaces_1d,order)
  TProductFESpace(space,spaces_1d,perm)
end

FESpaces.get_triangulation(f::TProductFESpace) = get_triangulation(f.space)

FESpaces.ConstraintStyle(::Type{<:TProductFESpace{A}}) where A = ConstraintStyle(A)

FESpaces.get_dirichlet_dof_values(f::TProductFESpace) = get_dirichlet_dof_values(f.space)

FESpaces.get_fe_basis(f::TProductFESpace) = get_fe_basis(f.space)

FESpaces.get_fe_dof_basis(f::TProductFESpace) = get_fe_dof_basis(f.space)

FESpaces.get_free_dof_ids(f::TProductFESpace) = get_free_dof_ids(f.space)

FESpaces.get_vector_type(f::TProductFESpace) = get_vector_type(f.space)

FESpaces.get_cell_dof_ids(f::TProductFESpace) = get_cell_dof_ids(f.space)

FESpaces.get_dirichlet_dof_ids(f::TProductFESpace) = get_dirichlet_dof_ids(f.space)

FESpaces.num_dirichlet_tags(f::TProductFESpace) = num_dirichlet_tags(f.space)

FESpaces.get_dirichlet_dof_tag(f::TProductFESpace) = get_dirichlet_dof_tag(f.space)

function FESpaces.scatter_free_and_dirichlet_values(f::TProductFESpace,fv,dv)
  scatter_free_and_dirichlet_values(f.space,fv,dv)
end

function FESpaces.gather_free_and_dirichlet_values!(fv,dv,f::TProductFESpace,cv)
  gather_free_and_dirichlet_values!(fv,dv,f.space,cv)
end
