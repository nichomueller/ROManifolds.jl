function get_tp_dof_index_map(space::FESpace,spaces_1d::AbstractVector{<:FESpace})
  @abstractmethod
end

function get_tp_dof_index_map(space::UnconstrainedFESpace,spaces_1d::AbstractVector{<:UnconstrainedFESpace})
  trians_1d = map(get_triangulation,spaces_1d)
  models_1d = map(get_background_model,trians_1d)
  index_map,index_maps_1d = get_tp_dof_index_map(space,models_1d,spaces_1d)
  return TProductIndexMap(index_map,index_maps_1d)
end

function get_tp_dof_index_map(
  space::UnconstrainedFESpace,
  models_1d::AbstractVector{<:CartesianDiscreteModel},
  spaces_1d::AbstractVector{<:UnconstrainedFESpace})

  dof = get_fe_dof_basis(space)
  T = get_dof_type(dof)
  order = get_polynomial_order(space)
  get_tp_dof_index_map(T,models_1d,spaces_1d,order)
end

function get_tp_dof_index_map(
  ::Type{T},
  models::AbstractVector,
  spaces::AbstractVector,
  order::Integer
  ) where T

  _get_tp_dof_index_map(models,spaces,order)
end

function get_tp_dof_index_map(
  ::Type{T},
  models::AbstractVector,
  spaces::AbstractVector,
  order::Integer
  ) where T<:MultiValue

  ncomp = num_components(T)
  dof_map,dof_maps_1d = get_tp_dof_index_map(eltype(T),models,spaces,order)
  ncomp_dof_map = compose_indices(dof_map,ncomp)
  return ncomp_dof_map,dof_maps_1d
end

# this function computes only the free dofs tensor product permutation
function _get_tp_dof_index_map(models::AbstractVector,spaces::AbstractVector,order::Integer)
  @assert length(models) == length(spaces)
  D = length(models)

  function _tensor_product(aprev::AbstractArray{T,M},a::AbstractVector) where {T,M}
    N = M+1
    s = (size(aprev)...,length(a))
    atp = zeros(T,s)
    slicesN = eachslice(atp,dims=N)
    @inbounds for (iN,sliceN) in enumerate(slicesN)
      sliceN .= aprev .+ a[iN]
    end
    return atp
  end
  function _local_dof_map(model,space)
    cell_ids = get_cell_dof_ids(space)
    dof_maps_1d = _get_dof_map(model,cell_ids,order)
    free_dof_maps_1d = dof_maps_1d[findall(dof_maps_1d.>0)]
    return free_dof_maps_1d
  end

  free_dof_maps_1d = map(_local_dof_map,models,spaces)

  function _d_dof_map(::Val{1},::Val{d′}) where d′
    @assert d′ == D
    space_d = spaces[1]
    ndofs_d = num_free_dofs(space_d)
    ndofs = ndofs_d
    free_dof_map_1d = free_dof_maps_1d[1]
    return _d_dof_map(free_dof_map_1d,ndofs,Val(2),Val(d′-1))
  end
  function _d_dof_map(node2dof_prev,ndofs_prev,::Val{d},::Val{d′}) where {d,d′}
    space_d = spaces[d]
    ndofs_d = num_free_dofs(space_d)
    ndofs = ndofs_prev*ndofs_d
    free_dof_map_1d = free_dof_maps_1d[d]

    add_dim = ndofs_prev .* collect(0:ndofs_d)
    add_dim_reorder = add_dim[free_dof_map_1d]
    node2dof_d = _tensor_product(node2dof_prev,add_dim_reorder)

    _d_dof_map(node2dof_d,ndofs,Val(d+1),Val(d′-1))
  end
  function _d_dof_map(node2dof,ndofs,::Val{d},::Val{0}) where d
    @assert d == D+1
    return node2dof
  end
  return _d_dof_map(Val(1),Val(D)),free_dof_maps_1d
end

# zeromean

function get_tp_dof_index_map(zs::ZeroMeanFESpace,spaces_1d::AbstractVector{<:UnconstrainedFESpace})
  space = zs.space.space
  dof_to_fix = zs.space.dof_to_fix
  index_map = get_tp_dof_index_map(space,spaces_1d)
  @warn "the code has changed, check if this is ok"
  return FixedDofIndexMap(index_map,findfirst(vec(index_map).==dof_to_fix))
end

# multi field

function get_tp_dof_index_map(ms::MultiFieldFESpace,spaces_1d::AbstractVector{<:UnconstrainedFESpace})
  index_maps = TProductIndexMap[]
  for space in ms
    push!(index_maps,get_tp_dof_index_map(space,spaces_1d))
  end
  return index_maps
end

# trial

for F in (:TrialFESpace,:TransientTrialFESpace)
  @eval begin
    function get_tp_dof_index_map(f::$F,spaces_1d::AbstractVector{<:UnconstrainedFESpace})
      get_tp_dof_index_map(f.space,spaces_1d)
    end
  end
end
