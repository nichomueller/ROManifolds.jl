"""
    get_tp_dof_dof_map(space::FESpace,spaces_1d::AbstractVector{<:FESpace}) -> AbstractDofMap

Returns the dofs of the FE space defined as the tensor product of `spaces_1d`,
sorted by coordinate order, for every dimension. The variable `space` represents
the FESpace defined on the TProductModel with a standard Gridap procedure.
The role of `space` is limited to providing the algorithm with the correct dof type
and polynomial order. If `space` is a D-dimensional FESpace, the vector `spaces_1d`
is of length D, and the output index map will be a subtype of AbstractDofMap{D}.

The following example clarifies the function's task:

cell_dof_ids_1 = Table([[1, 3, 2]])
cell_dof_ids_2 = Table([[1, 4, 2, 5, 3]])

  get_tp_dof_dof_map(⋅)
        ⟹ ⟹ ⟹

      [ 1  7  4  10  7
        3  9  6  11  9
        2  8  5  12  8 ]

"""
function get_tp_dof_dof_map(space::FESpace,spaces_1d::AbstractVector{<:FESpace})
  @abstractmethod
end

function get_tp_dof_dof_map(space::UnconstrainedFESpace,spaces_1d::AbstractVector{<:UnconstrainedFESpace})
  trians_1d = map(get_triangulation,spaces_1d)
  models_1d = map(get_background_model,trians_1d)
  dof_map,dof_maps_1d = get_tp_dof_dof_map(space,models_1d,spaces_1d)
  return TProductDofMap(dof_map,dof_maps_1d)
end

function get_tp_dof_dof_map(
  space::UnconstrainedFESpace,
  models_1d::AbstractVector{<:CartesianDiscreteModel},
  spaces_1d::AbstractVector{<:UnconstrainedFESpace})

  dof = get_fe_dof_basis(space)
  T = get_dof_type(dof)
  order = get_polynomial_order(space)
  get_tp_dof_dof_map(T,models_1d,spaces_1d,order)
end

function get_tp_dof_dof_map(
  ::Type{T},
  models::AbstractVector,
  spaces::AbstractVector,
  order::Integer
  ) where T

  _get_tp_dof_dof_map(models,spaces,order)
end

function get_tp_dof_dof_map(
  ::Type{T},
  models::AbstractVector,
  spaces::AbstractVector,
  order::Integer
  ) where T<:MultiValue

  ncomp = num_components(T)
  dof_map,dof_maps_1d = get_tp_dof_dof_map(eltype(T),models,spaces,order)
  ncomp_dof_map = repeat(dof_map;outer=(ntuple(_->1,ncomps)...,ncomps))
  return ncomp_dof_map,dof_maps_1d
end

# this function computes only the free dofs tensor product permutation
function _get_tp_dof_dof_map(models::AbstractVector,spaces::AbstractVector,order::Integer)
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
    trian_ids = 1:num_cells(model)
    dof_maps_1d = _get_dof_dof_map(model,cell_ids,trian_ids,order)
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

# spaces with constraints

function get_tp_dof_dof_map(ls::FESpaceWithLinearConstraints,spaces_1d::AbstractVector{<:UnconstrainedFESpace})
  space = ls.space
  mdof_to_bdof = ls.mDOF_to_DOF
  sdof_to_bdof = setdiff(1:ls.n_fdofs,mdof_to_bdof)
  i = get_tp_dof_dof_map(space,spaces_1d)
  ci = ConstrainedDofsDofMap(i,mdof_to_bdof,sdof_to_bdof)
  return TProductDofMap(ci,i.indices_1d)
end

function get_tp_dof_dof_map(cs::FESpaceWithConstantFixed,spaces_1d::AbstractVector{<:UnconstrainedFESpace})
  space = cs.space
  ndofs = num_free_dofs(space) + num_dirichlet_dofs(space)
  sdof_to_bdof = cs.dof_to_fix
  mdof_to_bdof = setdiff(1:ndofs,sdof_to_bdof)
  i = get_tp_dof_dof_map(space,spaces_1d)
  ci = ConstrainedDofsDofMap(i,mdof_to_bdof,sdof_to_bdof)
  return TProductDofMap(ci,i.indices_1d)
end

function get_tp_dof_dof_map(zs::ZeroMeanFESpace,spaces_1d::AbstractVector{<:UnconstrainedFESpace})
  space = zs.space
  get_tp_dof_dof_map(space,spaces_1d)
end
