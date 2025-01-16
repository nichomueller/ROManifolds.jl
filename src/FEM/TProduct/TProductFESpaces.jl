"""
    TProductFESpace{A,B,C,D,E,F} <: SingleFieldFESpace

Tensor product single field FE space, storing a vector of 1-D FE spaces `spaces_1d`
of length D, and the D-dimensional FE space `space` defined as their tensor product.
The tensor product triangulation `trian` is provided as a field to avoid
incompatibility issues when passing to MultiField scenarios
"""
struct TProductFESpace{A,B,C,D,E} <: SingleFieldFESpace
  space::A
  spaces_1d::B
  trian::C
  dof_map::D
  tp_dof_map::E
end

function TProductFESpace(
  trian::TProductTriangulation,
  reffe::Tuple{<:ReferenceFEName,Any,Any};
  kwargs...)

  basis,reffe_args,reffe_kwargs = reffe
  T,order = reffe_args

  model = get_background_model(trian)

  cell_reffe = ReferenceFE(model.model,basis,T,order;reffe_kwargs...)
  space = FESpace(trian.trian,cell_reffe;kwargs...)

  cell_reffes_1d = map(model->ReferenceFE(model,basis,eltype(T),order;reffe_kwargs...),model.models_1d)
  spaces_1d = univariate_spaces(model,trian,cell_reffes_1d;kwargs...)

  diri_entities = get_dirichlet_entities(spaces_1d)
  dof_map = get_dof_map(model.model,space,diri_entities)
  tp_dof_map = get_tp_dof_map(space,spaces_1d)

  TProductFESpace(space,spaces_1d,trian,dof_map,tp_dof_map)
end

# The FE space is defined on a triangulation that isn't tensor-product, but which
# admits a background tensor-product triangulation
function TProductFESpace(
  trian::Triangulation,
  tptrian::TProductTriangulation,
  reffe::Tuple{<:ReferenceFEName,Any,Any};
  kwargs...)

  @check Utils.is_included(trian,tptrian.trian)

  tpspace = TProductFESpace(tptrian,reffe;kwargs...)

  basis,reffe_args,reffe_kwargs = reffe
  T,order = reffe_args

  model = get_active_model(trian)

  cell_reffe = ReferenceFE(model,basis,T,order;reffe_kwargs...)
  space = FESpace(trian,cell_reffe;kwargs...)

  diri_entities = get_dirichlet_entities(tpspace.spaces_1d)
  dof_map = get_dof_map(model,space,tpspace,diri_entities)

  TProductFESpace(space,tpspace.spaces_1d,tptrian,dof_map,tpspace.tp_dof_map)
end

function univariate_spaces(
  model::TProductModel,
  trian::TProductTriangulation,
  cell_reffes;
  dirichlet_tags=Int[],
  conformity=nothing,
  vector_type=nothing,
  kwargs...)

  add_1d_tags!(model,dirichlet_tags)
  map((trian,cell_reffe) -> FESpace(trian,cell_reffe;dirichlet_tags,conformity,vector_type),
    trian.trians_1d,cell_reffes)
end

FESpaces.get_triangulation(f::TProductFESpace) = get_triangulation(f.space)

FESpaces.get_free_dof_ids(f::TProductFESpace) = get_free_dof_ids(f.space)

FESpaces.get_vector_type(f::TProductFESpace) = get_vector_type(f.space)

FESpaces.get_dof_value_type(f::TProductFESpace) = get_dof_value_type(f.space)

FESpaces.get_cell_dof_ids(f::TProductFESpace) = get_cell_dof_ids(f.space)

FESpaces.ConstraintStyle(::Type{<:TProductFESpace{A}}) where A = ConstraintStyle(A)

FESpaces.get_fe_basis(f::TProductFESpace) = get_fe_basis(f.space)

FESpaces.get_fe_dof_basis(f::TProductFESpace) = get_fe_dof_basis(f.space)

FESpaces.num_dirichlet_dofs(f::TProductFESpace) = num_dirichlet_dofs(f.space)

FESpaces.get_cell_isconstrained(f::TProductFESpace) = get_cell_isconstrained(f.space)

FESpaces.get_cell_constraints(f::TProductFESpace) = get_cell_constraints(f.space)

FESpaces.get_dirichlet_dof_ids(f::TProductFESpace) = get_dirichlet_dof_ids(f.space)

FESpaces.get_cell_is_dirichlet(f::TProductFESpace) = get_cell_is_dirichlet(f.space)

FESpaces.num_dirichlet_tags(f::TProductFESpace) = num_dirichlet_tags(f.space)

FESpaces.get_dirichlet_dof_tag(f::TProductFESpace) = get_dirichlet_dof_tag(f.space)

FESpaces.scatter_free_and_dirichlet_values(f::TProductFESpace,fv,dv) = scatter_free_and_dirichlet_values(f.space,fv,dv)

DofMaps.get_dof_map(f::TProductFESpace) = f.dof_map

DofMaps.get_univariate_dof_map(f::TProductFESpace) = get_univariate_dof_map(f.tp_dof_map)

get_tp_dof_map(f::TProductFESpace) = f.tp_dof_map

get_tp_triangulation(f::TProductFESpace) = f.trian

"""
"""
function get_tp_fe_basis(f::TProductFESpace)
  basis = map(get_fe_basis,f.spaces_1d)
  trian = get_tp_triangulation(f)
  TProductFEBasis(basis,trian)
end

"""
"""
function get_tp_trial_fe_basis(f::TProductFESpace)
  basis = map(get_trial_fe_basis,f.spaces_1d)
  trian = get_tp_triangulation(f)
  TProductFEBasis(basis,trian)
end

get_tp_trial_fe_basis(f::TrialFESpace{<:TProductFESpace}) = get_tp_trial_fe_basis(f.space)

function DofMaps.get_dirichlet_entities(f::TProductFESpace)
  get_dirichlet_entities(f.spaces_1d)
end

function DofMaps.get_dirichlet_entities(spaces::Vector{<:FESpace})
  D = length(spaces)
  isdirichlet = zeros(Bool,D,2)
  for (d,space) in enumerate(spaces)
    celldiri = get_cell_is_dirichlet(space)
    isdirichlet[d,1] = first(celldiri)
    isdirichlet[d,2] = last(celldiri)
  end
  return isdirichlet
end

function DofMaps.SparsityPattern(U::TProductFESpace,V::TProductFESpace)
  sparsity = SparsityPattern(U.space,V.space)
  sparsities_1d = map(SparsityPattern,U.spaces_1d,V.spaces_1d)
  return TProductSparsity(sparsity,sparsities_1d)
end

function DofMaps.SparsityPattern(
  U::TProductFESpace{A,B,C,<:DofMapPortion},
  V::TProductFESpace{A,B,C,<:DofMapPortion}
  ) where {A,B,C}

  sparsity = SparsityPattern(U.space,V.space)
  sparsities_1d = map(SparsityPattern,U.spaces_1d,V.spaces_1d)
  dof_to_parent_row = DofMaps.get_dof_to_parent_dof_map(V.dof_map)
  dof_to_parent_col = DofMaps.get_dof_to_parent_dof_map(U.dof_map)
  return SparsityToTProductSparsity(sparsity,sparsities_1d,dof_to_parent_row,dof_to_parent_col)
end

function DofMaps.order_sparsity(s::TProductSparsityPattern,U::TProductFESpace,V::TProductFESpace)
  dof_map_I = get_dof_map(V)
  dof_map_J = get_dof_map(U)
  dof_map_I_1d = get_tp_dof_map(V).indices_1d
  dof_map_J_1d = get_tp_dof_map(U).indices_1d
  order_sparsity(s,(dof_map_I,dof_map_I_1d),(dof_map_J,dof_map_J_1d))
end

function DofMaps.get_dof_map(
  model::DiscreteModelPortion,
  space::FESpace,
  tpspace::TProductFESpace,
  diri_entities::AbstractMatrix)

  parent_model = Geometry.get_parent_model(model)
  dof_map = get_dof_map(parent_model,space,diri_entities)
  parent_dof_map = get_dof_map(tpspace)
  DofMapPortion(dof_map,parent_dof_map)
end

# multi field

_remove_trial(f::SingleFieldFESpace) = _remove_trial(f.space)
_remove_trial(f::TProductFESpace) = f

function get_tp_triangulation(f::MultiFieldFESpace)
  s1 = _remove_trial(first(f.spaces))
  trian = get_tp_triangulation(s1)
  @check all(map(i->trian===get_tp_triangulation(_remove_trial(i)),f.spaces))
  trian
end

function get_tp_fe_basis(f::MultiFieldFESpace)
  D = length(_remove_trial(f[1]).spaces_1d)
  basis = map(1:D) do d
    sfd = map(sf -> _remove_trial(sf).spaces_1d[d],f.spaces)
    mfd = MultiFieldFESpace(sfd)
    get_fe_basis(mfd)
  end
  trian = get_tp_triangulation(f)
  TProductFEBasis(basis,trian)
end

function get_tp_trial_fe_basis(f::MultiFieldFESpace)
  D = length(_remove_trial(f[1]).spaces_1d)
  basis = map(1:D) do d
    sfd = map(sf -> _remove_trial(sf).spaces_1d[d],f.spaces)
    mfd = MultiFieldFESpace(sfd)
    get_trial_fe_basis(mfd)
  end
  trian = get_tp_triangulation(f)
  TProductFEBasis(basis,trian)
end
