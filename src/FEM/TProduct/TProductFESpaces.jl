"""
    TProductFESpace{A,B,C} <: SingleFieldFESpace

Tensor product single field FE space, storing a vector of 1-D FE spaces `spaces_1d`
of length D, and the D-dimensional FE space `space` defined as their tensor product.
The tensor product triangulation `trian` is provided as a field to avoid
incompatibility issues when passing to MultiField scenarios

"""
struct TProductFESpace{A,B,C} <: SingleFieldFESpace
  space::A
  spaces_1d::B
  trian::C
end

function FESpaces.FESpace(
  trian::TProductTriangulation,
  reffe::Tuple{<:ReferenceFEName,Any,Any};
  kwargs...)

  basis,reffe_args,reffe_kwargs = reffe
  T,order = reffe_args

  model = get_background_model(trian)
  cell_reffe = ReferenceFE(model.model,basis,T,order;reffe_kwargs...)
  cell_reffes_1d = map(model->ReferenceFE(model,basis,eltype(T),order;reffe_kwargs...),model.models_1d)

  space = FESpace(model.model,cell_reffe;kwargs...)
  spaces_1d = univariate_spaces(model,cell_reffes_1d;kwargs...)
  TProductFESpace(space,spaces_1d,trian)
end

function univariate_spaces(
  model::TProductModel,
  cell_reffes;
  dirichlet_tags=Int[],
  conformity=nothing,
  vector_type=nothing,
  kwargs...)

  add_1d_tags!(model,dirichlet_tags)
  map((model,cell_reffe) -> FESpace(model,cell_reffe;dirichlet_tags,conformity,vector_type),
    model.models_1d,cell_reffes)
end

get_space(f::TProductFESpace) = f.space

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

get_dof_index_map(f::TProductFESpace) = get_dof_index_map(f.space)

get_tp_dof_index_map(f::TProductFESpace) = get_tp_dof_index_map(f.space,f.spaces_1d)

get_tp_triangulation(f::TProductFESpace) = f.trian

function get_tp_fe_basis(f::TProductFESpace)
  basis = map(get_fe_basis,f.spaces_1d)
  trian = get_tp_triangulation(f)
  TProductFEBasis(basis,trian)
end

function get_tp_trial_fe_basis(f::TProductFESpace)
  basis = map(get_trial_fe_basis,f.spaces_1d)
  trian = get_tp_triangulation(f)
  TProductFEBasis(basis,trian)
end

function IndexMaps.get_sparsity(U::TProductFESpace,V::TProductFESpace)
  a = SparseMatrixAssembler(U,V)
  sparsity = get_sparsity(a.assem,U.space,V.space)
  sparsities_1d = map(get_sparsity,a.assems_1d,U.spaces_1d,V.spaces_1d)
  return TProductSparsityPattern(sparsity,sparsities_1d)
end

function IndexMaps.permute_sparsity(s::TProductSparsityPattern,U::TProductFESpace,V::TProductFESpace)
  index_map_I = get_dof_index_map(V)
  index_map_J = get_dof_index_map(U)
  index_map_I_1d = get_tp_dof_index_map(V).indices_1d
  index_map_J_1d = get_tp_dof_index_map(U).indices_1d
  permute_sparsity(s,(index_map_I,index_map_I_1d),(index_map_J,index_map_J_1d))
end

for F in (:TrialFESpace,:TransientTrialFESpace)
  @eval begin
    function get_matrix_index_map(U::$F{<:TProductFESpace},V::TProductFESpace)
      get_matrix_index_map(U.space,V)
    end
  end
end

# multi field

get_tp_trial_fe_basis(f::TrialFESpace{<:TProductFESpace}) = get_tp_trial_fe_basis(f.space)

function get_tp_triangulation(f::MultiFieldFESpace)
  s1 = first(f.spaces)
  trian = get_tp_triangulation(s1)
  @check all(map(i->trian===get_tp_triangulation(i),f.spaces))
  trian
end

function get_tp_fe_basis(f::MultiFieldFESpace)
  D = length(f.spaces[1].spaces_1d)
  basis = map(1:D) do d
    sfd = map(sf -> sf.spaces_1d[d],f.spaces)
    mfd = MultiFieldFESpace(sfd)
    get_fe_basis(mfd)
  end
  trian = get_tp_triangulation(f)
  TProductFEBasis(basis,trian)
end

function get_tp_trial_fe_basis(f::MultiFieldFESpace)
  D = length(f.spaces[1].spaces_1d)
  basis = map(1:D) do d
    sfd = map(sf -> sf.spaces_1d[d],f.spaces)
    mfd = MultiFieldFESpace(sfd)
    get_trial_fe_basis(mfd)
  end
  trian = get_tp_triangulation(f)
  TProductFEBasis(basis,trian)
end
