struct TProductFESpace{A,B} <: SingleFieldFESpace
  space::A
  spaces_1d::B
end

function FESpaces.FESpace(
  model::TProductModel,
  reffe::Tuple{<:ReferenceFEName,Any,Any};
  kwargs...)

  basis,reffe_args,reffe_kwargs = reffe
  T,order = reffe_args
  cell_reffe = ReferenceFE(model.model,basis,T,order;reffe_kwargs...)
  cell_reffes_1d = map(model->ReferenceFE(model,basis,eltype(T),order;reffe_kwargs...),model.models_1d)
  space = FESpace(model.model,cell_reffe;kwargs...)
  spaces_1d = univariate_spaces(model,cell_reffes_1d;kwargs...)
  TProductFESpace(space,spaces_1d)
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

FESpaces.get_vector_type(f::TProductFESpace) = f.vector_type

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

ParamFESpaces.get_dirichlet_cells(f::TProductFESpace) = get_dirichlet_cells(f.space)

ParamFESpaces.get_dof_index_map(f::TProductFESpace) = get_dof_index_map(f.space)

get_tp_dof_index_map(f::TProductFESpace) = get_tp_dof_index_map(f.space,f.spaces_1d)

# need to correct parametric, tproduct, zeromean constrained fespaces
function FESpaces.FEFunction(
  f::FESpaceToParamFESpace{<:TProductFESpace},
  free_values::AbstractTProductArray,
  dirichlet_values::AbstractTProductArray)

  tf = f.space
  f′ = FESpaceToParamFESpace(tf.space,param_length(f))
  FEFunction(f′,free_values,dirichlet_values)
end

function FESpaces.EvaluationFunction(
  f::FESpaceToParamFESpace{<:TProductFESpace},
  free_values::AbstractTProductArray)

  tf = f.space
  f′ = FESpaceToParamFESpace(tf.space,param_length(f))
  EvaluationFunction(f′,free_values)
end

function FESpaces.scatter_free_and_dirichlet_values(
  f::FESpaceToParamFESpace{<:TProductFESpace},
  fv::AbstractTProductArray,
  dv::AbstractTProductArray)

  tf = f.space
  f′ = FESpaceToParamFESpace(tf.space,param_length(f))
  scatter_free_and_dirichlet_values(f′,fv,dv)
end

function get_tp_triangulation(f::TProductFESpace)
  trian = get_triangulation(f.space)
  trians_1d = map(get_triangulation,f.spaces_1d)
  TProductTriangulation(trian,trians_1d)
end

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

get_tp_trial_fe_basis(f::TrialFESpace{<:TProductFESpace}) = get_tp_trial_fe_basis(f.space)

function ParamODEs.assemble_norm_matrix(f,U::FESpace,V::FESpace)
  assemble_matrix(f,U,V)
end

function ParamODEs.assemble_norm_matrix(f,U::TProductFESpace,V::TProductFESpace)
  a = SparseMatrixAssembler(U,V)
  v = get_tp_fe_basis(V)
  u = get_tp_trial_fe_basis(U)
  assemble_matrix(a,collect_cell_matrix(U,V,f(u,v)))
end

function ParamODEs.assemble_norm_matrix(f,U::TrialFESpace{<:TProductFESpace},V::TProductFESpace)
  assemble_norm_matrix(f,U.space,V)
end

function IndexMap.get_sparsity(U::TProductFESpace,V::TProductFESpace)
  a = SparseMatrixAssembler(U,V)
  sparsity = get_sparsity(a.assem,U.space,V.space)
  sparsities_1d = map(get_sparsity,a.assems_1d,U.spaces_1d,V.spaces_1d)
  return TProductSparsityPattern(sparsity,sparsities_1d)
end

function IndexMap.permute_sparsity(s::TProductSparsityPattern,U::TProductFESpace,V::TProductFESpace)
  index_map_I = get_dof_index_map(V)
  index_map_J = get_dof_index_map(U)
  index_map_I_1d = get_tp_dof_index_map(V).indices_1d
  index_map_J_1d = get_tp_dof_index_map(U).indices_1d
  permute_sparsity(s,(index_map_I,index_map_I_1d),(index_map_J,index_map_J_1d))
end

for F in (:TrialFESpace,:TransientTrialFESpace)
  @eval begin
    function IndexMap.get_sparsity(U::$F{<:TProductFESpace},V::TProductFESpace)
      get_sparsity(U.space,V)
    end

    function get_sparse_index_map(U::$F{<:TProductFESpace},V::TProductFESpace)
      get_sparse_index_map(U.space,V)
    end
  end
end

function get_sparse_index_map(U::TProductFESpace,V::TProductFESpace)
  sparsity = get_sparsity(U,V)
  psparsity = permute_sparsity(sparsity,U,V)
  I,J,_ = findnz(psparsity)
  i,j,_ = univariate_findnz(psparsity)
  g2l = global_2_local_nnz(psparsity,I,J,i,j)
  pg2l = permute_index_map(psparsity,g2l,U,V)
  return SparseIndexMap(pg2l,psparsity)
end

function global_2_local_nnz(sparsity::TProductSparsityPattern,I,J,i,j)
  IJ = get_nonzero_indices(sparsity)
  lids = map((ii,ji)->CartesianIndex.(ii,ji),i,j)

  unrows = univariate_num_rows(sparsity)
  uncols = univariate_num_cols(sparsity)
  unnz = univariate_nnz(sparsity)
  g2l = zeros(eltype(IJ),unnz...)

  @inbounds for (k,gid) = enumerate(IJ)
    irows = Tuple(tensorize_indices(I[k],unrows))
    icols = Tuple(tensorize_indices(J[k],uncols))
    iaxes = CartesianIndex.(irows,icols)
    lid = map((i,j) -> findfirst(i.==[j]),lids,iaxes)
    g2l[lid...] = gid
  end

  return g2l
end

function _permute_index_map(index_map,I,J)
  nrows = length(I)
  IJ = vec(I) .+ nrows .* (vec(J)'.-1)
  iperm = copy(index_map)
  @inbounds for (k,pk) in enumerate(index_map)
    iperm[k] = IJ[pk]
  end
  return IndexMap(iperm)
end

function permute_index_map(::TProductSparsityPattern,index_map,U::TProductFESpace,V::TProductFESpace)
  I = get_dof_index_map(V)
  J = get_dof_index_map(U)
  return _permute_index_map(index_map,I,J)
end

function permute_index_map(
  sparsity::TProductSparsityPattern{<:MultiValuePatternCSC},
  index_map,U::TProductFESpace,V::TProductFESpace)

  function _to_component_indices(i,ncomps,icomp)
    nrows = Int(num_free_dofs(V)/ncomps)
    ic = copy(i)
    @inbounds for (j,IJ) in enumerate(i)
      I = fast_index(IJ,nrows)
      J = slow_index(IJ,nrows)
      I′ = (I-1)*ncomps + icomp
      J′ = (J-1)*ncomps + icomp
      ic[j] = (J′-1)*nrows*ncomps + I′
    end
    return ic
  end

  I = get_dof_index_map(V)
  J = get_dof_index_map(U)
  I1 = get_component(I,1;multivalue=false)
  J1 = get_component(J,1;multivalue=false)
  indices = _permute_index_map(index_map,I1,J1)
  ncomps = num_components(sparsity)
  indices′ = map(icomp->_to_component_indices(indices,ncomps,icomp),1:ncomps)
  return MultiValueIndexMap(indices′)
end
