struct FEOperatorIndexMap{A,B}
  matrix_map::A
  vector_map::B
end

get_matrix_index_map(i::FEOperatorIndexMap) = i.matrix_map
get_vector_index_map(i::FEOperatorIndexMap) = i.vector_map

function FEOperatorIndexMap(trial::FESpace,test::FESpace)
  trial_model = get_background_model(get_triangulation(trial))
  test_model = get_background_model(get_triangulation(test))
  @notimplementedif trial_model != test_model

  vector_map = get_vector_index_map(test)
  matrix_map = get_matrix_index_map(trial,test)
  FEOperatorIndexMap(matrix_map,vector_map)
end

function get_vector_index_map(test::FESpace)
  TrivialIndexMap((Int32(num_free_dofs(test)),))
end

function get_vector_index_map(test::TProductFESpace)
  get_dof_index_map(test)
end

function get_vector_index_map(tests::MultiFieldFESpace)
  index_maps = AbstractIndexMap[]
  for test in tests
    push!(index_maps,get_vector_index_map(test))
  end
  return index_maps
end

function get_matrix_index_map(trial::FESpace,test::FESpace)
  sparsity = get_sparsity(trial,test)
  i = Int32.(get_nonzero_indices(sparsity))
  return IndexMap(i)
end

function get_matrix_index_map(U::TProductFESpace,V::TProductFESpace)
  sparsity = get_sparsity(U,V)
  psparsity = permute_sparsity(sparsity,U,V)
  I,J,_ = findnz(psparsity)
  i,j,_ = univariate_findnz(psparsity)
  g2l = global_2_local_nnz(psparsity,I,J,i,j)
  pg2l = permute_index_map(psparsity,g2l,U,V)
  return SparseIndexMap(pg2l,psparsity)
end

for F in (:TrialFESpace,:TransientTrialFESpace)
  @eval begin
    get_matrix_index_map(trial::$F,tests::SingleFieldFESpace) = get_matrix_index_map(trial.space,tests)
  end
end

function get_matrix_index_map(trials::MultiFieldFESpace,tests::MultiFieldFESpace)
  index_maps = AbstractIndexMap[]
  for (trial,test) in zip(trials,tests)
    push!(index_maps,get_matrix_index_map(trial,test))
  end
  return index_maps
end

# utils

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
  sparsity::TProductSparsityPattern{<:MultiValueSparsityPatternCSC},
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
