"""
    struct FEOperatorIndexMap{A,B}
      matrix_map::A
      vector_map::B
    end

Used to store the index maps related to jacobians and residuals in a FE problem

"""
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

"""
    get_vector_index_map(test::FESpace) -> AbstractIndexMap

Returns the index maps related to residuals in a FE problem. The default output
is a TrivialIndexMap; when the test space is of type TProductFESpace, a
nontrivial index map is returned

"""
function get_vector_index_map(test::FESpace)
  TrivialIndexMap(LinearIndices((num_free_dofs(test),)))
end

function get_vector_index_map(test::TProductFESpace)
  get_dof_index_map(test)
end

function get_vector_index_map(tests::MultiFieldFESpace)
  ntests = num_fields(tests)
  index_maps = Vector{AbstractIndexMap}(undef,ntests)
  for i in 1:ntests
    index_maps[i] = get_vector_index_map(tests[i])
  end
  return index_maps
end

"""
    get_matrix_index_map(trial::FESpace,test::FESpace) -> AbstractIndexMap

Returns the index maps related to jacobians in a FE problem. The default output
is a TrivialIndexMap; when the trial and test spaces are of type TProductFESpace,
a SparseIndexMap is returned

"""
function get_matrix_index_map(trial::FESpace,test::FESpace)
  sparsity = get_sparsity(trial,test)
  return TrivialIndexMap(get_nonzero_indices(sparsity))
end

function get_matrix_index_map(trial::TProductFESpace,test::TProductFESpace)
  sparsity = get_sparsity(trial,test)
  psparsity = permute_sparsity(sparsity,trial,test)
  I,J,_ = findnz(psparsity)
  i,j,_ = IndexMaps.univariate_findnz(psparsity)
  g2l = _global_2_local_nnz(psparsity,I,J,i,j)
  pg2l = permute_index_map(psparsity,g2l,trial,test)
  return SparseIndexMap(pg2l,psparsity)
end

for F in (:TrialFESpace,:TransientTrialFESpace)
  @eval begin
    get_matrix_index_map(trial::$F,tests::SingleFieldFESpace) = get_matrix_index_map(trial.space,tests)
  end
end

function get_matrix_index_map(trials::MultiFieldFESpace,tests::MultiFieldFESpace)
  ntests = num_fields(tests)
  ntrials = num_fields(trials)
  index_maps = Matrix{AbstractIndexMap}(undef,ntests,ntrials)
  for (i,j) in Iterators.product(1:ntests,1:ntrials)
    index_maps[i,j] = get_matrix_index_map(trials[j],tests[i])
  end
  return index_maps
end

# utils

function _global_2_local_nnz(sparsity::TProductSparsityPattern,I,J,i,j)
  IJ = get_nonzero_indices(sparsity)
  lids = map((ii,ji)->CartesianIndex.(ii,ji),i,j)

  unrows = IndexMaps.univariate_num_rows(sparsity)
  uncols = IndexMaps.univariate_num_cols(sparsity)
  unnz = IndexMaps.univariate_nnz(sparsity)
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

function permute_index_map(::TProductSparsityPattern,index_map,trial::TProductFESpace,test::TProductFESpace)
  I = get_dof_index_map(test)
  J = get_dof_index_map(trial)
  return _permute_index_map(index_map,I,J)
end

function permute_index_map(
  sparsity::TProductSparsityPattern{<:MultiValueSparsityPatternCSC},
  index_map,
  trial::TProductFESpace,
  test::TProductFESpace)

  function _to_component_indices(i,ncomps,icomp)
    nrows = Int(num_free_dofs(test)/ncomps)
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

  I = get_dof_index_map(test)
  J = get_dof_index_map(trial)
  I1 = get_component(I,1;multivalue=false)
  J1 = get_component(J,1;multivalue=false)
  indices = _permute_index_map(index_map,I1,J1)
  ncomps = num_components(sparsity)
  indices′ = map(icomp->_to_component_indices(indices,ncomps,icomp),1:ncomps)
  return MultiValueIndexMap(indices′)
end
