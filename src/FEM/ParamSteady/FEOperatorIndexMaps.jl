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
  if length(test.spaces_1d) == 1 # in the 1-D case, we return a trivial map
    get_vector_index_map(test.space)
  else
    get_dof_index_map(test)
  end
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
  TrivialIndexMap(LinearIndices((nnz(sparsity),)))
end

function get_matrix_index_map(trial::TProductFESpace,test::TProductFESpace)
  if length(trial.spaces_1d) == length(test.spaces_1d) == 1 # in the 1-D case, we return a trivial map
    get_matrix_index_map(trial.space,test.space)
  else
    sparsity = get_sparsity(trial,test)
    psparsity = permute_sparsity(sparsity,trial,test)
    I,J,_ = findnz(psparsity)
    i,j,_ = IndexMaps.univariate_findnz(psparsity)
    g2l = _global_2_local_nnz(psparsity,I,J,i,j)
    pg2l = _permute_index_map(g2l,trial,test)
    SparseIndexMap(pg2l,psparsity)
  end
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

function _add_fixed_dofs(index_map::AbstractIndexMap)
  if any(index_map.==0)
    return FixedDofsIndexMap(index_map,findall(index_map.==zero(eltype(index_map))))
  end
  return index_map
end

function _add_fixed_dofs(index_map::AbstractMatrix)
  _add_fixed_dofs(IndexMap(index_map))
end

function _permute_index_map(index_map,I,J,nrows)
  IJ = vec(I) .+ nrows .* (vec(J)'.-1)
  iperm = copy(index_map)
  @inbounds for (k,pk) in enumerate(index_map)
    if pk > 0
      iperm[k] = IJ[pk]
    end
  end
  return _add_fixed_dofs(iperm)
end

function _permute_index_map(index_map,I::FixedDofsIndexMap,J,nrows)
  _permute_index_map(index_map,remove_fixed_dof(I),J,nrows)
end

function _permute_index_map(index_map,I,J::FixedDofsIndexMap,nrows)
  _permute_index_map(index_map,I,remove_fixed_dof(J),nrows)
end

function _permute_index_map(index_map,I::FixedDofsIndexMap,J::FixedDofsIndexMap,nrows)
  _permute_index_map(index_map,remove_fixed_dof(I),remove_fixed_dof(J),nrows)
end

function _permute_index_map(index_map,I::AbstractMultiValueIndexMap,J::AbstractMultiValueIndexMap,nrows)
  function _to_component_indices(i,ncomps,icomp,nrows)
    ic = copy(i)
    @inbounds for (j,IJ) in enumerate(i)
      IJ == 0 && continue
      I = fast_index(IJ,nrows)
      J = slow_index(IJ,nrows)
      I′ = (I-1)*ncomps + icomp
      J′ = (J-1)*ncomps + icomp
      ic[j] = (J′-1)*nrows*ncomps + I′
    end
    return _add_fixed_dofs(ic)
  end

  ncomps_I = num_components(I)
  ncomps_J = num_components(J)
  @check ncomps_I == ncomps_J
  ncomps = ncomps_I
  nrows_per_comp = Int(nrows/ncomps)

  I1 = get_component(I,1;multivalue=false)
  J1 = get_component(J,1;multivalue=false)
  index_map′ = _permute_index_map(index_map,I1,J1,nrows_per_comp)

  index_map′′ = map(icomp->_to_component_indices(index_map′,ncomps,icomp,nrows_per_comp),1:ncomps)
  return MultiValueIndexMap(index_map′′)
end

for T in (:AbstractIndexMap,:FixedDofsIndexMap)
  @eval begin
    function _permute_index_map(index_map,I::AbstractMultiValueIndexMap,J::$T,nrows)
      function _to_component_indices(i,ncomps,icomp,nrows)
        ic = copy(i)
        @inbounds for (j,IJ) in enumerate(i)
          IJ == 0 && continue
          I = fast_index(IJ,nrows)
          J = slow_index(IJ,nrows)
          I′ = (I-1)*ncomps + icomp
          ic[j] = (J-1)*nrows*ncomps + I′
        end
        return _add_fixed_dofs(ic)
      end

      ncomps = num_components(I)
      nrows_per_comp = Int(nrows/ncomps)

      I1 = get_component(I,1;multivalue=false)
      index_map′ = _permute_index_map(index_map,I1,J,nrows_per_comp)

      index_map′′ = map(icomp->_to_component_indices(index_map′,ncomps,icomp,nrows_per_comp),1:ncomps)
      return MultiValueIndexMap(index_map′′)
    end

    function _permute_index_map(index_map,I::$T,J::AbstractMultiValueIndexMap,nrows)
      function _to_component_indices(i,ncomps,icomp,nrows)
        ic = copy(i)
        @inbounds for (j,IJ) in enumerate(i)
          IJ == 0 && continue
          I = fast_index(IJ,nrows)
          J = slow_index(IJ,nrows)
          J′ = (J-1)*ncomps + icomp
          ic[j] = (J′-1)*nrows + I
        end
        return _add_fixed_dofs(ic)
      end

      ncomps = num_components(J)

      J1 = get_component(J,1;multivalue=false)
      index_map′ = _permute_index_map(index_map,I,J1,nrows)
      index_map′′ = map(icomp->_to_component_indices(index_map′,ncomps,icomp,nrows),1:ncomps)
      return MultiValueIndexMap(index_map′′)
    end
  end
end

function _permute_index_map(index_map,trial::TProductFESpace,test::TProductFESpace)
  I = get_dof_index_map(test)
  J = get_dof_index_map(trial)
  nrows = num_free_dofs(test)
  return _permute_index_map(index_map,I,J,nrows)
end
