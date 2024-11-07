function IndexMaps.get_vector_index_map(test::TProductFESpace,t::Triangulation...)
  if length(test.spaces_1d) == 1 # in the 1-D case, we return a trivial map
    get_vector_index_map(test.space,t...)
  else
    get_dof_index_map(test,t...)
  end
end

function IndexMaps.get_matrix_index_map(trial::TProductFESpace,test::TProductFESpace,t::Triangulation...)
  if length(trial.spaces_1d) == length(test.spaces_1d) == 1 # in the 1-D case, we return a trivial map
    get_matrix_index_map(trial.space,test.space,t...)
  else
    sparsity = get_sparsity(trial,test,t...)
    psparsity = permute_sparsity(sparsity,trial,test)
    I,J,_ = findnz(psparsity)
    i,j,_ = IndexMaps.univariate_findnz(psparsity)
    g2l_sparse = _global_2_local(psparsity,I,J,i,j)
    pg2l_sparse = _permute_index_map(g2l_sparse,trial,test)
    pg2l = to_nz_index(pg2l_sparse,sparsity)
    SparseIndexMap(pg2l,pg2l_sparse,psparsity)
  end
end

# utils

function _global_2_local(sparsity::TProductSparsityPattern,I,J,i,j)
  IJ = get_nonzero_indices(sparsity)
  lids = map((ii,ji)->CartesianIndex.(ii,ji),i,j)

  unrows = IndexMaps.univariate_num_rows(sparsity)
  uncols = IndexMaps.univariate_num_cols(sparsity)
  unnz = IndexMaps.univariate_nnz(sparsity)

  tprows = CartesianIndices(unrows)
  tpcols = CartesianIndices(uncols)

  g2l = zeros(eltype(IJ),unnz...)
  lid = zeros(Int,length(lids))
  @inbounds for (k,gid) = enumerate(IJ)
    irows = tprows[I[k]]
    icols = tpcols[J[k]]
    @inbounds for d in eachindex(lids)
      lidd = lids[d]
      indd = CartesianIndex((irows.I[d],icols.I[d]))
      @inbounds for (l,liddl) in enumerate(lidd)
        if liddl == indd
          lid[d] = l
          break
        end
      end
    end
    g2l[lid...] = gid
  end

  return g2l
end

function _permute_index_map(index_map,I,J,nrows)
  IJ = vectorize_map(I) .+ nrows .* (vectorize_map(J)'.-1)
  iperm = copy(index_map)
  @inbounds for (k,pk) in enumerate(index_map)
    if pk > 0
      iperm[k] = IJ[pk]
    end
  end
  return IndexMap(iperm)
end

function _permute_index_map(index_map,I::AbstractMultiValueIndexMap,J::AbstractMultiValueIndexMap,nrows)
  function _to_component_indices(i,ncomps,icomp_I,icomp_J,nrows)
    ic = copy(i)
    @inbounds for (j,IJ) in enumerate(i)
      IJ == 0 && continue
      I = fast_index(IJ,nrows)
      J = slow_index(IJ,nrows)
      I′ = (I-1)*ncomps + icomp_I
      J′ = (J-1)*ncomps + icomp_J
      ic[j] = (J′-1)*nrows*ncomps + I′
    end
    return IndexMap(ic)
  end

  ncomps_I = num_components(I)
  ncomps_J = num_components(J)
  @check ncomps_I == ncomps_J
  ncomps = ncomps_I
  nrows_per_comp = Int(nrows/ncomps)

  I1 = get_component(I,1;multivalue=false)
  J1 = get_component(J,1;multivalue=false)
  index_map′ = _permute_index_map(index_map,I1,J1,nrows_per_comp)

  index_map′′ = map(CartesianIndices((ncomps,ncomps))) do icomp
    _to_component_indices(index_map′,ncomps,icomp.I[1],icomp.I[2],nrows_per_comp)
  end
  return MultiValueIndexMap(index_map′′)
end

function _permute_index_map(index_map,I::AbstractMultiValueIndexMap,J::AbstractIndexMap,nrows)
  function _to_component_indices(i,ncomps,icomp,nrows)
    ic = copy(i)
    @inbounds for (j,IJ) in enumerate(i)
      IJ == 0 && continue
      I = fast_index(IJ,nrows)
      J = slow_index(IJ,nrows)
      I′ = (I-1)*ncomps + icomp
      ic[j] = (J-1)*nrows*ncomps + I′
    end
    return IndexMap(ic)
  end

  ncomps = num_components(I)
  nrows_per_comp = Int(nrows/ncomps)

  I1 = get_component(I,1;multivalue=false)
  index_map′ = _permute_index_map(index_map,I1,J,nrows_per_comp)

  index_map′′ = map(icomp->_to_component_indices(index_map′,ncomps,icomp,nrows_per_comp),1:ncomps)
  return MultiValueIndexMap(index_map′′)
end

function _permute_index_map(index_map,I::AbstractIndexMap,J::AbstractMultiValueIndexMap,nrows)
  function _to_component_indices(i,ncomps,icomp,nrows)
    ic = copy(i)
    @inbounds for (j,IJ) in enumerate(i)
      IJ == 0 && continue
      I = fast_index(IJ,nrows)
      J = slow_index(IJ,nrows)
      J′ = (J-1)*ncomps + icomp
      ic[j] = (J′-1)*nrows + I
    end
    return IndexMap(ic)
  end

  ncomps = num_components(J)

  J1 = get_component(J,1;multivalue=false)
  index_map′ = _permute_index_map(index_map,I,J1,nrows)
  index_map′′ = map(icomp->_to_component_indices(index_map′,ncomps,icomp,nrows),1:ncomps)
  return MultiValueIndexMap(index_map′′)
end

function _permute_index_map(index_map,trial::TProductFESpace,test::TProductFESpace)
  I = get_dof_index_map(test)
  J = get_dof_index_map(trial)
  nrows = num_free_dofs(test)
  return _permute_index_map(index_map,I,J,nrows)
end
