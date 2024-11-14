function DofMaps.get_vector_dof_map(test::TProductFESpace,t::Triangulation...)
  if length(test.spaces_1d) == 1 # in the 1-D case, we return a trivial map
    get_vector_dof_map(test.space,t...)
  else
    get_ordered_dof_map(test,t...)
  end
end

function DofMaps.get_matrix_dof_map(trial::TProductFESpace,test::TProductFESpace,t::Triangulation...)
  if length(trial.spaces_1d) == length(test.spaces_1d) == 1 # in the 1-D case, we return a trivial map
    get_matrix_dof_map(trial.space,test.space,t...)
  else
    sparsity = SparsityPattern(trial,test,t...)
    psparsity = order_sparsity(sparsity,trial,test)
    I,J,_ = findnz(psparsity)
    i,j,_ = DofMaps.univariate_findnz(psparsity)
    g2l_sparse = _global_2_local(psparsity,I,J,i,j)
    pg2l_sparse = _permute_dof_map(g2l_sparse,trial,test)
    pg2l = to_nz_index(pg2l_sparse,sparsity)
    SparseDofMap(pg2l,pg2l_sparse,psparsity)
  end
end

# utils

function _global_2_local(sparsity::TProductSparsityPattern,I,J,i,j)
  IJ = get_nonzero_indices(sparsity)
  lids = map((ii,ji)->CartesianIndex.(ii,ji),i,j)

  unrows = DofMaps.univariate_num_rows(sparsity)
  uncols = DofMaps.univariate_num_cols(sparsity)
  unnz = DofMaps.univariate_nnz(sparsity)

  tprows = CartesianIndices(unrows)
  tpcols = CartesianIndices(uncols)

  g2l = zeros(eltype(IJ),unnz...)
  lid = zeros(Int,length(lids))
  @inbounds for (k,gid) = enumerate(IJ)
    fill!(lid,0)
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

  return DofMap(g2l)
end

function _permute_dof_map(dof_map,I,J,nrows)
  IJ = vectorize_map(I) .+ nrows .* (vectorize_map(J)'.-1)
  iperm = copy(dof_map)
  @inbounds for (k,pk) in enumerate(dof_map)
    if pk > 0
      iperm[k] = IJ[pk]
    end
  end
  return iperm
end

function _permute_dof_map(dof_map,I::AbstractMultiValueDofMap,J::AbstractMultiValueDofMap,nrows)
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
    return ic
  end

  ncomps_I = num_components(I)
  ncomps_J = num_components(J)
  @check ncomps_I == ncomps_J
  ncomps = ncomps_I
  nrows_per_comp = Int(nrows/ncomps)

  I1 = get_component(I,1;multivalue=false)
  J1 = get_component(J,1;multivalue=false)
  dof_map′ = _permute_dof_map(dof_map,I1,J1,nrows_per_comp)

  dof_map′′ = map(CartesianIndices((ncomps,ncomps))) do icomp
    _to_component_indices(dof_map′,ncomps,icomp.I[1],icomp.I[2],nrows_per_comp)
  end
  return MultiValueDofMap(dof_map′′)
end

function _permute_dof_map(dof_map,I::AbstractMultiValueDofMap,J::AbstractDofMap,nrows)
  function _to_component_indices(i,ncomps,icomp,nrows)
    ic = copy(i)
    @inbounds for (j,IJ) in enumerate(i)
      IJ == 0 && continue
      I = fast_index(IJ,nrows)
      J = slow_index(IJ,nrows)
      I′ = (I-1)*ncomps + icomp
      ic[j] = (J-1)*nrows*ncomps + I′
    end
    return ic
  end

  ncomps = num_components(I)
  nrows_per_comp = Int(nrows/ncomps)

  I1 = get_component(I,1;multivalue=false)
  dof_map′ = _permute_dof_map(dof_map,I1,J,nrows_per_comp)

  dof_map′′ = map(icomp->_to_component_indices(dof_map′,ncomps,icomp,nrows_per_comp),1:ncomps)
  return MultiValueDofMap(dof_map′′)
end

function _permute_dof_map(dof_map,I::AbstractDofMap,J::AbstractMultiValueDofMap,nrows)
  function _to_component_indices(i,ncomps,icomp,nrows)
    ic = copy(i)
    @inbounds for (j,IJ) in enumerate(i)
      IJ == 0 && continue
      I = fast_index(IJ,nrows)
      J = slow_index(IJ,nrows)
      J′ = (J-1)*ncomps + icomp
      ic[j] = (J′-1)*nrows + I
    end
    return ic
  end

  ncomps = num_components(J)

  J1 = get_component(J,1;multivalue=false)
  dof_map′ = _permute_dof_map(dof_map,I,J1,nrows)
  dof_map′′ = map(icomp->_to_component_indices(dof_map′,ncomps,icomp,nrows),1:ncomps)
  return MultiValueDofMap(dof_map′′)
end

function _permute_dof_map(dof_map,trial::TProductFESpace,test::TProductFESpace)
  I = get_ordered_dof_map(test)
  J = get_ordered_dof_map(trial)
  nrows = num_free_dofs(test)
  return _permute_dof_map(dof_map,I,J,nrows)
end
