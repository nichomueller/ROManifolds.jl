function FESpaces.collect_cell_matrix(
  trial::FESpace,
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation)

  scell_mat = get_contribution(a,strian)
  cell_mat,trian = move_contributions(scell_mat,strian)
  @assert ndims(eltype(cell_mat)) == 2
  cell_mat_c = attach_constraints_cols(trial,cell_mat,trian)
  cell_mat_rc = attach_constraints_rows(test,cell_mat_c,trian)
  rows = get_cell_dof_ids(test,trian)
  cols = get_cell_dof_ids(trial,trian)
  return cell_mat_rc,rows,cols
end

function FESpaces.collect_cell_vector(
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation)

  scell_vec = get_contribution(a,strian)
  cell_vec,trian = move_contributions(scell_vec,strian)
  @assert ndims(eltype(cell_vec)) == 1
  cell_vec_r = attach_constraints_rows(test,cell_vec,trian)
  rows = get_cell_dof_ids(test,trian)
  return cell_vec_r,rows
end

function param_transient_numeric_loop_matrix!(A,a::GenericSparseMatrixAssembler,matdata)
  for (all_cellmat,_cellidsrows,_cellidscols) in zip(matdata...)
    cellidsrows = map_cell_rows(a.strategy,_cellidsrows)
    cellidscols = map_cell_cols(a.strategy,_cellidscols)
    @assert length(cellidscols) == length(cellidsrows)
    if length(cellidsrows) > 0
      all_vals_cache = array_cache(all_cellmat)
      rows_cache = array_cache(cellidsrows)
      cols_cache = array_cache(cellidscols)
      rows1 = getindex!(rows_cache,cellidsrows,1)
      cols1 = getindex!(cols_cache,cellidscols,1)
      add! = AddEntriesMap(+)
      for n = 1:N
        An = A[n]
        cellmatn = getindex!(all_vals_cache,all_cellmat,n)
        @assert length(cellmat) == length(cellidsrows)
        vals_cache = array_cache(cellmatn)
        mat1 = getindex!(vals_cache,cellmat,1)
        add_cache = return_cache(add!,A,mat1,rows1,cols1)
        caches = add_cache, vals_cache, rows_cache, cols_cache
        _numeric_loop_matrix!(An,caches,cellmat,cellidsrows,cellidscols)
        copyto!(view(A,n),An)
      end
    end
  end
  A
end

function param_transient_numeric_loop_vector!(b,a::GenericSparseMatrixAssembler,vecdata)
  for (all_cellvec, _cellids) in zip(vecdata...)
    cellids = map_cell_rows(a.strategy,_cellids)
    if length(cellids) > 0
      all_vals_cache = array_cache(all_cellvec)
      rows_cache = array_cache(cellids)
      rows1 = getindex!(rows_cache,cellids,1)
      for k = eachindex(all_cellvec)
        bk = b[k]
        cellveck = getindex!(all_vals_cache,all_cellvec,k)
        vals_cache = array_cache(cellveck)
        vals1 = getindex!(vals_cache,cellveck,1)
        add! = AddEntriesMap(+)
        add_cache = return_cache(add!,b,vals1,rows1)
        caches = add_cache, vals_cache, rows_cache
        _numeric_loop_vector!(bk,caches,cellvec,cellids)
        copyto!(view(b,k),bk)
      end
    end
  end
  b
end

function param_transient_numeric_loop_matrix!(A,a::GenericSparseMatrixAssembler,matdata)
  for (all_cellmat,_cellidsrows,_cellidscols) in zip(matdata...)
    cellidsrows = map_cell_rows(a.strategy,_cellidsrows)
    cellidscols = map_cell_cols(a.strategy,_cellidscols)
    @assert length(cellidscols) == length(cellidsrows)
    if length(cellidsrows) > 0
      all_vals_cache = array_cache(all_cellmat)
      rows_cache = array_cache(cellidsrows)
      cols_cache = array_cache(cellidscols)
      rows1 = getindex!(rows_cache,cellidsrows,1)
      cols1 = getindex!(cols_cache,cellidscols,1)
      add! = AddEntriesMap(+)
      for k = eachindex(all_cellmat)
        Ak = A[k]
        cellmatk = getindex!(all_vals_cache,all_cellmat,k)
        @assert length(cellmat) == length(cellidsrows)
        vals_cache = array_cache(cellmatk)
        mat1 = getindex!(vals_cache,cellmatk,1)
        add_cache = return_cache(add!,A,mat1,rows1,cols1)
        caches = add_cache, vals_cache, rows_cache, cols_cache
        _numeric_loop_matrix!(Ak,caches,cellmat,cellidsrows,cellidscols)
        copyto!(view(A,k),Ak)
      end
    end
  end
  A
end
