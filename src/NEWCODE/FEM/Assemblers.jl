function generate_fe_snapshots(feop,solver,nsnap)
  sols = solve(solver,feop,nsnap)
  cache,nonzero_idx = snapshot_cache(solver,feop,nsnap)
  snaps = pmap(sol->get_solution!(cache,sol),sols)
  mat_snaps = EMatrix(first.(snaps))
  param_snaps = Table(last.(snaps))
  Snapshots(mat_snaps,param_snaps,nonzero_idx)
end

function assemble_vector(a::Assembler,vecdata)
  vecs = Float[]
  for data in vecdata
    b = allocate_vector(a,data)
    assemble_vector!(b,a,vecdata)
    push!(vecs,b)
  end
  vecs
end

function allocate_vector(a::SparseMatrixAssembler,vecdata)
  v1 = nz_counter(get_vector_builder(a),(get_rows(a),))
  symbolic_loop_vector!(v1,a,vecdata)
  v2 = nz_allocation(v1)
  symbolic_loop_vector!(v2,a,vecdata)
  v3 = create_from_nz(v2)
  v3
end

function assemble_vector!(b,a::SparseMatrixAssembler,vecdata)
  fill!(b,zero(eltype(b)))
  assemble_vector_add!(b,a,vecdata)
end

function assemble_vector_add!(b,a::SparseMatrixAssembler,vecdata)
  numeric_loop_vector!(b,a,vecdata)
  create_from_nz(b)
end

function symbolic_loop_vector!(b,a::GenericSparseMatrixAssembler,vecdata)
  get_vec(a::Tuple) = a[1]
  get_vec(a) = a
  if LoopStyle(b) == DoNotLoop()
    return b
  end
  for (cellvec,_cellids) in zip(vecdata...)
    cellids = map_cell_rows(a.strategy,_cellids)
    if length(cellids) > 0
      rows_cache = array_cache(cellids)
      vec1 = get_vec(first(cellvec))
      rows1 = getindex!(rows_cache,cellids,1)
      touch! = TouchEntriesMap()
      touch_cache = return_cache(touch!,b,vec1,rows1)
      caches = touch_cache, rows_cache
      _symbolic_loop_vector!(b,caches,cellids,vec1)
    end
  end
  b
end

@noinline function _symbolic_loop_vector!(A,caches,cellids,vec1)
  touch_cache, rows_cache = caches
  touch! = TouchEntriesMap()
  for cell in 1:length(cellids)
    rows = getindex!(rows_cache,cellids,cell)
    evaluate!(touch_cache,touch!,A,vec1,rows)
  end
end

function numeric_loop_vector!(b,a::GenericSparseMatrixAssembler,vecdata)
  for (cellvec, _cellids) in zip(vecdata...)
    cellids = map_cell_rows(a.strategy,_cellids)
    if length(cellvec) > 0
      rows_cache = array_cache(cellids)
      vals_cache = array_cache(cellvec)
      vals1 = getindex!(vals_cache,cellvec,1)
      rows1 = getindex!(rows_cache,cellids,1)
      add! = AddEntriesMap(+)
      add_cache = return_cache(add!,b,vals1,rows1)
      caches = add_cache, vals_cache, rows_cache
      _numeric_loop_vector!(b,caches,cellvec,cellids)
    end
  end
  b
end

@noinline function _numeric_loop_vector!(vec,caches,cell_vals,cell_rows)
  add_cache, vals_cache, rows_cache = caches
  @assert length(cell_vals) == length(cell_rows)
  add! = AddEntriesMap(+)
  for cell in 1:length(cell_rows)
    rows = getindex!(rows_cache,cell_rows,cell)
    vals = getindex!(vals_cache,cell_vals,cell)
    evaluate!(add_cache,add!,vec,vals,rows)
  end
end

function assemble_matrix(a::Assembler,matdata)
  A = allocate_matrix(a,matdata)
  assemble_matrix!(A,a,matdata)
  A
end

function allocate_matrix(a::SparseMatrixAssembler,matdata)
  m1 = nz_counter(get_matrix_builder(a),(get_rows(a),get_cols(a)))
  symbolic_loop_matrix!(m1,a,matdata)
  m2 = nz_allocation(m1)
  symbolic_loop_matrix!(m2,a,matdata)
  m3 = create_from_nz(m2)
  m3
end

function symbolic_loop_matrix!(A,a::GenericSparseMatrixAssembler,matdata)
  get_mat(a::Tuple) = a[1]
  get_mat(a) = a
  if LoopStyle(A) == DoNotLoop()
    return A
  end
  for (cellmat,_cellidsrows,_cellidscols) in zip(matdata...)
    cellidsrows = map_cell_rows(a.strategy,_cellidsrows)
    cellidscols = map_cell_cols(a.strategy,_cellidscols)
    @assert length(cellidscols) == length(cellidsrows)
    if length(cellidscols) > 0
      rows_cache = array_cache(cellidsrows)
      cols_cache = array_cache(cellidscols)
      mat1 = get_mat(first(cellmat))
      rows1 = getindex!(rows_cache,cellidsrows,1)
      cols1 = getindex!(cols_cache,cellidscols,1)
      touch! = TouchEntriesMap()
      touch_cache = return_cache(touch!,A,mat1,rows1,cols1)
      caches = touch_cache, rows_cache, cols_cache
      _symbolic_loop_matrix!(A,caches,cellidsrows,cellidscols,mat1)
    end
  end
  A
end

@noinline function _symbolic_loop_matrix!(A,caches,cell_rows,cell_cols,mat1)
  touch_cache, rows_cache, cols_cache = caches
  touch! = TouchEntriesMap()
  for cell in 1:length(cell_cols)
    rows = getindex!(rows_cache,cell_rows,cell)
    cols = getindex!(cols_cache,cell_cols,cell)
    evaluate!(touch_cache,touch!,A,mat1,rows,cols)
  end
end

function numeric_loop_matrix!(A,a::GenericSparseMatrixAssembler,matdata)
  for (cellmat,_cellidsrows,_cellidscols) in zip(matdata...)
    cellidsrows = map_cell_rows(a.strategy,_cellidsrows)
    cellidscols = map_cell_cols(a.strategy,_cellidscols)
    @assert length(cellidscols) == length(cellidsrows)
    @assert length(cellmat) == length(cellidsrows)
    if length(cellmat) > 0
      rows_cache = array_cache(cellidsrows)
      cols_cache = array_cache(cellidscols)
      vals_cache = array_cache(cellmat)
      mat1 = getindex!(vals_cache,cellmat,1)
      rows1 = getindex!(rows_cache,cellidsrows,1)
      cols1 = getindex!(cols_cache,cellidscols,1)
      add! = AddEntriesMap(+)
      add_cache = return_cache(add!,A,mat1,rows1,cols1)
      caches = add_cache, vals_cache, rows_cache, cols_cache
      _numeric_loop_matrix!(A,caches,cellmat,cellidsrows,cellidscols)
    end
  end
  A
end

@noinline function _numeric_loop_matrix!(mat,caches,cell_vals,cell_rows,cell_cols)
  add_cache, vals_cache, rows_cache, cols_cache = caches
  add! = AddEntriesMap(+)
  for cell in 1:length(cell_cols)
    rows = getindex!(rows_cache,cell_rows,cell)
    cols = getindex!(cols_cache,cell_cols,cell)
    vals = getindex!(vals_cache,cell_vals,cell)
    evaluate!(add_cache,add!,mat,vals,rows,cols)
  end
end
