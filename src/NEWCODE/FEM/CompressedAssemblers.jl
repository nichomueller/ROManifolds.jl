function residuals_cache(assem::SparseMatrixAssembler,vecdata)
  d = first(vecdata)
  vec = allocate_vector(assem,d)
  vec
end

function jacobians_cache(assem::SparseMatrixAssembler,matdata)
  d = first(matdata)
  mat = allocate_matrix(assem,d)
  mat_nnz = compress(mat)
  mat,mat_nnz
end

function collect_residuals!(cache,assem::SparseMatrixAssembler,vecdata)
  numeric_loop_vector!(cache,assem,vecdata)
  cache
end

function collect_jacobians!(cache,assem::SparseMatrixAssembler,matdata)
  mat,mat_nnz = cache
  numeric_loop_matrix!(mat,assem,matdata)
  nnz_i,nnz_v = compress_array(mat)
  mat_nnz.nonzero_val = nnz_v
  mat_nnz.nonzero_idx = nnz_i
  mat_nnz
end
