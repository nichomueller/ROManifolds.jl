function _filter_vecdata(
  a::SparseMatrixAssembler,
  vecdata::Tuple{Vararg{Any}},
  filter::Tuple{Vararg{Int}})

  vals,rowids, = vecdata
  r_filter, = filter
  r = _idx_in_block(get_rows(a),r_filter)
  d = _filter_data(vals,filter),_filter_data(rowids,r_filter)
  r,d
end

function _filter_matdata(
  a::SparseMatrixAssembler,
  matdata::Tuple{Vararg{Any}},
  filter::Tuple{Vararg{Int}})

  vals,rowids,colids = matdata
  r_filter,c_filter = filter
  r = _idx_in_block(get_rows(a),r_filter)
  c = _idx_in_block(get_cols(a),c_filter)
  d = _filter_data(vals,filter),_filter_data(rowids,r_filter),_filter_data(colids,c_filter)
  r,c,d
end

function _idx_in_block(ndofs::Base.OneTo{Int},args...)
  ndofs
end

function _idx_in_block(ndofs::BlockedUnitRange,filter::Int)
  nd = [0,ndofs.lasts...]
  [idx1+1:idx2 for (idx1,idx2) in zip(nd[1:end-1],nd[2:end])][filter]
end

_filter_data(data,args...) = data

function _filter_data(data::Vector{Any},filter::Tuple{Vararg{Int}}) # loop over domain contributions
  map(d->_filter_data(d,filter),data)
end

function _filter_data(data::LazyArray,filter::Tuple{Vararg{Int}}) # loop over cells
  lazy_map(d->_filter_data(d,filter),data)
end

function _filter_data(data::ArrayBlock,filter::Tuple{Vararg{Int}})
  data[filter...]
end

function _filter_data(
  data::Tuple{MatrixBlock{Matrix{Float}},VectorBlock{Vector{Float}}},
  filter::Tuple{Vararg{Int}})

  mdata,vdata = data
  r_filter,c_filter = filter
  mdata[r_filter,c_filter],vdata[r_filter]
end

# Compressed MDEIM snapshots generation interface
# function residuals_cache(
#   a::SparseMatrixAssembler,
#   ::FESolver,
#   params::Table,
#   vecdata::Function)

#   r,d = vecdata(rand(params))
#   vec = allocate_vector(a,d)
#   vec_r = vec[r]
#   vec_r
# end

# function jacobians_cache(
#   a::SparseMatrixAssembler,
#   ::FESolver,
#   params::Table,
#   matdata::Function)

#   r,c,d = matdata(rand(params))
#   mat = allocate_matrix(a,d)
#   mat_rc = mat[r,c]
#   mat_rc,compress(mat_rc)
# end

function residuals_cache(a::SparseMatrixAssembler,vecdata)
  r,d = first(vecdata)
  vec = allocate_vector(a,d)
  vec_r = vec[r]
  vec_r
end

function jacobians_cache(a::SparseMatrixAssembler,matdata)
  r,c,d = first(matdata)
  mat = allocate_matrix(a,d)
  mat_rc = mat[r,c]
  mat_rc,compress(mat_rc)
end

function assemble_compressed_vector_add!(
  vec::AbstractVector,
  a::SparseMatrixAssembler,
  vecdata)

  numeric_loop_vector!(vec,a,vecdata)
  vec
end

function assemble_compressed_matrix_add!(
  mat::SparseMatrixCSC,
  mat_nnz::NnzArray,
  a::SparseMatrixAssembler,
  matdata)

  numeric_loop_matrix!(mat,a,matdata)
  nnz_i,nnz_v = compress_array(mat)
  mat_nnz.nonzero_val = nnz_v
  mat_nnz.nonzero_idx = nnz_i
  mat_nnz
end

function collect_residuals!(cache,op::ParamTransientFEOperator,d)
  vecdata = last(d)
  assemble_compressed_vector_add!(cache,op.assem,vecdata)
end

function collect_jacobians!(cache,op::ParamTransientFEOperator,d)
  j,jnnz = cache
  matdata = last(d)
  assemble_compressed_matrix_add!(j,jnnz,op.assem,matdata)
end
