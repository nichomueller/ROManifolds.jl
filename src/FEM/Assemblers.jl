function Gridap.FESpaces.allocate_vector(
  a::SparseMatrixAssembler,
  vecdata,
  filter)

  r,d = _filter_data(a,vecdata,filter)
  vec = allocate_vector(a,d)
  vec[r]
end

function Gridap.FESpaces.assemble_vector(
  a::SparseMatrixAssembler,
  vecdata,
  filter)

  r,d = _filter_data(a,vecdata,filter)
  vec = assemble_vector(a,d)
  vec[r]
end

function Gridap.FESpaces.allocate_matrix(
  a::SparseMatrixAssembler,
  matdata,
  filter)

  r,c,d = _filter_data(a,matdata,filter)
  mat = allocate_matrix(a,d)
  mat[r,c]
end

function Gridap.FESpaces.assemble_matrix(
  a::SparseMatrixAssembler,
  matdata,
  filter)

  r,c,d = _filter_data(a,matdata,filter)
  mat = assemble_matrix(a,d)
  mat[r,c]
end

# function _filter_vecidx(a::SparseMatrixAssembler,filter)
#   r_filter, = filter
#   _idx_in_block(get_rows(a),r_filter)
# end

# function _filter_matidx(a::SparseMatrixAssembler,filter)
#   r_filter,c_filter = filter
#   r = _idx_in_block(get_rows(a),r_filter)
#   c = _idx_in_block(get_cols(a),c_filter)
#   r,c
# end

function _filter_vecdata(
  a::SparseMatrixAssembler,
  vecdata::Tuple{Vararg{Any}},
  filter)

  vals,rowids, = vecdata
  r_filter, = filter
  r = _idx_in_block(get_rows(a),r_filter)
  d = _filter_data(vals,filter),_filter_data(rowids,r_filter)
  r,d
end

function _filter_matdata(
  a::SparseMatrixAssembler,
  matdata::Tuple{Vararg{Any}},
  filter)

  vals,rowids,colids = matdata
  r_filter,c_filter = filter
  r = _idx_in_block(get_rows(a),r_filter)
  c = _idx_in_block(get_cols(a),c_filter)
  d = _filter_data(vals,filter),_filter_data(rowids,r_filter),_filter_data(colids,c_filter)
  r,c,d
end

function _idx_in_block(ndofs::Vector{Int},filter::Int)
  @assert filter == 1
  first(ndofs):last(ndofs)
end

function _idx_in_block(ndofs::BlockedUnitRange,filter::Int)
  nd = [0,ndofs.lasts...]
  [idx1+1:idx2 for (idx1,idx2) in zip(nd[1:end-1],nd[2:end])][filter]
end

_filter_data(data,args...) = data

function _filter_data(data::Vector{Any},filter)
  [_filter_data(d,filter) for d = data]
end

function _filter_data(data::LazyArray,filter)
  lazy_map(d->_filter_data(d,filter),data)
end

function _filter_data(data::ArrayBlock,filter)
  data[filter...]
end

function _filter_data(
  data::Tuple{MatrixBlock{Matrix{Float}},VectorBlock{Vector{Float}}},
  filter::NTuple{2,Int})

  mdata,vdata = data
  r_filter,c_filter = filter
  mdata[r_filter,c_filter],vdata[r_filter]
end
