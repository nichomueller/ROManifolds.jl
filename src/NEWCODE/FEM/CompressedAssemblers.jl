function _filter_data(
  vecdata::Tuple{Vararg{Any}},
  filter::Tuple{Vararg{Any}})

  vals,ids = vecdata
  _filter_data(vals,filter),ids...
end

function _filter_data(data::Vector{Any},filter::Tuple{Vararg{Any}})
  map(data) do dout # loop over domain contributions
    lazy_map(dout) do din # loop over cells
      _filter_data(din,filter)
    end
  end
end

_filter_data(data,args...) = data

function _filter_data(data::ArrayBlock,filter::Tuple{Vararg{Any}})
  data[filter...]
end

function _filter_data(
  data::Tuple{MatrixBlock{Matrix{Float}},VectorBlock{Vector{Float}}},
  filter::Tuple{Vararg{Any}})

  mdata,vdata = data
  r_filter,c_filter = filter
  mdata[r_filter,c_filter],vdata[r_filter]
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

for Top in (:ParamFEOperator,:ParamTransientFEOperator)
  @eval begin
    function residuals_cache(op::$Top,vecdata,filter::Tuple{Vararg{Any}})
      r_filter, = filter
      assem = SparseMatrixAssembler(op.test[r_filter],op.test[r_filter])
      d = first(vecdata)
      vec = allocate_vector(assem,d)
      vec
    end

    function jacobians_cache(op::$Top,matdata,filter::Tuple{Vararg{Any}})
      r_filter,c_filter = filter
      assem = SparseMatrixAssembler(op.trial[c_filter],op.test[r_filter])
      d = first(matdata)
      mat = allocate_matrix(a,d)
      mat_nnz = compress(mat)
      mat,mat_nnz
    end

    function collect_residuals!(cache,op::$Top,d)
      vecdata = last(d)
      assemble_compressed_vector_add!(cache,op.assem,vecdata)
    end

    function collect_jacobians!(cache,op::$Top,d)
      j,jnnz = cache
      matdata = last(d)
      assemble_compressed_matrix_add!(j,jnnz,op.assem,matdata)
    end
  end
end
