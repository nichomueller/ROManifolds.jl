function Gridap.FESpaces.allocate_vector(
  a::SparseMatrixAssembler,
  vecdata,
  filter)

  r,d = _filter_vecdata(a,vecdata,filter)
  vec = allocate_vector(a,d)
  vec[r]
end

function Gridap.FESpaces.assemble_vector(
  a::SparseMatrixAssembler,
  vecdata,
  filter)

  r,d = _filter_vecdata(a,vecdata,filter)
  vec = assemble_vector(a,d)
  vec[r]
end

function Gridap.FESpaces.assemble_vector_add!(
  vec::AbstractVector,
  a::SparseMatrixAssembler,
  vecdata,
  filter)

  _,d = _filter_vecdata(a,vecdata,filter)
  assemble_vector_add!(vec,a,d)
end

function Gridap.FESpaces.allocate_matrix(
  a::SparseMatrixAssembler,
  matdata,
  filter)

  r,c,d = _filter_matdata(a,matdata,filter)
  mat = allocate_matrix(a,d)
  mat[r,c]
end

function Gridap.FESpaces.assemble_matrix(
  a::SparseMatrixAssembler,
  matdata,
  filter)

  r,c,d = _filter_matdata(a,matdata,filter)
  mat = assemble_matrix(a,d)
  mat[r,c]
end

function Gridap.FESpaces.assemble_matrix_add!(
  mat::AbstractVector,
  a::SparseMatrixAssembler,
  matdata,
  filter)

  _,_,d = _filter_matdata(a,matdata,filter)
  assemble_matrix_add!(mat,a,d)
end

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

# MDEIM snapshots generation interface

function assemble_residual(
  ::FESolver,
  op::ParamFEOperator,
  sols::AbstractMatrix,
  params::Table,
  filter)

  vecdatum = _vecdata_residual(op,sols,params)
  vecdata = pmap(μ -> vecdatum(μ),params)
  b = allocate_vector(op.assem,first(vecdata),filter)
  pmap(d -> assemble_vector_add!(b,op.assem,d,filter),vecdata...)
end

function assemble_residual(
  solver::θMethod,
  op::ParamTransientFEOperator,
  sols::AbstractMatrix,
  params::Table,
  filter)

  vecdatum = _vecdata_residual(solver,op,sols,params)
  vecdata = pmap(μ -> map(t -> vecdatum(μ),times),params)
  b = allocate_vector(op.assem,first(vecdata...),filter)
  pmap(d -> assemble_vector_add!(b,op.assem,d,filter),vecdata...)
end

function assemble_jacobian(
  op::ParamFEOperator,
  params::Table,
  sols::AbstractMatrix,
  filter)

  matdatum = _matdata_jacobian(op,sols,params)
  matdata = pmap(μ -> matdatum(μ),params)
  A = allocate_matrix(op.assem,first(matdata),filter)
  pmap(d -> assemble_matrix_add!(A,op.assem,d,filter),matdata...)
end

function assemble_jacobian(
  solver::θMethod,
  op::ParamTransientFEOperator,
  params::Table,
  sols::AbstractMatrix,
  filter)

  matdatum = _matdata_jacobian(solver,op,sols,params)
  matdata = pmap(μ -> map(t -> matdatum(μ),times),params)
  A = allocate_matrix(op.assem,first(matdata...),filter)
  pmap(d -> assemble_matrix_add!(A,op.assem,d,filter),matdata...)
end

for T in (:ParamMultiFieldTrialFESpace,:ParamTransientMultiFieldTrialFESpace)

  @eval begin
    function get_snapshots(
      trial::T,
      test::FESpace,
      biform::Function,
      liform::Function,
      args...)

      am = SparseMatrixAssembler(trial,test)
      av = SparseMatrixAssembler(test,test)
      matvecdata,matdata,vecdata =
        collect_cell_matrix_and_vector(trial,test,biform,liform,args...)
      nfields = test.nfields
      for r_filter = 1:nfields, c_filter = 1:nfields
        vecs = allocate_vector()
      end
    end

  end

end
