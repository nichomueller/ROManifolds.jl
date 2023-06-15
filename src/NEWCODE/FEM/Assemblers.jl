# Generic assembler interface

function Gridap.FESpaces.assemble_vector(
  a::SparseMatrixAssembler,
  vecdata,
  filter::Tuple{Vararg{Int}})

  r,d = _filter_vecdata(a,vecdata,filter)
  vec = assemble_vector(a,d)
  vec[r]
end

function Gridap.FESpaces.assemble_matrix(
  a::SparseMatrixAssembler,
  matdata,
  filter::Tuple{Vararg{Int}})

  r,c,d = _filter_matdata(a,matdata,filter)
  mat = assemble_matrix(a,d)
  mat[r,c]
end

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

# MDEIM snapshots generation interface
function Gridap.FESpaces.allocate_vector(
  a::SparseMatrixAssembler,
  vecdata::Function,
  filter::Tuple{Vararg{Int}},
  args...)

  d = vecdata(first.(args)...)
  allocate_vector(a,d,filter)
end

function Gridap.FESpaces.allocate_vector(
  a::SparseMatrixAssembler,
  vecdata,
  filter::Tuple{Vararg{Int}})

  r,d = _filter_vecdata(a,vecdata,filter)
  vec = allocate_vector(a,d)
  vec[r]
end

function Gridap.FESpaces.assemble_vector_add!(
  vec::AbstractVector,
  a::SparseMatrixAssembler,
  vecdata,
  filter::Tuple{Vararg{Int}})

  _,d = _filter_vecdata(a,vecdata,filter)
  assemble_vector_add!(vec,a,d)
end

function Gridap.FESpaces.allocate_matrix(
  a::SparseMatrixAssembler,
  matdata::Function,
  filter::Tuple{Vararg{Int}},
  args...)

  d = matdata(first.(args)...)
  allocate_matrix(a,d,filter)
end

function Gridap.FESpaces.allocate_matrix(
  a::SparseMatrixAssembler,
  matdata,
  filter::Tuple{Vararg{Int}})

  r,c,d = _filter_matdata(a,matdata,filter)
  mat = allocate_matrix(a,d)
  mat[r,c]
end

function Gridap.FESpaces.assemble_matrix_add!(
  mat::AbstractMatrix,
  a::SparseMatrixAssembler,
  matdata,
  filter::Tuple{Vararg{Int}})

  _,_,d = _filter_matdata(a,matdata,filter)
  assemble_matrix_add!(mat,a,d)
end

function collect_trian(f::Function,args...)
  domcontrib = f(first.(args)...)
  collect_trian(domcontrib)
end

function compressed_assemble_matrix_add!(mat,a::SparseMatrixAssembler,matdata)
  numeric_loop_matrix!(mat,a,matdata)
  compressed_create_from_nz(mat)
end

function compressed_create_from_nz(a::InserterCSC{Tv,Ti}) where {Tv,Ti}
  k = 1
  for j in 1:a.ncols
    pini = Int(a.colptr[j])
    pend = pini + Int(a.colnnz[j]) - 1
    for p in pini:pend
      a.nzval[k] = a.nzval[p]
      k += 1
    end
  end
  @inbounds for j in 1:a.ncols
    a.colptr[j+1] = a.colnnz[j]
  end
  length_to_ptrs!(a.colptr)
  nnz = a.colptr[end]-1
  resize!(a.nzval,nnz)

  a.nzval
end

function assemble_residual(
  op::ParamFEOperator,
  ::FESolver,
  sols::AbstractMatrix,
  params::Table,
  filter::Tuple{Vararg{Int}})

  vecdatum = _vecdata_jacobian(op,sols,params)
  aff = Affinity(vecdatum,params)
  b = allocate_vector(op.assem,vecdatum(first(params)),filter)
  if isa(aff,ParamAffinity)
    vecdata = vecdatum(first(params))
    assemble_vector_add!(b,op.assem,vecdata,filter)
  elseif isa(aff,NonAffinity)
    vecdata = pmap(μ -> vecdatum(μ),params)
    pmap(d -> assemble_vector_add!(b,op.assem,d,filter),vecdata)
  else
    @unreachable
  end
  b
end

function assemble_residual(
  op::ParamTransientFEOperator,
  solver::θMethod,
  sols::AbstractMatrix,
  params::Table,
  filter::Tuple{Vararg{Int}})

  times = get_times(solver)
  vecdatum = _vecdata_jacobian(op,solver,sols,params)
  aff = Affinity(vecdatum,params,times)
  b = allocate_vector(op.assem,vecdatum(first(params),first(times)),filter)
  if isa(aff,ParamTimeAffinity)
    vecdata = vecdatum(first(params),first(times))
    assemble_vector_add!(b,op.assem,vecdata,filter)
  elseif isa(aff,ParamAffinity)
    vecdata = pmap(t -> vecdatum(first(params),t),times)
    pmap(d -> assemble_vector_add!(b,op.assem,d,filter),vecdata)
  elseif isa(aff,TimeAffinity)
    vecdata = pmap(μ -> vecdatum(μ,first(times)),params)
    pmap(d -> assemble_vector_add!(b,op.assem,d,filter),vecdata)
  elseif isa(aff,NonAffinity)
    vecdata = pmap(μ -> map(t -> vecdatum(μ,t),times),params)
    pmap(d -> assemble_vector_add!(b,op.assem,d,filter),vecdata...)
  else
    @unreachable
  end
  b
end

function assemble_jacobian(
  op::ParamFEOperator,
  ::FESolver,
  sols::AbstractMatrix,
  params::Table,
  filter::Tuple{Vararg{Int}})

  matdatum = _matdata_jacobian(op,sols,params)
  aff = Affinity(matdatum,params)
  A = allocate_matrix(op.assem,matdatum(first(params)),filter)
  if isa(aff,ParamAffinity)
    matdata = matdatum(first(params))
    assemble_matrix_add!(A,op.assem,matdata,filter)
  elseif isa(aff,NonAffinity)
    matdata = pmap(μ -> matdatum(μ),params)
    pmap(d -> assemble_matrix_add!(A,op.assem,d,filter),matdata)
  else
    @unreachable
  end
  A
end

function assemble_jacobian(
  op::ParamTransientFEOperator,
  solver::θMethod,
  matdatum::Function,
  params::Table,
  filter::Tuple{Vararg{Int}})

  times = get_times(solver)
  aff = Affinity(matdatum,params,times)
  A = allocate_matrix(op.assem,matdatum,filter,params,times)
  if isa(aff,ParamTimeAffinity)
    matdata = matdatum(first(params),first(times))
    assemble_matrix_add!(A,op.assem,matdata,filter)
  elseif isa(aff,ParamAffinity)
    matdata = pmap(t -> matdatum(first(params),t),times)
    pmap(d -> assemble_matrix_add!(A,op.assem,d,filter),matdata)
  elseif isa(aff,TimeAffinity)
    matdata = pmap(μ -> matdatum(μ,first(times)),params)
    pmap(d -> assemble_matrix_add!(A,op.assem,d,filter),matdata)
  elseif isa(aff,NonAffinity)
    matdata = pmap(μ -> map(t -> matdatum(μ,t),times),params)
    pmap(dp -> map(dt -> assemble_matrix_add!(A,op.assem,dt,filter),dp),matdata)
  else
    @unreachable
  end
end
