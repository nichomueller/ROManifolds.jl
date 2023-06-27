# Compressed MDEIM snapshots generation interface
function allocate_compressed_vector(
  a::SparseMatrixAssembler,
  vecdata::Function,
  filter::Tuple{Vararg{Int}},
  args...)

  d = vecdata(rand.(args)...)
  allocate_compressed_vector(a,d,filter)
end

function allocate_compressed_matrix(
  a::SparseMatrixAssembler,
  matdata::Function,
  filter::Tuple{Vararg{Int}},
  args...)

  d = matdata(rand.(args)...)
  allocate_compressed_matrix(a,d,filter)
end

function allocate_compressed_vector(
  a::SparseMatrixAssembler,
  vecdata,
  filter::Tuple{Vararg{Int}})

  r,d = _filter_vecdata(a,vecdata,filter)
  vec = allocate_vector(a,d)
  vec[r]
end

function allocate_compressed_matrix(
  a::SparseMatrixAssembler,
  matdata,
  filter::Tuple{Vararg{Int}})

  r,c,d = _filter_matdata(a,matdata,filter)
  mat = allocate_matrix(a,d)
  mat_rc = mat[r,c]
  mat_rc,NnzArray(mat_rc)
end

function assemble_compressed_vector_add!(
  vec::AbstractVector,
  a::SparseMatrixAssembler,
  vecdata,
  filter::Tuple{Vararg{Int}})

  _,d = _filter_vecdata(a,vecdata,filter)
  numeric_loop_vector!(vec,a,d)
  vec
end

function assemble_compressed_matrix_add!(
  mat::AbstractMatrix,
  mat_nnz::NnzArray{<:SparseMatrixCSC},
  a::SparseMatrixAssembler,
  matdata,
  filter::Tuple{Vararg{Int}})

  _,_,d = _filter_matdata(a,matdata,filter)
  numeric_loop_matrix!(mat,a,d)
  nnz_i,nnz_v = compress(mat)
  mat_nnz.array = nnz_v
  mat_nnz.nonzero_idx = nnz_i
  mat_nnz
end

function assemble_compressed_residual(
  op::ParamTransientFEOperator,
  solver::θMethod,
  trian::Triangulation,
  s::SingleFieldSnapshots,
  params::Table,
  filter::Tuple{Vararg{Int}})

  times = get_times(solver)
  sols = get_data(s)
  vecdatum = _vecdata_residual(feop,solver,sols,params,trian)
  aff = Affinity(vecdatum,params,times)
  r = allocate_compressed_vector(op.assem,vecdatum,filter,params,times)
  rtemp = if isa(aff,ParamTimeAffinity)
    vecdata = vecdatum(first(params),first(times))
    reshape(assemble_compressed_vector_add!(r,op.assem,vecdata,filter),:,1)
  elseif isa(aff,ParamAffinity)
    vecdata = map(t -> vecdatum(first(params),t),times)
    map(d -> assemble_compressed_vector_add!(r,op.assem,d,filter),vecdata)
  elseif isa(aff,TimeAffinity)
    vecdata = map(μ -> vecdatum(μ,first(times)),params)
    map(d -> assemble_compressed_vector_add!(r,op.assem,d,filter),vecdata)
  elseif isa(aff,NonAffinity)
    map(params) do μ
      vecdata = map(t -> vecdatum(μ,t),times)
      hcat(map(d -> assemble_compressed_vector_add!(
        r,op.assem,d,filter),vecdata)...)
    end
  else
    @unreachable
  end
  rnnz = NnzArray(rtemp)
  Snapshots(rnnz)
end

function assemble_compressed_jacobian(
  op::ParamTransientFEOperator,
  solver::θMethod,
  trian::Triangulation,
  s::SingleFieldSnapshots,
  params::Table,
  filter::Tuple{Vararg{Int}})

  times = get_times(solver)
  sols = get_data(s)
  matdatum = _matdata_jacobian(feop,solver,sols,params,trian)
  aff = Affinity(matdatum,params,times)
  J,Jnnz = allocate_compressed_matrix(op.assem,matdatum,filter,params,times)
  Jnnz = if isa(aff,ParamTimeAffinity)
    matdata = matdatum(first(params),first(times))
    assemble_compressed_matrix_add!(J,Jnnz,op.assem,matdata,filter)
  elseif isa(aff,ParamAffinity)
    matdata = map(t -> matdatum(first(params),t),times)
    map(d -> assemble_compressed_matrix_add!(J,Jnnz,op.assem,d,filter),matdata)
  elseif isa(aff,TimeAffinity)
    matdata = map(μ -> matdatum(μ,first(times)),params)
    map(d -> assemble_compressed_matrix_add!(J,Jnnz,op.assem,d,filter),matdata)
  elseif isa(aff,NonAffinity)
    map(params) do μ
      matdata = map(t -> matdatum(μ,t),times)
      compress(map(d -> assemble_compressed_matrix_add!(
        J,Jnnz,op.assem,d,filter),matdata);type=Matrix{Float})
    end
  else
    @unreachable
  end
  Snapshots(Jnnz)
end

# function assemble_compressed_jacobian(
#   op::ParamTransientFEOperator,
#   solver::θMethod,
#   s::SingleFieldSnapshots,
#   trian::Triangulation,
#   filter::Tuple{Vararg{Int}})

#   times = get_times(solver)
#   sols,params = get_data(s)
#   matdatum = _matdata_jacobian(feop,solver,sols,params,trian)
#   aff = Affinity(matdatum,params,times)
#   J,Jnnz = allocate_compressed_matrix(op.assem,matdatum,filter,params,times)
#   Jnnz = if isa(aff,ParamTimeAffinity)
#     matdata = matdatum(first(params),first(times))
#     assemble_compressed_matrix_add!(J,Jnnz,op.assem,matdata,filter)
#   elseif isa(aff,ParamAffinity)
#     matdata = pmap(t -> matdatum(first(params),t),times)
#     pmap(d -> assemble_compressed_matrix_add!(J,Jnnz,op.assem,d,filter),matdata)
#   elseif isa(aff,TimeAffinity)
#     matdata = pmap(μ -> matdatum(μ,first(times)),params)
#     pmap(d -> assemble_compressed_matrix_add!(J,Jnnz,op.assem,d,filter),matdata)
#   elseif isa(aff,NonAffinity)
#     map(params) do μ
#       matdata = map(t -> matdatum(μ,t),times)
#       compress(map(d -> assemble_compressed_matrix_add!(
#         J,Jnnz,op.assem,d,filter),matdata);as_emat=false)
#     end
#   else
#     @unreachable
#   end
#   compress(Jnnz)
# end
