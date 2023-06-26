# Compressed MDEIM snapshots generation interface
function allocate_compressed_matrix(
  a::SparseMatrixAssembler,
  matdata::Function,
  filter::Tuple{Vararg{Int}},
  args...)

  d = matdata(rand.(args)...)
  allocate_compressed_matrix(a,d,filter)
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

function assemble_compressed_jacobian(
  op::ParamTransientFEOperator,
  solver::θMethod,
  s::SingleFieldSnapshots,
  trian::Triangulation,
  filter::Tuple{Vararg{Int}})

  times = get_times(solver)
  sols,params = get_data(s)
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
        J,Jnnz,op.assem,d,filter),matdata);as_emat=false)
    end
  else
    @unreachable
  end
  compress(Jnnz)
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
