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
  mat_rc,compress(mat_rc)
end

function assemble_compressed_matrix_add!(
  mat::AbstractMatrix,
  mat_nnz::AbstractMatrix,
  a::SparseMatrixAssembler,
  matdata,
  filter::Tuple{Vararg{Int}})

  _,_,d = _filter_matdata(a,matdata,filter)
  numeric_loop_matrix!(mat,a,d)
  mat_nnz.array = create_compressed_from_nz(mat)
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
    matdata = pmap(t -> matdatum(first(params),t),times)
    pmap(d -> assemble_compressed_matrix_add!(J,Jnnz,op.assem,d,filter),matdata)
  elseif isa(aff,TimeAffinity)
    matdata = pmap(μ -> matdatum(μ,first(times)),params)
    pmap(d -> assemble_compressed_matrix_add!(J,Jnnz,op.assem,d,filter),matdata)
  elseif isa(aff,NonAffinity)
    matdata = pmap(μ -> map(t -> matdatum(μ,t),times),params)
    pmap(dp -> map(dt -> assemble_compressed_matrix_add!(J,Jnnz,op.assem,dt,filter),dp),matdata)
  else
    @unreachable
  end
  Snapshots(Jnnz)
end
