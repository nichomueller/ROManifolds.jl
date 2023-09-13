_ndims(a) = ndims(eltype(a))
_ndims(a::PTArray) = ndims(eltype(testitem(a)))

function FESpaces.collect_cell_vector(
  test::FESpace,
  a::PTDomainContribution)

  w = []
  r = []
  for strian in get_domains(a)
    scell_vec = get_contribution(a,strian)
    cell_vec,trian = move_contributions(scell_vec,strian)
    @assert _ndims(cell_vec) == 1
    cell_vec_r = attach_constraints_rows(test,cell_vec,trian)
    rows = get_cell_dof_ids(test,trian)
    push!(w,cell_vec_r)
    push!(r,rows)
  end
  (w,r)
end

function FESpaces.collect_cell_vector(
  test::FESpace,
  a::PTDomainContribution,
  strian::Triangulation)

  scell_vec = get_contribution(a,strian)
  cell_vec,trian = move_contributions(scell_vec,strian)
  @assert _ndims(cell_vec) == 1
  cell_vec_r = attach_constraints_rows(test,cell_vec,trian)
  rows = get_cell_dof_ids(test,trian)
  return cell_vec_r,rows
end

function FESpaces.collect_cell_matrix(
  trial::FESpace,
  test::FESpace,
  a::PTDomainContribution)

  w = []
  r = []
  c = []
  for strian in get_domains(a)
    scell_mat = get_contribution(a,strian)
    cell_mat, trian = move_contributions(scell_mat,strian)
    @assert _ndims(cell_mat) == 2
    cell_mat_c = attach_constraints_cols(trial,cell_mat,trian)
    cell_mat_rc = attach_constraints_rows(test,cell_mat_c,trian)
    rows = get_cell_dof_ids(test,trian)
    cols = get_cell_dof_ids(trial,trian)
    push!(w,cell_mat_rc)
    push!(r,rows)
    push!(c,cols)
  end
  (w,r,c)
end

function FESpaces.collect_cell_matrix(
  trial::FESpace,
  test::FESpace,
  a::PTDomainContribution,
  strian::Triangulation)

  scell_mat = get_contribution(a,strian)
  cell_mat,trian = move_contributions(scell_mat,strian)
  @assert _ndims(cell_mat) == 2
  cell_mat_c = attach_constraints_cols(trial,cell_mat,trian)
  cell_mat_rc = attach_constraints_rows(test,cell_mat_c,trian)
  rows = get_cell_dof_ids(test,trian)
  cols = get_cell_dof_ids(trial,trian)
  return cell_mat_rc,rows,cols
end

function FESpaces.numeric_loop_matrix!(
  A::PTArray,
  a::GenericSparseMatrixAssembler,
  matdata)

  for (cellmat,_cellidsrows,_cellidscols) in zip(matdata...)
    cellidsrows = map_cell_rows(a.strategy,_cellidsrows)
    cellidscols = map_cell_cols(a.strategy,_cellidscols)
    cellmat1 = testitem(cellmat)
    @assert length(cellidscols) == length(cellidsrows)
    @assert length(cellmat1) == length(cellidsrows)
    if length(cellmat1) > 0
      vals_cache = array_cache(cellmat)
      rows_cache = array_cache(cellidsrows)
      cols_cache = array_cache(cellidscols)
      mat1 = getindex!(vals_cache,cellmat,1)
      rows1 = getindex!(rows_cache,cellidsrows,1)
      cols1 = getindex!(cols_cache,cellidscols,1)
      add! = AddEntriesMap(+)
      add_cache = return_cache(add!,A,mat1,rows1,cols1)
      caches = add_cache,vals_cache,rows_cache,cols_cache
      FESpaces._numeric_loop_matrix!(A,caches,cellmat,cellidsrows,cellidscols)
    end
  end
  A
end

# @noinline function FESpaces._numeric_loop_matrix!(
#   mat::PTArray,
#   caches,cell_vals,cell_rows,cell_cols)

#   mat_cache,add_cache,vals_cache,rows_cache,cols_cache = caches
#   add! = AddEntriesMap(+)
#   @inbounds for k = eachindex(mat)
#     matk = similar(mat_cache)
#     fillstored!(matk,zero(eltype(matk)))
#     for cell in 1:length(cell_cols)
#       rows = getindex!(rows_cache,cell_rows,cell)
#       cols = getindex!(cols_cache,cell_cols,cell)
#       vals = getindex!(vals_cache,cell_vals,cell)
#       evaluate!(add_cache,add!,matk,vals,rows,cols)
#     end
#     mat.array[k] .= matk
#   end
# end

# @noinline function FESpaces._numeric_loop_matrix!(arr::PTArray,caches,cell_vals,cell_rows,cell_cols)
#   add_cache, vals_cache, rows_cache, cols_cache = caches
#   add! = AddEntriesMap(+)
#   for cell in 1:length(cell_cols)
#     rows = getindex!(rows_cache,cell_rows,cell)
#     cols = getindex!(cols_cache,cell_cols,cell)
#     idx = cols*arr.nrows .+ rows
#     vals = getindex!(vals_cache,cell_vals,cell)
#     evaluate!(add_cache,add!,arr,vals,idx)
#   end
# end
