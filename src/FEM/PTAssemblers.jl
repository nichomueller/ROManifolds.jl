function FESpaces.collect_cell_vector(
  test::FESpace,
  a::PTDomainContribution)

  w = []
  r = []
  for strian in get_domains(a)
    scell_vec = get_contribution(a,strian)
    cell_vec,trian = move_contributions(scell_vec,strian)
    @assert ndims(eltype(cell_vec)) == 1
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
  @assert ndims(eltype(cell_vec)) == 1
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
    @assert ndims(eltype(cell_mat)) == 2
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
  @assert ndims(eltype(cell_mat)) == 2
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

  Acache = zeros(A)
  for (cellmat,_cellidsrows,_cellidscols) in zip(matdata...)
    cellidsrows = FESpaces.map_cell_rows(a.strategy,_cellidsrows)
    cellidscols = FESpaces.map_cell_cols(a.strategy,_cellidscols)
    cellmat1 = get_at_index(1,cellmat)
    @assert length(cellidscols) == length(cellidsrows)
    @assert length(cellmat1) == length(cellidsrows)
    if length(cellmat1) > 0
      vals_cache = array_cache(cellmat1)
      rows_cache = array_cache(cellidsrows)
      cols_cache = array_cache(cellidscols)
      mat1 = getindex!(vals_cache,cellmat1,1)
      rows1 = getindex!(rows_cache,cellidsrows,1)
      cols1 = getindex!(cols_cache,cellidscols,1)
      add! = AddEntriesMap(+)
      add_cache = return_cache(add!,Acache,mat1,rows1,cols1)
      caches = Acache,add_cache,vals_cache,rows_cache,cols_cache
      FESpaces._numeric_loop_matrix!(A,caches,cellmat,cellidsrows,cellidscols)
    end
  end
  A
end

@noinline function FESpaces._numeric_loop_matrix!(
  mat::PTArray,caches,cell_vals,cell_rows,cell_cols)

  matcache,add_cache,vals_cache,rows_cache,cols_cache = caches
  add! = AddEntriesMap(+)
  for k in eachindex(matcache)
    matk = matcache[k]
    cell_valsk = get_at_index(k,cell_vals)
    for cell in eachindex(cell_cols)
      rows = getindex!(rows_cache,cell_rows,cell)
      cols = getindex!(cols_cache,cell_cols,cell)
      valsk = getindex!(vals_cache,cell_valsk,cell)
      evaluate!(add_cache,add!,matk,valsk,rows,cols)
      mat[k] = matk
    end
  end
end

function FESpaces.numeric_loop_vector!(
  b::PTArray,
  a::GenericSparseMatrixAssembler,
  vecdata)

  bcache = zeros(b)
  for (cellvec,_cellids) in zip(vecdata...)
    cellids = FESpaces.map_cell_rows(a.strategy,_cellids)
    cellvec1 = get_at_index(1,cellvec)
    if length(cellvec1) > 0
      vals_cache = array_cache(cellvec1)
      rows_cache = array_cache(cellids)
      vec1 = getindex!(vals_cache,cellvec1,1)
      rows1 = getindex!(rows_cache,cellids,1)
      add! = AddEntriesMap(+)
      add_cache = return_cache(add!,bcache,vec1,rows1)
      caches = bcache,add_cache,vals_cache,rows_cache
      FESpaces._numeric_loop_vector!(b,caches,cellvec,cellids)
    end
  end
  b
end

@noinline function FESpaces._numeric_loop_vector!(
  vec::PTArray,caches,cell_vals,cell_rows)

  veccache,add_cache,vals_cache,rows_cache = caches
  add! = AddEntriesMap(+)
  for k in eachindex(veccache)
    veck = veccache[k]
    cell_valsk = get_at_index(k,cell_vals)
    for cell in eachindex(rows_cache)
      rows = getindex!(rows_cache,cell_rows,cell)
      valsk = getindex!(vals_cache,cell_valsk,cell)
      evaluate!(add_cache,add!,veck,valsk,rows)
      vec[k] = veck
    end
  end
end
