# function CellData.attach_constraints_rows(
#   cellvec::LazyArray{<:Fill{<:PTMap{IntegrationMap}}},
#   cellconstr,
#   cellmask=Fill(true,length(cellconstr)))

#   N = length(cellvec)
#   ptmap = PTMap(ConstrainRowsMap())
#   lazy_map(ptmap,cellvec,Fill(cellconstr,N),Fill(cellmask,N))
# end

# function CellData.attach_constraints_cols(
#   cellmat::LazyArray{<:Fill{<:PTMap{IntegrationMap}}},
#   cellconstr,
#   cellmask=Fill(true,length(cellconstr)))

#   N = length(cellvec)
#   ptmap = PTMap(ConstrainColsMap())
#   cellconstr_t = lazy_map(transpose,cellconstr)
#   lazy_map(ptmap,cellmat,Fill(cellconstr_t,N),Fill(cellmask,N))
# end

_ndims(x) = isconcretetype(x) ? eltype(x) : _ndims(eltype(x))

function FESpaces.collect_cell_vector(
  test::FESpace,
  a::DomainContribution)

  w = []
  r = []
  for strian in get_domains(a)
    scell_vec = get_contribution(a,strian)
    cell_vec, trian = move_contributions(scell_vec,strian)
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
  a::DomainContribution,
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
  a::DomainContribution,
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

function collect_cell_matrix(
  trial::FESpace,
  test::FESpace,
  a::DomainContribution)

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

function allocate_nnz_vector(a::SparseMatrixAssembler,matdata::LazyArray{<:Fill{<:PTMap}})
  as_vector = true
  cache = array_cache(matdata)
  matdata1 = getindex!(cache,matdata,1)

  m1 = nz_counter(get_matrix_builder(a),(get_rows(a),get_cols(a)))
  symbolic_loop_matrix!(m1,a,matdata1)
  m2 = nz_allocation(m1)
  symbolic_loop_matrix!(m2,a,matdata1)
  m3 = create_from_nz(Val(as_vector),m2)
  m3
end

function FESpaces.create_from_nz(::Val{false},a::InserterCSC)
  create_from_nz(a)
end

function FESpaces.create_from_nz(::Val{true},a::InserterCSC)
  k = 1
  for j in 1:a.ncols
    pini = Int(a.colptr[j])
    pend = pini + Int(a.colnnz[j]) - 1
    for p in pini:pend
      a.nzval[k] = a.nzval[p]
      a.rowval[k] = a.rowval[p]
      k += 1
    end
  end
  @inbounds for j in 1:a.ncols
    a.colptr[j+1] = a.colnnz[j]
  end
  length_to_ptrs!(a.colptr)
  nnz = a.colptr[end]-1
  resize!(a.rowval,nnz)
  resize!(a.nzval,nnz)
  nnzvec_idx = get_nonzero_idx(a.colptr,a.rowval)
  T = eltype(a.nzval)
  NnzArray{T}(a.nzval,nnzvec_idx,a.nrows)
end

function FESpaces.numeric_loop_matrix!(
  A::Vector{T},
  a::GenericSparseMatrixAssembler,
  matdata) where {T<:NnzArray}

  for (all_cellmat,_cellidsrows,_cellidscols) in zip(matdata...)
    cellidsrows = map_cell_rows(a.strategy,_cellidsrows)
    cellidscols = map_cell_cols(a.strategy,_cellidscols)
    @assert length(cellidscols) == length(cellidsrows)
    if length(cellidsrows) > 0
      all_vals_cache = array_cache(all_cellmat)
      rows_cache = array_cache(cellidsrows)
      cols_cache = array_cache(cellidscols)
      rows1 = getindex!(rows_cache,cellidsrows,1)
      cols1 = getindex!(cols_cache,cellidscols,1)
      add! = AddEntriesMap(+)
      loop_cache = (all_cellmat,cellidsrows,cellidscols,all_vals_cache,
        rows_cache,cols_cache,rows1,cols1,add!)
      _cellmat_loop!(A,loop_cache)
    end
  end
  A
end

function _cellmat_loop!(A,loop_cache)
  (all_cellmat,cellidsrows,cellidscols,all_vals_cache,rows_cache,cols_cache,
    rows1,cols1,add!) = loop_cache
  Acache = first(A)
  for k = eachindex(all_cellmat)
    Ak = copy(Acache)
    cellmatk = getindex!(all_vals_cache,all_cellmat,k)
    @assert length(cellmat) == length(cellidsrows)
    vals_cache = array_cache(cellmatk)
    mat1 = getindex!(vals_cache,cellmatk,1)
    add_cache = return_cache(add!,A,mat1,rows1,cols1)
    caches = add_cache, vals_cache, rows_cache, cols_cache
    _numeric_loop_matrix!(Ak,caches,cellmat,cellidsrows,cellidscols)
    A[k] = Ak
  end
end

@noinline function FESpaces._numeric_loop_matrix!(arr::NnzArray,caches,cell_vals,cell_rows,cell_cols)
  add_cache, vals_cache, rows_cache, cols_cache = caches
  add! = AddEntriesMap(+)
  for cell in 1:length(cell_cols)
    rows = getindex!(rows_cache,cell_rows,cell)
    cols = getindex!(cols_cache,cell_cols,cell)
    idx = cols*arr.nrows .+ rows
    vals = getindex!(vals_cache,cell_vals,cell)
    evaluate!(add_cache,add!,arr,vals,idx)
  end
end

function FESpaces.numeric_loop_vector!(
  b::Vector{<:AbstractVector},
  a::GenericSparseMatrixAssembler,
  vecdata)

  for (all_cellvec, _cellids) in zip(vecdata...)
    cellids = FESpaces.map_cell_rows(a.strategy,_cellids)
    if length(cellids) > 0
      all_vals_cache = array_cache(all_cellvec)
      rows_cache = array_cache(cellids)
      rows1 = getindex!(rows_cache,cellids,1)
      add! = AddEntriesMap(+)
      loop_cache = all_cellvec,cellids,all_vals_cache,rows_cache,rows1,add!
      _cellvec_loop!(b,loop_cache)
    end
  end
  b
end

function _cellvec_loop!(b,loop_cache)
  all_cellvec,cellids,all_vals_cache,rows_cache,rows1,add! = loop_cache
  bcache = first(b)
  for k = eachindex(all_cellvec)
    bk = copy(bcache)
    cellveck = getindex!(all_vals_cache,all_cellvec,k)
    vals_cache = array_cache(cellveck)
    vals1 = getindex!(vals_cache,cellveck,1)
    add_cache = return_cache(add!,b,vals1,rows1)
    caches = add_cache, vals_cache, rows_cache
    FESpaces._numeric_loop_vector!(bk,caches,cellveck,cellids)
    b[k] = bk
  end
end
