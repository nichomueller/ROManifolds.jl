"""
    parameterize(a::SparseMatrixAssembler,r::AbstractRealization) -> SparseMatrixAssembler

Returns an assembler that also stores the parametric length of `r`. This function
is to be used to assemble parametric residuals and Jacobians. The assembly routines
follow the same pipeline as in `Gridap`
"""
function ParamDataStructures.parameterize(a::SparseMatrixAssembler,r::AbstractRealization)
  matrix_builder = parameterize(get_matrix_builder(a),r)
  vector_builder = parameterize(get_vector_builder(a),r)
  rows = FESpaces.get_rows(a)
  cols = FESpaces.get_cols(a)
  strategy = FESpaces.get_assembly_strategy(a)
  GenericSparseMatrixAssembler(matrix_builder,vector_builder,rows,cols,strategy)
end

function ParamDataStructures.parameterize(
  a::MultiField.BlockSparseMatrixAssembler{NB,NV,SB,P},
  r::AbstractRealization) where {NB,NV,SB,P}

  matrix_builder = parameterize(_getfirst(get_matrix_builder(a)),r)
  vector_builder = parameterize(_getfirst(get_vector_builder(a)),r)
  rows = FESpaces.get_rows(a)
  cols = FESpaces.get_cols(a)
  strategy = FESpaces.get_assembly_strategy(a)
  block_idx = CartesianIndices((NB,NB))
  block_assemblers = map(block_idx) do idx
    r = rows[idx[1]]
    c = cols[idx[2]]
    s = strategy[idx[1],idx[2]]
    GenericSparseMatrixAssembler(matrix_builder,vector_builder,r,c,s)
  end
  MultiField.BlockSparseMatrixAssembler{NB,NV,SB,P}(block_assemblers)
end

_getfirst(a::Fields.ArrayBlock) = a[findfirst(a.touched)]
_getfirst(a::Fields.ArrayBlockView) = _getfirst(a.array)

function FESpaces.assemble_vector_add!(
  b::BlockParamVector,
  a::MultiField.BlockSparseMatrixAssembler,
  vecdata)
  b1 = ArrayBlock(blocks(b),fill(true,blocksize(b)))
  b2 = MultiField.expand_blocks(a,b1)
  FESpaces.assemble_vector_add!(b2,a,vecdata)
end

function FESpaces.assemble_matrix_add!(
  mat::BlockParamMatrix,
  a::MultiField.BlockSparseMatrixAssembler,
  matdata)
  m1 = ArrayBlock(blocks(mat),fill(true,blocksize(mat)))
  m2 = MultiField.expand_blocks(a,m1)
  FESpaces.assemble_matrix_add!(m2,a,matdata)
end

function FESpaces.assemble_matrix_and_vector_add!(
  A::BlockParamMatrix,
  b::BlockParamVector,
  a::MultiField.BlockSparseMatrixAssembler,
  data)
  m1 = ArrayBlock(blocks(A),fill(true,blocksize(A)))
  m2 = MultiField.expand_blocks(a,m1)
  b1 = ArrayBlock(blocks(b),fill(true,blocksize(b)))
  b2 = MultiField.expand_blocks(a,b1)
  FESpaces.assemble_matrix_and_vector_add!(m2,b2,a,data)
end

"""
    function collect_lazy_cell_matrix(
      trial::FESpace,
      test::FESpace,
      a::DomainContribution,
      index::Int
      ) -> Tuple{Vector{<:Any},Vector{<:Any},Vector{<:Any}}

For parametric applications, returns the cell-wise data needed to assemble a
global sparse matrix corresponding to the parameter #`index`
"""
function collect_lazy_cell_matrix(
  trial::FESpace,
  test::FESpace,
  a::DomainContribution,
  index::Int)

  w = []
  r = []
  c = []
  for strian in get_domains(a)
    scell_mat = get_contribution(a,strian)
    cell_mat, trian = move_contributions(scell_mat,strian)
    @assert ndims(eltype(cell_mat)) == 2
    cell_mat_c = attach_constraints_cols(trial,cell_mat,trian)
    cell_mat_rc = attach_constraints_rows(test,cell_mat_c,trian)
    cell_mat_rc_i = lazy_param_getindex(cell_mat_rc,index)
    rows = get_cell_dof_ids(test,trian)
    cols = get_cell_dof_ids(trial,trian)
    push!(w,cell_mat_rc_i)
    push!(r,rows)
    push!(c,cols)
  end
  (w,r,c)
end

"""
    function collect_lazy_cell_vector(
      test::FESpace,
      a::DomainContribution,
      index::Int
      ) -> Tuple{Vector{<:Any},Vector{<:Any}}

For parametric applications, returns the cell-wise data needed to assemble a
global vector corresponding to the parameter #`index`
"""
function collect_lazy_cell_vector(
  test::FESpace,
  a::DomainContribution,
  index::Int)

  w = []
  r = []
  for strian in get_domains(a)
    scell_vec = get_contribution(a,strian)
    cell_vec, trian = move_contributions(scell_vec,strian)
    @assert ndims(eltype(cell_vec)) == 1
    cell_vec_r = attach_constraints_rows(test,cell_vec,trian)
    cell_vec_r_i = lazy_param_getindex(cell_vec_r,index)
    rows = get_cell_dof_ids(test,trian)
    push!(w,cell_vec_r_i)
    push!(r,rows)
  end
  (w,r)
end

function assemble_lazy_matrix_add!(A,a::SparseMatrixAssembler,matdata,index::Int)
  numeric_loop_lazy_matrix!(A,a,matdata,index)
  create_from_nz(A)
end

function assemble_lazy_vector_add!(b,a::SparseMatrixAssembler,vecdata,index::Int)
  numeric_loop_lazy_vector!(b,a,vecdata)
  create_from_nz(b)
end

function numeric_loop_lazy_matrix!(A,a::SparseMatrixAssembler,matdata,index::Int)
  strategy = get_assembly_strategy(a)
  for (cellmat,_cellidsrows,_cellidscols) in zip(matdata...)
    cellidsrows = map_cell_rows(strategy,_cellidsrows)
    cellidscols = map_cell_cols(strategy,_cellidscols)
    @assert length(cellidscols) == length(cellidsrows)
    @assert length(cellmat) == length(cellidsrows)
    if length(cellmat) > 0
      cellmat_i = lazy_param_getindex(cellmat,index)
      rows_cache = array_cache(cellidsrows)
      cols_cache = array_cache(cellidscols)
      vals_cache = array_cache(cellmat_i)
      mat1 = getindex!(vals_cache,cellmat_i,1)
      rows1 = getindex!(rows_cache,cellidsrows,1)
      cols1 = getindex!(cols_cache,cellidscols,1)
      add! = FESpaces.AddEntriesMap(+)
      add_cache = return_cache(add!,A,mat1,rows1,cols1)
      caches = add_cache,vals_cache,rows_cache,cols_cache
      FESpaces._numeric_loop_matrix!(A,caches,cellmat_i,cellidsrows,cellidscols)
    end
  end
  A
end

function numeric_loop_lazy_vector!(b,a::SparseMatrixAssembler,vecdata,index::Int)
  strategy = get_assembly_strategy(a)
  for (cellvec, _cellids) in zip(vecdata...)
    cellids = map_cell_rows(strategy,_cellids)
    if length(cellvec) > 0
      cellvec_i = lazy_param_getindex(cellvec,index)
      rows_cache = array_cache(cellids)
      vals_cache = array_cache(cellvec_i)
      vals1 = getindex!(vals_cache,cellvec_i,1)
      rows1 = getindex!(rows_cache,cellids,1)
      add! = AddEntriesMap(+)
      add_cache = return_cache(add!,b,vals1,rows1)
      caches = add_cache, vals_cache, rows_cache
      _numeric_loop_vector!(b,caches,cellvec_i,cellids)
    end
  end
  b
end
