function get_param_matrix_builder(
  a::SparseMatrixAssembler,
  r::AbstractRealization)

  mat = get_matrix_builder(a)
  M = get_array_type(mat)
  T = eltype(M)
  L = length(r)
  pmatrix_type = MatrixOfSparseMatricesCSC{T,Int,L}
  SparseMatrixBuilder(pmatrix_type)
end

function get_param_vector_builder(
  a::SparseMatrixAssembler,
  r::AbstractRealization)

  vec = get_vector_builder(a)
  V = get_array_type(vec)
  T = eltype(V)
  L = length(r)
  pvector_type = ConsecutiveVectorOfVectors{T,L}
  ArrayBuilder(pvector_type)
end

"""
    get_param_assembler(a::SparseMatrixAssembler,r::AbstractRealization) -> SparseMatrixAssembler

Returns a parametric assembler, which assembles parametric residuals and jacobians.
The consequent assembly routines remain the same as in [`Gridap`](@ref), i.e.
SparseMatrixBuilder -> ArrayCounter -> ArrayAllocator -> Array

"""
function get_param_assembler(
  a::SparseMatrixAssembler,
  r::AbstractRealization)

  matrix_builder = get_param_matrix_builder(a,r)
  vector_builder = get_param_vector_builder(a,r)
  rows = FESpaces.get_rows(a)
  cols = FESpaces.get_cols(a)
  strategy = FESpaces.get_assembly_strategy(a)
  GenericSparseMatrixAssembler(matrix_builder,vector_builder,rows,cols,strategy)
end

# like collect_cell_matrix, but for a fixed triangulation `strian`
function collect_cell_matrix_for_trian(
  trial::FESpace,
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation)

  scell_mat = get_contribution(a,strian)
  cell_mat,trian = move_contributions(scell_mat,strian)
  @assert ndims(eltype(cell_mat)) == 2
  cell_mat_c = attach_constraints_cols(trial,cell_mat,trian)
  cell_mat_rc = attach_constraints_rows(test,cell_mat_c,trian)
  rows = get_cell_dof_ids(test,trian)
  cols = get_cell_dof_ids(trial,trian)
  [cell_mat_rc],[rows],[cols]
end

# like collect_cell_vector, but for a fixed triangulation `strian`
function collect_cell_vector_for_trian(
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation)

  scell_vec = get_contribution(a,strian)
  cell_vec,trian = move_contributions(scell_vec,strian)
  @assert ndims(eltype(cell_vec)) == 1
  cell_vec_r = attach_constraints_rows(test,cell_vec,trian)
  rows = get_cell_dof_ids(test,trian)
  [cell_vec_r],[rows]
end

function test_passembler(a::Assembler,matdata,vecdata,data)
  A = allocate_matrix(a,matdata)
  assemble_matrix!(A,a,matdata)
  assemble_matrix_add!(A,a,matdata)
  A = assemble_matrix(a,matdata)
  b = allocate_vector(a,vecdata)
  assemble_vector!(b,a,vecdata)
  assemble_vector_add!(b,a,vecdata)
  b = assemble_vector(a,vecdata)
  A, b = allocate_matrix_and_vector(a,data)
  assemble_matrix_and_vector!(A,b,a,data)
  assemble_matrix_and_vector_add!(A,b,a,data)
  A, b = assemble_matrix_and_vector(a,data)
end
