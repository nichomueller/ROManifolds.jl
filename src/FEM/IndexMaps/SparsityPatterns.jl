abstract type SparsityPattern end

struct SparsityPatternCSC{Tv,Ti} <: SparsityPattern
  matrix::SparseMatrixCSC{Tv,Ti}
end

SparsityPattern(a::SparseMatrixCSC) = SparsityPatternCSC(a)

get_sparsity(a::SparsityPatternCSC) = a
num_rows(a::SparsityPatternCSC) = size(a.matrix,1)
num_cols(a::SparsityPatternCSC) = size(a.matrix,2)
SparseArrays.findnz(a::SparsityPatternCSC) = findnz(a.matrix)
SparseArrays.nnz(a::SparsityPatternCSC) = nnz(a.matrix)
get_nonzero_indices(a::SparsityPatternCSC) = get_nonzero_indices(a.matrix)

function permute_sparsity(a::SparsityPatternCSC,i::AbstractVector,j::AbstractVector)
  SparsityPatternCSC(a.matrix[i,j])
end

function permute_sparsity(a::SparsityPatternCSC,i::AbstractArray,j::AbstractArray)
  permute_sparsity(a,vec(i),vec(j))
end

struct MultiValuePatternCSC{Tv,Ti} <: SparsityPattern
  matrix::SparseMatrixCSC{Tv,Ti}
  ncomps::Int
end

get_sparsity(a::MultiValuePatternCSC) = a
num_rows(a::MultiValuePatternCSC) = size(a.matrix,1)
num_cols(a::MultiValuePatternCSC) = size(a.matrix,2)
SparseArrays.findnz(a::MultiValuePatternCSC) = findnz(a.matrix)
SparseArrays.nnz(a::MultiValuePatternCSC) = nnz(a.matrix)
get_nonzero_indices(a::MultiValuePatternCSC) = get_nonzero_indices(a.matrix)
TensorValues.num_components(a::MultiValuePatternCSC) = a.ncomps

struct TProductSparsityPattern{A,B} <: SparsityPattern
  sparsity::A
  sparsities_1d::B
end

get_sparsity(a::TProductSparsityPattern) = a.sparsity
get_univariate_sparsity(a::TProductSparsityPattern) = a.sparsities_1d
num_rows(a::TProductSparsityPattern) = num_rows(a.sparsity)
num_cols(a::TProductSparsityPattern) = num_cols(a.sparsity)
SparseArrays.findnz(a::TProductSparsityPattern) = findnz(a.sparsity)
SparseArrays.nnz(a::TProductSparsityPattern) = nnz(a.sparsity)
get_nonzero_indices(a::TProductSparsityPattern) = get_nonzero_indices(a.sparsity)
univariate_num_rows(a::TProductSparsityPattern) = num_rows.(a.sparsities_1d)
univariate_num_cols(a::TProductSparsityPattern) = num_cols.(a.sparsities_1d)
univariate_findnz(a::TProductSparsityPattern) = tuple_of_arrays(findnz.(a.sparsities_1d))
univariate_nnz(a::TProductSparsityPattern) = nnz.(a.sparsities_1d)
univariate_nonzero_indices(a::TProductSparsityPattern) = get_nonzero_indices.(a.sparsities_1d)

TensorValues.num_components(a::TProductSparsityPattern{<:MultiValuePatternCSC}) = num_components(a.sparsity)

function get_sparsity(a::SparseMatrixAssembler,U::FESpace,V::FESpace)
  m1 = nz_counter(get_matrix_builder(a),(get_rows(a),get_cols(a)))
  cellidsrows = get_cell_dof_ids(V)
  cellidscols = get_cell_dof_ids(U)
  trivial_symbolic_loop_matrix!(m1,cellidsrows,cellidscols)
  m2 = nz_allocation(m1)
  trivial_symbolic_loop_matrix!(m2,cellidsrows,cellidscols)
  m3 = create_from_nz(m2)
  SparsityPattern(m3)
end

function get_sparsity(U::FESpace,V::FESpace)
  get_sparsity(SparseMatrixAssembler(U,V),U,V)
end

function trivial_symbolic_loop_matrix!(A,cellidsrows,cellidscols)
  mat1 = nothing
  rows_cache = array_cache(cellidsrows)
  cols_cache = array_cache(cellidscols)

  rows1 = getindex!(rows_cache,cellidsrows,1)
  cols1 = getindex!(cols_cache,cellidscols,1)

  touch! = FESpaces.TouchEntriesMap()
  touch_cache = return_cache(touch!,A,mat1,rows1,cols1)
  caches = touch_cache,rows_cache,cols_cache

  FESpaces._symbolic_loop_matrix!(A,caches,cellidsrows,cellidscols,mat1)
  return A
end

function permute_sparsity(a::TProductSparsityPattern,i,j)
  is,is_1d = i
  js,js_1d = j
  psparsity = permute_sparsity(a.sparsity,is,js)
  psparsities_1d = map(permute_sparsity,a.sparsities_1d,is_1d,js_1d)
  TProductSparsityPattern(psparsity,psparsities_1d)
end

function permute_sparsity(s::SparsityPattern,U::FESpace,V::FESpace)
  psparsity = permute_sparsity(s.sparsity,U.space,V.space)
  psparsities = map(permute_sparsity,s.sparsities_1d,U.spaces_1d,V.spaces_1d)
  TProductSparsityPattern(psparsity,psparsities)
end
