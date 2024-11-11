function get_sparsity(a::SparseMatrixAssembler,U::FESpace,V::FESpace,args...)
  m1 = nz_counter(get_matrix_builder(a),(get_rows(a),get_cols(a)))
  cellidsrows = get_cell_dof_ids(V,args...)
  cellidscols = get_cell_dof_ids(U,args...)
  trivial_symbolic_loop_matrix!(m1,cellidsrows,cellidscols)
  m2 = nz_allocation(m1)
  trivial_symbolic_loop_matrix!(m2,cellidsrows,cellidscols)
  m3 = create_from_nz(m2)
  SparsityPattern(m3)
end

function get_sparsity(U::FESpace,V::FESpace,args...)
  get_sparsity(SparseMatrixAssembler(U,V),U,V,args...)
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

"""
    abstract type SparsityPattern end

Type used to represent the sparsity pattern of a sparse matrix, usually the
jacobian in a FE problem.

Subtypes:
- [`SparsityPatternCSC`](@ref)
- [`TProductSparsityPattern`](@ref)

"""
abstract type SparsityPattern end

function to_nz_index(i::AbstractArray,sparsity::SparsityPattern)
  i′ = copy(i)
  to_nz_index!(i′,sparsity)
  return i′
end

function to_nz_index!(i::AbstractArray,sparsity::SparsityPattern)
  @abstractmethod
end

function permute_sparsity(s::SparsityPattern,U::FESpace,V::FESpace)
  index_map_I = get_dof_index_map(V)
  index_map_J = get_dof_index_map(U)
  permute_sparsity(s,index_map_I,index_map_J)
end

"""
"""
struct SparsityPatternCSC{Tv,Ti} <: SparsityPattern
  matrix::SparseMatrixCSC{Tv,Ti}
end

SparsityPattern(a::SparseMatrixCSC) = SparsityPatternCSC(a)

get_sparsity(a::SparsityPatternCSC) = a
num_rows(a::SparsityPatternCSC) = size(a.matrix,1)
num_cols(a::SparsityPatternCSC) = size(a.matrix,2)
SparseArrays.findnz(a::SparsityPatternCSC) = findnz(a.matrix)
SparseArrays.nnz(a::SparsityPatternCSC) = nnz(a.matrix)
SparseArrays.nzrange(a::SparsityPatternCSC,row::Integer) = nzrange(a.matrix,row)
SparseArrays.rowvals(a::SparsityPatternCSC) = rowvals(a.matrix)
get_nonzero_indices(a::SparsityPatternCSC) = get_nonzero_indices(a.matrix)

recast(v::AbstractVector,A::SparseMatrixCSC) = SparseMatrixCSC(A.m,A.n,A.colptr,A.rowval,v)
recast(A::AbstractArray,a::SparsityPatternCSC) = recast(A,a.matrix)

"""
    permute_sparsity(a::SparsityPattern,i,j) -> SparsityPattern

Permutes a sparsity patterns according to indices specified by `i` and `j`,
representing the rows and columns respectively

"""
function permute_sparsity(a::SparsityPatternCSC,i::AbstractVector,j::AbstractVector)
  SparsityPatternCSC(a.matrix[i,j])
end

function sum_sparsities(a::Tuple{Vararg{SparsityPatternCSC}})
  get_matrix(a::SparsityPatternCSC) = a.matrix
  matrices = map(get_matrix,a)
  matrix = _sparse_sum_preserve_sparsity(matrices)
  SparsityPatternCSC(matrix)
end

function to_nz_index!(i::AbstractArray,sparsity::SparsityPatternCSC)
  nrows = num_rows(sparsity)
  for (j,index) in enumerate(i)
    if index > 0
      irow = fast_index(index,nrows)
      icol = slow_index(index,nrows)
      i[j] = nz_index(sparsity.matrix,irow,icol)
    end
  end
end

struct MultiValueSparsityPatternCSC{Tv,Ti} <: SparsityPattern
  matrix::SparseMatrixCSC{Tv,Ti}
  ncomps::Int
end

get_sparsity(a::MultiValueSparsityPatternCSC) = a
num_rows(a::MultiValueSparsityPatternCSC) = size(a.matrix,1)
num_cols(a::MultiValueSparsityPatternCSC) = size(a.matrix,2)
SparseArrays.findnz(a::MultiValueSparsityPatternCSC) = findnz(a.matrix)
SparseArrays.nnz(a::MultiValueSparsityPatternCSC) = nnz(a.matrix)
get_nonzero_indices(a::MultiValueSparsityPatternCSC) = get_nonzero_indices(a.matrix)
TensorValues.num_components(a::MultiValueSparsityPatternCSC) = a.ncomps

recast(A::AbstractArray,a::MultiValueSparsityPatternCSC) = @notimplemented

function sum_sparsities(a::Tuple{Vararg{MultiValueSparsityPatternCSC}})
  item = first(a)
  ncomps = item.ncomps
  @check all((num_components(ai)==ncomps for ai in a))

  get_matrix(a::MultiValueSparsityPatternCSC) = a.matrix
  matrices = map(get_matrix,a)
  matrix = _sparse_sum_preserve_sparsity(matrices)
  MultiValueSparsityPatternCSC(matrix,ncomps)
end

function to_nz_index!(i::AbstractArray,sparsity::MultiValueSparsityPatternCSC)
  nrows = num_rows(sparsity)
  for (j,index) in enumerate(i)
    if index > 0
      irow = fast_index(index,nrows)
      icol = slow_index(index,nrows)
      i[j] = nz_index(sparsity.matrix,irow,icol)
    end
  end
end

"""
"""
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
univariate_num_rows(a::TProductSparsityPattern) = Tuple(num_rows.(a.sparsities_1d))
univariate_num_cols(a::TProductSparsityPattern) = Tuple(num_cols.(a.sparsities_1d))
univariate_findnz(a::TProductSparsityPattern) = tuple_of_arrays(findnz.(a.sparsities_1d))
univariate_nnz(a::TProductSparsityPattern) = Tuple(nnz.(a.sparsities_1d))
univariate_nonzero_indices(a::TProductSparsityPattern) = Tuple(get_nonzero_indices.(a.sparsities_1d))

recast(A::AbstractArray,a::TProductSparsityPattern) = recast(A,a.sparsity)

function permute_sparsity(a::TProductSparsityPattern,i,j)
  is,is_1d = i
  js,js_1d = j
  psparsity = permute_sparsity(a.sparsity,is,js)
  psparsities_1d = map(permute_sparsity,a.sparsities_1d,is_1d,js_1d)
  TProductSparsityPattern(psparsity,psparsities_1d)
end

function sum_sparsities(s::Tuple{Vararg{TProductSparsityPattern}})
  item = first(s)
  sitem_1d = get_univariate_sparsity(item)
  ssparsity = sum_sparsities(map(get_sparsity,s))
  TProductSparsityPattern(ssparsity,sitem_1d)
end

function to_nz_index!(i::AbstractArray,sparsity::TProductSparsityPattern)
  to_nz_index!(i,get_sparsity(sparsity))
end

# utils

function _sparse_sum_preserve_sparsity(A::Tuple{Vararg{SparseMatrixCSC{Tv}}}) where Tv
  for a in A
    fill!(a.nzval,one(Tv))
  end
  B = sum(A)
  fill!(B.nzval,zero(Tv))
  return B
end
