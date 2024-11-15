

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

# constructors

function SparsityPattern(a::SparseMatrixAssembler,U::FESpace,V::FESpace)
  m1 = nz_counter(get_matrix_builder(a),(get_rows(a),get_cols(a)))
  cellidsrows = get_cell_dof_ids(V)
  cellidscols = get_cell_dof_ids(U)
  trivial_symbolic_loop_matrix!(m1,cellidsrows,cellidscols)
  m2 = nz_allocation(m1)
  trivial_symbolic_loop_matrix!(m2,cellidsrows,cellidscols)
  m3 = create_from_nz(m2)
  SparsityPattern(m3)
end

function SparsityPattern(U::FESpace,V::FESpace)
  a = SparseMatrixAssembler(U,V)
  SparsityPattern(a,U,V)
end

get_background_matrix(a::SparsityPattern) = @abstractmethod

num_rows(a::SparsityPattern) = size(get_background_matrix(a),1)
num_cols(a::SparsityPattern) = size(get_background_matrix(a),2)
SparseArrays.findnz(a::SparsityPattern) = findnz(get_background_matrix(a))
SparseArrays.nnz(a::SparsityPattern) = nnz(get_background_matrix(a))
SparseArrays.nonzeros(a::SparsityPattern) = nonzeros(get_background_matrix(a))
SparseArrays.nzrange(a::SparsityPattern,row::Integer) = nzrange(get_background_matrix(a),row)
Algebra.nz_index(a::SparsityPattern,row::Integer,col::Integer) = nz_index(get_background_matrix(a),row,col)
get_nonzero_indices(a::SparsityPattern) = get_nonzero_indices(get_background_matrix(a))

recast(v::AbstractVector,A::AbstractSparseMatrix) = @abstractmethod
recast(v::AbstractVector,A::SparseMatrixCSC) = SparseMatrixCSC(A.m,A.n,A.colptr,A.rowval,v)
recast(v::AbstractVector,A::SparseMatrixCSR{Bi}) where Bi = SparseMatrixCSR{Bi}(A.m,A.n,A.rowptr,A.colval,v)
recast(A::AbstractArray,a::SparsityPattern) = recast(A,get_background_matrix(a))

function to_nz_index(i::AbstractArray,a::SparsityPattern)
  i′ = copy(i)
  to_nz_index!(i′,a)
  return i′
end

function to_nz_index!(i::AbstractArray,a::SparsityPattern)
  @abstractmethod
end

function order_sparsity(s::SparsityPattern,U::FESpace,V::FESpace)
  rows = get_dof_map(V)
  cols = get_dof_map(U)
  order_sparsity(s,rows,cols)
end

function order_sparsity(s::SparsityPattern,rows::AbstractDofMap,cols::AbstractDofMap)
  order_sparsity(s,vectorize(rows),vectorize(cols))
end

function CellData.change_domain(
  a::SparsityPattern,
  row::AbstractTrivialDofMap,
  col::AbstractTrivialDofMap
  )

  a
end

"""
"""
struct SparsityPatternCSC{Tv,Ti} <: SparsityPattern
  matrix::SparseMatrixCSC{Tv,Ti}
end

function SparsityPattern(a::SparseMatrixCSC)
  SparsityPatternCSC(a)
end

get_background_matrix(a::SparsityPatternCSC) = a.matrix
SparseArrays.rowvals(a::SparsityPattern) = rowvals(get_background_matrix(a))
SparseArrays.getcolptr(a::SparsityPatternCSC) = SparseArrays.getcolptr(a.matrix)

"""
    order_sparsity(a::SparsityPattern,i,j) -> SparsityPattern

Permutes a sparsity patterns according to indices specified by `i` and `j`,
representing the rows and columns respectively

"""
function order_sparsity(a::SparsityPatternCSC,i::Vector{<:Integer},j::Vector{<:Integer})
  SparsityPatternCSC(a.matrix[i,j])
end

function to_nz_index!(i::AbstractArray,a::SparsityPatternCSC)
  nrows = num_rows(a)
  for (j,index) in enumerate(i)
    if index > 0
      irow = fast_index(index,nrows)
      icol = slow_index(index,nrows)
      i[j] = nz_index(a,irow,icol)
    end
  end
end

function CellData.change_domain(
  a::SparsityPatternCSC,
  row::DofMap{D,Ti},
  col::DofMap{D,Ti}
  ) where {D,Ti}

  m = num_rows(a)
  n = num_cols(a)
  rowval = copy(rowvals(a))
  colptr = copy(SparseArrays.getcolptr(a))
  nzval = copy(nonzeros(a))
  colnnz = zeros(Ti,n)
  entries_to_delete = fill(true,nnz(a))

  cache_row = array_cache(row.dof_to_cell)
  cache_col = array_cache(col.dof_to_cell)
  for j in col
    if j > 0
      cells_col = getindex!(cache_col,col.dof_to_cell,j)
      for i in row
        if i > 0
          cells_row = getindex!(cache_row,row.dof_to_cell,i)
          isempty(intersect(cells_col,cells_row)) && continue
          ij = nz_index(a,i,j)
          entries_to_delete[ij] = false
          colnnz[j] += 1
        end
      end
    end
  end

  @inbounds for j in 1:n
    colptr[j+1] = colnnz[j]
  end
  length_to_ptrs!(colptr)

  deleteat!(rowval,entries_to_delete)
  deleteat!(nzval,entries_to_delete)
  matrix = SparseMatrixCSC(m,n,colptr,rowval,nzval)

  return SparsityPatternCSC(matrix)
end

struct TProductSparsityPattern{A,B} <: SparsityPattern
  sparsity::A
  sparsities_1d::B
end

get_background_matrix(a::TProductSparsityPattern) = get_background_matrix(a.sparsity)

univariate_num_rows(a::TProductSparsityPattern) = Tuple(num_rows.(a.sparsities_1d))
univariate_num_cols(a::TProductSparsityPattern) = Tuple(num_cols.(a.sparsities_1d))
univariate_findnz(a::TProductSparsityPattern) = tuple_of_arrays(findnz.(a.sparsities_1d))
univariate_nnz(a::TProductSparsityPattern) = Tuple(nnz.(a.sparsities_1d))

function order_sparsity(a::TProductSparsityPattern,i::Tuple,j::Tuple)
  is,is_1d = i
  js,js_1d = j
  a′ = order_sparsity(a.sparsity,is,js)
  a′_1d = map(order_sparsity,a.sparsities_1d,is_1d,js_1d)
  TProductSparsityPattern(a′,a′_1d)
end

function to_nz_index!(i::AbstractArray,a::TProductSparsityPattern)
  to_nz_index!(i,a.sparsity)
end

# univariate (tensor product factors) sparse dofs to sparse dofs
function get_sparse_dof_map(sparsity::TProductSparsityPattern,I,J,i,j)
  IJ = get_nonzero_indices(sparsity)
  uids = map((ii,ji)->CartesianIndex.(ii,ji),i,j)

  unrows = univariate_num_rows(sparsity)
  uncols = univariate_num_cols(sparsity)
  unnz = univariate_nnz(sparsity)

  tprows = CartesianIndices(unrows)
  tpcols = CartesianIndices(uncols)

  sparse_dof_map = zeros(eltype(IJ),unnz...)
  uid = zeros(Int,length(uids))
  @inbounds for (k,id) = enumerate(IJ)
    fill!(uid,0)
    irows = tprows[I[k]]
    icols = tpcols[J[k]]
    @inbounds for d in eachindex(uids)
      uidd = uids[d]
      idd = CartesianIndex((irows.I[d],icols.I[d]))
      @inbounds for (l,uiddl) in enumerate(uidd)
        if uiddl == idd
          uid[d] = l
          break
        end
      end
    end
    sparse_dof_map[uid...] = id
  end

  return sparse_dof_map
end
