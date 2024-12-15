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

abstract type SparsityOrderStyle end
struct OrderedSparsityStyle <: SparsityOrderStyle end
struct DefaultSparsityStyle <: SparsityOrderStyle end

is_ordered(::OrderedSparsityStyle) = true
is_ordered(::DefaultSparsityStyle) = false

"""
    abstract type SparsityPattern end

Type used to represent the sparsity pattern of a sparse matrix, usually the
jacobian in a FE problem.

Subtypes:
- [`SparsityCSC`](@ref)
- [`TProductSparsity`](@ref)

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

SparsityOrderStyle(a::SparsityPattern) = DefaultSparsityStyle()

is_ordered(a::SparsityPattern) = is_ordered(SparsityOrderStyle(a))

get_background_matrix(a::SparsityPattern) = @abstractmethod
get_background_sparsity(a::SparsityPattern) = SparsityPattern(get_background_matrix(a))

num_rows(a::SparsityPattern) = size(get_background_matrix(a),1)
num_cols(a::SparsityPattern) = size(get_background_matrix(a),2)
SparseArrays.findnz(a::SparsityPattern) = findnz(get_background_matrix(a))
SparseArrays.nnz(a::SparsityPattern) = nnz(get_background_matrix(a))
SparseArrays.nonzeros(a::SparsityPattern) = nonzeros(get_background_matrix(a))
SparseArrays.nzrange(a::SparsityPattern,row::Integer) = nzrange(get_background_matrix(a),row)
Algebra.nz_index(a::SparsityPattern,row::Integer,col::Integer) = nz_index(get_background_matrix(a),row,col)
get_nonzero_indices(a::SparsityPattern) = get_nonzero_indices(get_background_matrix(a))

# if CSC trait
SparseArrays.rowvals(a::SparsityPattern) = rowvals(get_background_matrix(a))
SparseArrays.getcolptr(a::SparsityPattern) = SparseArrays.getcolptr(get_background_matrix(a))

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
  to_nz_index!(i,get_background_matrix(a))
end

function to_nz_index!(i::AbstractArray,a::AbstractSparseMatrix)
  @abstractmethod
end

function to_nz_index!(i::AbstractArray,a::SparseMatrixCSC)
  nrows = size(a,1)
  for (j,index) in enumerate(i)
    if index > 0
      irow = fast_index(index,nrows)
      icol = slow_index(index,nrows)
      i[j] = nz_index(a,irow,icol)
    end
  end
end

function order_sparsity(s::SparsityPattern,U::FESpace,V::FESpace)
  rows = get_dof_map(V)
  cols = get_dof_map(U)
  order_sparsity(s,rows,cols)
end

function CellData.change_domain(
  a::AbstractMatrix{<:SparsityPattern},
  rows::AbstractVector{<:AbstractDofMap},
  cols::AbstractVector{<:AbstractDofMap}
  )

  @check size(a,1) == length(rows)
  @check size(a,2) == length(cols)
  map(Iterators.product(1:length(rows),1:length(cols))) do (i,j)
    change_domain(a[i,j],rows[i],cols[j])
  end
end

function CellData.change_domain(
  a::SparsityPattern,
  rows::ArrayContribution,
  cols::ArrayContribution)

  @check all( ( tr === tc for (tr,tc) in zip(get_domains(rows),get_domains(cols)) ) )

  trians = get_domains(rows)
  contribution(trians) do trian
    rows_t = get_component(trian,rows[trian];to_scalar=true)
    cols_t = get_component(trian,cols[trian];to_scalar=true)
    change_domain(a,rows_t,cols_t)
  end
end

function Utils.Contribution(
  v::Tuple{Vararg{SparsityPattern}},
  t::Tuple{Vararg{Triangulation}})

  matrix = get_background_matrix(first(v))
  Tv = eltype(matrix)
  ArrayContribution{Tv,2}(v,t)
end

"""
"""
struct SparsityCSC{Tv,Ti} <: SparsityPattern
  matrix::SparseMatrixCSC{Tv,Ti}
end

function SparsityPattern(a::SparseMatrixCSC)
  SparsityCSC(a)
end

get_background_matrix(a::SparsityCSC) = a.matrix

"""
    order_sparsity(a::SparsityPattern,i,j) -> SparsityPattern

Permutes a sparsity patterns according to indices specified by `i` and `j`,
representing the rows and columns respectively

"""
function order_sparsity(a::SparsityCSC,i::AbstractArray,j::AbstractArray)
  unsorted_matrix = a.matrix
  matrix = unsorted_matrix[i,j]
  OrderedSparsity(matrix,unsorted_matrix)
end

abstract type OrderedSparsityPattern <: SparsityPattern end

get_unsorted_matrix(a::OrderedSparsityPattern) = @abstractmethod

#TODO because of code shortcomings, this sparsity pattern should already be ordered;
# to be changed in future versions
function order_sparsity(a::OrderedSparsityPattern,i::AbstractArray,j::AbstractArray)
  a
end

struct OrderedSparsity{A} <: OrderedSparsityPattern
  matrix::A
  unsorted_matrix::A
end

SparsityOrderStyle(a::OrderedSparsity) = OrderedSparsityStyle()

get_background_matrix(a::OrderedSparsity) = a.matrix
get_unsorted_matrix(a::OrderedSparsity) = a.unsorted_matrix

function CellData.change_domain(
  a::SparsityCSC{Tv,Ti},
  row::AbstractDofMap{D},
  col::AbstractDofMap{D}
  ) where {Tv,Ti,D}

  dof_to_cell_row = get_dof_to_cell(row)
  dof_to_cell_col = get_dof_to_cell(col)

  m = num_rows(a)
  n = num_cols(a)
  rowval = copy(rowvals(a))
  colptr = copy(SparseArrays.getcolptr(a))
  nzval = copy(nonzeros(a))
  colnnz = zeros(Ti,n)
  entries_to_delete = fill(true,nnz(a))

  cache_row = array_cache(dof_to_cell_row)
  cache_col = array_cache(dof_to_cell_col)
  for j in col
    if j > 0
      cells_col = getindex!(cache_col,dof_to_cell_col,j)
      for i in row
        if i > 0
          cells_row = getindex!(cache_row,dof_to_cell_row,i)
          if _cell_intersection(row,col,cells_row,cells_col)
            k = nz_index(a,i,j)
            entries_to_delete[k] = false
            colnnz[j] += 1
          end
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

  return SparsityCSC(matrix)
end

function CellData.change_domain(
  a::OrderedSparsity,
  row::AbstractDofMap{D},
  col::AbstractDofMap{D}
  ) where D

  dof_to_cell_row = get_dof_to_cell(row)
  dof_to_cell_col = get_dof_to_cell(col)

  m = num_rows(a)
  n = num_cols(a)
  rowval = copy(rowvals(a))
  colptr = copy(SparseArrays.getcolptr(a))
  nzval = copy(nonzeros(a))
  colnnz = zeros(eltype(rowval),n)
  entries_to_delete = fill(true,nnz(a))

  cache_row = array_cache(dof_to_cell_row)
  cache_col = array_cache(dof_to_cell_col)
  for (ij,j) in enumerate(col)
    if j > 0
      cells_col = getindex!(cache_col,dof_to_cell_col,j)
      for (ii,i) in enumerate(row)
        if i > 0
          cells_row = getindex!(cache_row,dof_to_cell_row,i)
          if _cell_intersection(row,col,cells_row,cells_col)
            k = nz_index(a,ii,ij)
            entries_to_delete[k] = false
            colnnz[ij] += 1
          end
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

  return OrderedSparsity(matrix,get_unsorted_matrix(a))
end

function _cell_intersection(row::AbstractDofMap,col::AbstractDofMap,cells_row,cells_col)
  tface_to_mask_row = get_tface_to_mask(row)
  tface_to_mask_col = get_tface_to_mask(col)
  check = false
  for cell_row in cells_row
    if !tface_to_mask_row[cell_row]
      for cell_col in cells_col
        if !tface_to_mask_col[cell_col]
          if cell_row == cell_col
            check = true
            break
          end
        end
      end
    end
  end
  return check
end

abstract type TProductSparsityPattern <: SparsityPattern end

get_univariate_sparsities(a::TProductSparsityPattern) = @abstractmethod
univariate_num_rows(a::TProductSparsityPattern) = Tuple(num_rows.(get_univariate_sparsities(a)))
univariate_num_cols(a::TProductSparsityPattern) = Tuple(num_cols.(get_univariate_sparsities(a)))
univariate_findnz(a::TProductSparsityPattern) = tuple_of_arrays(findnz.(get_univariate_sparsities(a)))
univariate_nnz(a::TProductSparsityPattern) = Tuple(nnz.(get_univariate_sparsities(a)))

# univariate (tensor product factors) sparse dofs to sparse dofs
function get_sparse_dof_map(a::TProductSparsityPattern,I,J,i,j)
  nrows = num_rows(a)

  uids = map((ii,ji)->CartesianIndex.(ii,ji),i,j)
  unrows = univariate_num_rows(a)
  uncols = univariate_num_cols(a)
  unnz = univariate_nnz(a)

  tprows = CartesianIndices(unrows)
  tpcols = CartesianIndices(uncols)

  sparse_dof_map = zeros(Int32,unnz...)
  uid = zeros(Int32,length(uids))
  @inbounds for k = eachindex(I)
    Ik = I[k]
    Jk = J[k]
    (iszero(Ik) || iszero(Jk)) && continue
    IJk = Ik + (Jk - 1)*nrows
    irows = tprows[Ik]
    icols = tpcols[Jk]
    fill!(uid,0)
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
    sparse_dof_map[uid...] = IJk
  end

  return sparse_dof_map
end

function order_sparse_dof_map(a::TProductSparsityPattern,sparse_dof_map,I,J)
  nrows = num_rows(a)
  IJ = range_2d(vectorize(I),vectorize(J),nrows)
  sparse_odof_map = copy(sparse_dof_map)
  @inbounds for (k,dofk) in enumerate(sparse_dof_map)
    if dofk > 0
      sparse_odof_map[k] = IJ[dofk]
    end
  end
  return sparse_odof_map
end

struct TProductSparsity{A,B} <: TProductSparsityPattern
  sparsity::A
  sparsities_1d::B
end

SparsityOrderStyle(a::TProductSparsity) = SparsityOrderStyle(a.sparsity)

get_background_matrix(a::TProductSparsity) = get_background_matrix(a.sparsity)

get_univariate_sparsities(a::TProductSparsity) = a.sparsities_1d

function order_sparsity(a::TProductSparsity,i::Tuple,j::Tuple)
  is,is_1d = i
  js,js_1d = j
  a′ = order_sparsity(a.sparsity,is,js)
  a′_1d = map(order_sparsity,a.sparsities_1d,is_1d,js_1d)
  TProductSparsity(a′,a′_1d)
end

function CellData.change_domain(
  a::TProductSparsity,
  row::DofMap{D,Ti},
  col::DofMap{D,Ti}
  ) where {D,Ti}

  a′ = change_domain(a.sparsity,row,col)
  TProductSparsity(a′,a.sparsities_1d)
end

struct SparsityToTProductSparsity{A,B,C,D} <: TProductSparsityPattern
  sparsity::A
  sparsities_1d::B
  dof_to_parent_dof_row::C
  dof_to_parent_dof_col::D
end

SparsityOrderStyle(a::SparsityToTProductSparsity) = SparsityOrderStyle(a.sparsity)

get_background_matrix(a::SparsityToTProductSparsity) = get_background_matrix(a.sparsity)

get_univariate_sparsities(a::SparsityToTProductSparsity) = a.sparsities_1d

function order_sparsity(a::SparsityToTProductSparsity,i::Tuple,j::Tuple)
  is_ordered(a) && return a
  is,is_1d = i
  js,js_1d = j
  tp_matrix = get_parent_matrix(a)
  a′ = order_sparsity(SparsityPattern(tp_matrix),get_parent_map(is),get_parent_map(js))
  a′_1d = map(order_sparsity,get_univariate_sparsities(a),is_1d,js_1d)
  SparsityToTProductSparsity(a′,a′_1d,a.dof_to_parent_dof_row,a.dof_to_parent_dof_col)
end

function order_sparse_dof_map(a::SparsityToTProductSparsity,sparse_dof_map,I,J)
  nrows = num_rows(a)
  onrows = maximum(Int,a.dof_to_parent_dof_row)
  IJ = range_2d(vectorize(get_parent_map(I)),vectorize(get_parent_map(J)),nrows)
  sparse_odof_map = copy(sparse_dof_map)
  fill!(sparse_odof_map,zero(eltype(sparse_odof_map)))
  @inbounds for (k,dofk) in enumerate(sparse_dof_map)
    if dofk > 0
      IJk = IJ[dofk]
      Ik = fast_index(IJk,nrows)
      Jk = slow_index(IJk,nrows)
      ik = a.dof_to_parent_dof_row[Ik]
      jk = a.dof_to_parent_dof_col[Jk]
      sparse_odof_map[k] = ik + (jk-1)*onrows
    end
  end
  return sparse_odof_map
end

function CellData.change_domain(
  a::SparsityToTProductSparsity,
  row::DofMapPortion{D,Ti},
  col::DofMapPortion{D,Ti}
  ) where {D,Ti}

  @notimplemented
end

function CellData.change_domain(
  a::SparsityToTProductSparsity{<:OrderedSparsityPattern},
  row::DofMapPortion{D,Ti},
  col::DofMapPortion{D,Ti}
  ) where {D,Ti}

  a′ = change_domain(a.sparsity,row.parent_map,col.parent_map)
  SparsityToTProductSparsity(a′,a.sparsities_1d,a.dof_to_parent_dof_row,a.dof_to_parent_dof_col)
end

function recast(A::AbstractArray,a::SparsityToTProductSparsity)
  recast(A,get_parent_matrix(a))
end

function get_parent_matrix(a::SparsityToTProductSparsity)
  I,J,V = get_parent_findnz(a)
  m = length(a.dof_to_parent_dof_row)
  n = length(a.dof_to_parent_dof_col)
  sparse(I,J,V,m,n)
end

function get_parent_matrix(a::SparsityToTProductSparsity{<:OrderedSparsityPattern})
  get_background_matrix(a)
end

function get_parent_findnz(a::SparsityToTProductSparsity)
  A = get_background_matrix(a)
  get_parent_findnz(A,a.dof_to_parent_dof_row,a.dof_to_parent_dof_col)
end

function get_parent_findnz(
  A::SparseMatrixCSC{Tv,Ti},
  dof_to_parent_dof_row::AbstractArray,
  dof_to_parent_dof_col::AbstractArray
  ) where {Tv,Ti}

  I,J,V = findnz(A)
  l2pl_row = get_parent_locations(dof_to_parent_dof_row)
  l2pl_col = get_parent_locations(dof_to_parent_dof_col)

  for k in eachindex(I)
    I[k] = l2pl_row[I[k]]
    J[k] = l2pl_col[J[k]]
  end

  return I,J,V
end

# utils

function Base.getindex(
  A::AbstractSparseMatrixCSC{Tv,Ti},
  I::AbstractDofMap,
  J::AbstractDofMap
  ) where {Tv,Ti<:Integer}

  I′ = vectorize(I)
  J′ = vectorize(J)

  fill!(A.nzval,one(Tv))
  k = findfirst(iszero,A)
  fill!(A.nzval,zero(Tv))
  iz,jz = Tuple(k)

  for (ii,i) in enumerate(I′)
    if i == 0
      I′[ii] = iz
    end
  end

  for (ij,j) in enumerate(J′)
    if j == 0
      J′[ij] = jz
    end
  end

  return A[I′,J′]
end

# to nz indices in case we don't provide a sparsity pattern

function to_nz_index(i::AbstractArray)
  nz_sortperm(i)
end

function nz_sortperm(a::AbstractArray)
  a′ = similar(a)
  nz_sortperm!(a′,a)
  a′
end

function nz_sortperm!(a′::AbstractArray,a::AbstractArray)
  fill!(a′,zero(eltype(a′)))
  inz = findall(!iszero,a)
  anz = sortperm(a[inz])
  inds = LinearIndices(size(anz))
  a′[inz[anz]] = inds
end

function get_parent_locations(a::AbstractArray{Ti}) where Ti
  inds = LinearIndices(a)
  anz = inds[findall(!iszero,a)]
  l2pl = zeros(Ti,length(anz))
  for (inz,anzi) in enumerate(anz)
    l2pl[inz] = anzi
  end
  return l2pl
end
