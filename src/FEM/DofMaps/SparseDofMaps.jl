# sparse maps

abstract type SparseDofMapStyle end
struct FullDofMapIndexing <: SparseDofMapStyle end
struct SparseDofMapIndexing <: SparseDofMapStyle end

# trivial case

struct TrivialSparseDofMap{A<:SparsityPattern} <: AbstractTrivialDofMap{Int32}
  sparsity::A
end

function TrivialDofMap(i::TrivialSparseDofMap)
  TrivialSparseDofMap(i.sparsity)
end

Base.size(i::TrivialSparseDofMap) = (nnz(i.sparsity),)
Base.getindex(i::TrivialSparseDofMap,j::Integer) = j

recast(a::AbstractArray,i::TrivialSparseDofMap) = recast(a,i.sparsity)

SparseDofMapStyle(i::TrivialSparseDofMap) = FullDofMapIndexing()

function CellData.change_domain(
  i::TrivialSparseDofMap,
  row::AbstractDofMap{D,Ti},
  col::AbstractDofMap{D,Ti}
  ) where {D,Ti}

  sparsity = change_domain(i.sparsity,row,col)
  TrivialSparseDofMap(sparsity)
end

# non trivial case

"""
    SparseDofMap{D,Ti,A<:AbstractDofMap{D,Ti},B<:TProductSparsityPattern} <: AbstractDofMap{D,Ti}

Index map used to select the nonzero entries of a sparse matrix. The field `sparsity`
contains the tensor product sparsity of the matrix to be indexed. The field `indices`
refers to the nonzero entries of the sparse matrix, whereas `indices_sparse` is
used to access the corresponding sparse entries

"""
struct SparseDofMap{D,Ti,A<:SparsityPattern,B<:SparseDofMapStyle} <: AbstractDofMap{D,Ti}
  indices::Array{Ti,D}
  indices_sparse::Array{Ti,D}
  sparsity::A
  index_style::B
end

function SparseDofMap(
  indices::AbstractArray,
  indices_sparse::AbstractArray,
  sparsity::SparsityPattern)

  index_style = FullDofMapIndexing()
  SparseDofMap(indices,indices_sparse,sparsity,index_style)
end

SparseDofMapStyle(i::SparseDofMap) = i.index_style

function FullDofMapIndexing(i::SparseDofMap)
  SparseDofMap(i.indices,i.indices_sparse,i.sparsity,FullDofMapIndexing())
end

function SparseDofMapIndexing(i::SparseDofMap)
  SparseDofMap(i.indices,i.indices_sparse,i.sparsity,SparseDofMapIndexing())
end

Base.size(i::SparseDofMap) = size(i.indices)

function Base.getindex(i::SparseDofMap{D,Ti,A,FullDofMapIndexing},j::Integer) where {D,Ti,A}
  getindex(i.indices,j)
end

function Base.setindex!(i::SparseDofMap{D,Ti,A,FullDofMapIndexing},v,j::Integer) where {D,Ti,A}
  setindex!(i.indices,v,j)
end

function Base.getindex(i::SparseDofMap{D,Ti,A,SparseDofMapIndexing},j::Integer) where {D,Ti,A}
  getindex(i.indices_sparse,j)
end

function Base.setindex!(i::SparseDofMap{D,Ti,A,SparseDofMapIndexing},v,j::Integer) where {D,Ti,A}
  setindex!(i.indices_sparse,v,j)
end

function Base.copy(i::SparseDofMap)
  SparseDofMap(copy(i.indices),copy(i.indices_sparse),i.sparsity,i.index_style)
end

function Base.similar(i::SparseDofMap)
  SparseDofMap(similar(i.indices),similar(i.indices_sparse),i.sparsity,i.index_style)
end

recast(A::AbstractArray,i::SparseDofMap) = recast(A,i.sparsity)

function CellData.change_domain(
  i::SparseDofMap,
  row::DofMap{D,Ti},
  col::DofMap{D,Ti}
  ) where {D,Ti}

  @notimplemented "Implementation needed, read commented lines"
  # nrows = num_rows(i.sparsity)
  # sparsity = change_domain(i.sparsity,row,col)
  # indices_sparse = sparse_change_domain(i.indices_sparse,row,col,nrows)
  # indices = to_nz_index(indices_sparse,i.sparsity)
  # SparseDofMap(indices,indices_sparse,sparsity,i.index_style)
  # SparseDofMap(i.indices,i.indices_sparse,sparsity,i.index_style)
end

# function sparse_change_domain(
#   indices::Array{Ti,D},
#   row::DofMap{D,Ti},
#   col::DofMap{D,Ti},
#   nrows::Int) where {D,Ti}

#   indices′ = zeros(Ti,size(indices))
#   for (j,ij) in enumerate(indices)
#     col_dof = fast_index(ij,nrows)
#     row_dof = slow_index(ij,nrows)
#     if show_dof(row,row_dof) && show_dof(col,col_dof)
#       indices′[j] = indices[j]
#     end
#   end
#   return indices′
# end

# optimization

function CellData.change_domain(
  i::TrivialSparseDofMap,
  row::EntireDofMap{D,Ti},
  col::EntireDofMap{D,Ti}
  ) where {D,Ti}

  i
end

function CellData.change_domain(
  i::SparseDofMap,
  row::EntireDofMap{D,Ti},
  col::EntireDofMap{D,Ti}
  ) where {D,Ti}

  i
end
