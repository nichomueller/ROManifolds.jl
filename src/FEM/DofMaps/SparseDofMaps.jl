# sparse maps

abstract type SparseDofMapStyle end
struct FullDofMapIndexing <: SparseDofMapStyle end
struct SparseDofMapIndexing <: SparseDofMapStyle end

# trivial case

struct TrivialSparseDofMap{A<:SparsityPattern} <: AbstractTrivialDofMap
  sparsity::A
end

TrivialDofMap(sparsity::SparsityPattern) = TrivialSparseDofMap(sparsity)
TrivialDofMap(i::TrivialSparseDofMap) = i
Base.size(i::TrivialSparseDofMap) = (nnz(i.sparsity),)

recast(a::AbstractArray,i::TrivialSparseDofMap) = recast(a,i.sparsity)

SparseDofMapStyle(i::TrivialSparseDofMap) = FullDofMapIndexing()

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

# reindexing

SparseDofMapStyle(i::SparseDofMap) = i.index_style

function FullDofMapIndexing(i::SparseDofMap)
  SparseDofMap(i.indices,i.indices_sparse,i.sparsity,FullDofMapIndexing())
end

function SparseDofMapIndexing(i::SparseDofMap)
  SparseDofMap(i.indices,i.indices_sparse,i.sparsity,SparseDofMapIndexing())
end

Base.size(i::SparseDofMap) = size(i.indices)

function Base.getindex(i::SparseDofMap{D,Ti,A,FullDofMapIndexing},j::Vararg{Integer,D}) where {D,Ti,A}
  getindex(i.indices,j...)
end

function Base.setindex!(i::SparseDofMap{D,Ti,A,FullDofMapIndexing},v,j::Vararg{Integer,D}) where {D,Ti,A}
  setindex!(i.indices,v,j...)
end

function Base.getindex(i::SparseDofMap{D,Ti,A,SparseDofMapIndexing},j::Vararg{Integer,D}) where {D,Ti,A}
  getindex(i.indices_sparse,j...)
end

function Base.setindex!(i::SparseDofMap{D,Ti,A,SparseDofMapIndexing},v,j::Vararg{Integer,D}) where {D,Ti,A}
  setindex!(i.indices_sparse,v,j...)
end

function Base.copy(i::SparseDofMap)
  SparseDofMap(copy(i.indices),copy(i.indices_sparse),i.sparsity,i.index_style)
end

function Base.similar(i::SparseDofMap)
  SparseDofMap(similar(i.indices),similar(i.indices_sparse),i.sparsity,i.index_style)
end

recast(A::AbstractArray,i::SparseDofMap) = recast(A,i.sparsity)
