struct MatrixOfSparseMatricesCSC{Tv,Ti<:Integer} <: AbstractArrayOfSimilarArrays{Tv,2,2}
  m::Int64
  n::Int64
  colptr::Vector{Ti}
  rowval::Vector{Ti}
  nzval::Matrix{Tv}
end

SparseArrays.getcolptr(A::MatrixOfSparseMatricesCSC) = A.colptr
SparseArrays.rowvals(A::MatrixOfSparseMatricesCSC) = A.rowval
_nonzeros(A::MatrixOfSparseMatricesCSC,iblock::Integer...) = @inbounds getindex(A.nzval,:,iblock...)

function MatrixOfSparseMatricesCSC(A::AbstractVector{SparseMatrixCSC{Tv,Ti}}) where {Tv,Ti}
  m,n = innersize(A)
  colptr,rowval,nzval = innerpattern(A)
  matval = Matrix{Tv}(undef,innersize(nzval)...,length(nzval))
  copyto!(eachcol(matval),nzval)
  MatrixOfSparseMatricesCSC(m,n,colptr,rowval,matval)
end

function innerpattern(A::AbstractVector{<:SparseMatrixCSC})
  @check !isempty(A)
  colptr,rowval = first(A).colptr,first(A).rowval
  if (any(A->A.colptr != colptr, A) || any(A->A.rowval != rowval, A))
    throw(DimensionMismatch("Sparsity pattern of element csc matrices is not equal"))
  end
  nzval = map(nonzeros,A)
  return colptr,rowval,nzval
end

@inline function ArraysOfArrays.innersize(A::MatrixOfSparseMatricesCSC)
  (A.m,A.n)
end

@inline function ArraysOfArrays._innerlength(A::AbstractArray{<:SparseMatrixCSC})
  prod(innersize(A))
end

ArraysOfArrays.flatview(A::MatrixOfSparseMatricesCSC) = A.nzval

Base.size(A::MatrixOfSparseMatricesCSC) = (size(A.nzval,2),size(A.nzval,2))

# function outerindex(A::MatrixOfSparseMatricesCSC,irow::Integer,icol::Integer)
#   blockrow = slow_index(irow,size(A,1))
#   blockcol = slow_index(icol,size(A,2))
#   return blockrow,blockcol
# end

# function innerindex(A::MatrixOfSparseMatricesCSC,irow::Integer,icol::Integer)
#   blockrow = fast_index(irow,size(A,1))
#   blockcol = fast_index(icol,size(A,2))
#   return blockrow,blockcol
# end

# @inline function is_diagonal(A::MatrixOfSparseMatricesCSC,irow::Integer,icol::Integer)
#   blockrow,blockcol = outerindex(A,irow,icol)
#   blockrow==blockcol
# end

function Base.show(io::IO,::MIME"text/plain",A::MatrixOfSparseMatricesCSC)
  println(io, "Block diagonal matrix of sparse matrices, with sparsity pattern: ")
  show(io,MIME("text/plain"),A[1])
end

Base.@propagate_inbounds function Base.getindex(A::MatrixOfSparseMatricesCSC,iblock::Integer)
  irow = fast_index(iblock,size(A,1))
  icol = slow_index(iblock,size(A,1))
  getindex(A,irow,icol)
end

Base.@propagate_inbounds function Base.getindex(A::MatrixOfSparseMatricesCSC,irow::Integer,icol::Integer)
  diagonal_getindex(Val(irow==icol),A,irow)
end

Base.@propagate_inbounds function diagonal_getindex(
  ::Val{true},
  A::MatrixOfSparseMatricesCSC{T},
  iblock::Integer) where T

  SparseMatrixCSC(A.m,A.n,A.colptr,A.rowval,_nonzeros(A,iblock))
end

Base.@propagate_inbounds function diagonal_getindex(
  ::Val{false},
  A::MatrixOfSparseMatricesCSC{T},
  iblock::Integer) where T

  spzeros(innersize(A))
end

function Base.similar(A::MatrixOfSparseMatricesCSC)
  colptr = similar(A.colptr)
  rowval = similar(A.rowval)
  nzvals = similar(A.nzval)
  MatrixOfSparseMatricesCSC(A.m,A.n,colptr,rowval,nzvals)
end

function Base.similar(
  A::MatrixOfSparseMatricesCSC{Tv,Ti},
  ::Type{<:SparseMatrixCSC{Tv′,Ti′}}
  ) where {Tv,Ti,Tv′,Ti′}

  colptr = similar(A.colptr,Ti′)
  rowval = similar(A.rowval,Ti′)
  nzvals = similar(A.nzval,Tv′)
  MatrixOfSparseMatricesCSC(A.m,A.n,colptr,rowval,nzvals)
end

function Base.copyto!(A::MatrixOfSparseMatricesCSC,B::MatrixOfSparseMatricesCSC)
  @check size(A) == size(B)
  copyto!(B.colptr,A.colptr)
  copyto!(B.rowval,A.rowval)
  copyto!(B.nzval,A.nzval)
  B
end


# Parametric functions

param_length(A::AbstractArrayOfSimilarArrays) = length(A.data)
param_length(A::MatrixOfSparseMatricesCSC) = size(A.nzval,2)

param_eachindex(A::AbstractArrayOfSimilarArrays) = Base.OneTo(length(A))

Base.@propagate_inbounds function param_getindex(A::VectorOfSimilarArrays,i::Integer)
  getindex(A,i)
end

Base.@propagate_inbounds function param_getindex(A::MatrixOfSparseMatricesCSC,i::Integer)
  @boundscheck Base.checkbounds(A.nzval,:,i)
  SparseMatrixCSC(A.m,A.n,A.colptr,A.rowval,_nonzeros(A,i))
end
