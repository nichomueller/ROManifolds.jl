struct MatrixOfSparseMatricesCSC{Tv,Ti<:Integer,P<:Matrix{Tv}} <: AbstractMatrixOfMatrices{Tv,2,2}
  m::Int64
  n::Int64
  colptr::Vector{Ti}
  rowval::Vector{Ti}
  nzval::P
end

SparseArrays.getcolptr(A::MatrixOfSparseMatricesCSC) = A.colptr
SparseArrays.rowvals(A::MatrixOfSparseMatricesCSC) = A.rowval
SparseArrays.nonzeros(A::MatrixOfSparseMatricesCSC) = A.nzval
_nonzeros(A::MatrixOfSparseMatricesCSC,iblock::Integer...) = @inbounds getindex(A.nzval,:,iblock...)

function MatrixOfSparseMatricesCSC(A::AbstractVector{SparseMatrixCSC{Tv,Ti}}) where {Tv,Ti}
  m,n = innersize(A)
  colptr,rowval,nzval = innerpattern(A)
  matval = Matrix{Tv}(undef,innersize(nzval)...,length(nzval))
  copyto!(eachcol(matval),nzval)
  MatrixOfSparseMatricesCSC(m,n,colptr,rowval,matval)
end

function ArraysOfArrays.ArrayOfSimilarArrays(A::AbstractVector{<:SparseMatrixCSC})
  MatrixOfSparseMatricesCSC(A)
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

Base.@propagate_inbounds function Base.setindex!(A::MatrixOfSparseMatricesCSC,v,i::Integer...)
  A[i...] = v
  A
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
  copyto!(A.colptr,B.colptr)
  copyto!(A.rowval,B.rowval)
  copyto!(A.nzval,B.nzval)
  A
end
