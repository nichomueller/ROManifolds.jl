struct MatrixOfSparseMatricesCSC{Tv,Ti<:Integer,L,P<:AbstractMatrix{Tv}} <: ParamSparseMatrixCSC{Tv,Ti,L}
  m::Int64
  n::Int64
  colptr::Vector{Ti}
  rowval::Vector{Ti}
  data::P
  function MatrixOfSparseMatricesCSC(
    m::Int64,
    n::Int64,
    colptr::Vector{Ti},
    rowval::Vector{Ti},
    data::P
    ) where {Tv,Ti,P<:AbstractMatrix{Tv}}

    L = size(data,2)
    new{Tv,Ti,L,P}(m,n,colptr,rowval,data)
  end
end

SparseArrays.getcolptr(A::MatrixOfSparseMatricesCSC) = A.colptr
SparseArrays.rowvals(A::MatrixOfSparseMatricesCSC) = A.rowval
SparseArrays.nonzeros(A::MatrixOfSparseMatricesCSC) = A.data
_nonzeros(A::MatrixOfSparseMatricesCSC,iblock::Integer...) = @inbounds getindex(A.data,:,iblock...)

function MatrixOfSparseMatricesCSC(A::AbstractVector{SparseMatrixCSC{Tv,Ti}}) where {Tv,Ti}
  m,n = innersize(A)
  colptr,rowval,data = innerpattern(A)
  B = ArrayOfSimilarArrays(data)
  MatrixOfSparseMatricesCSC(m,n,colptr,rowval,B.data)
end

function innerpattern(A::AbstractVector{<:SparseMatrixCSC})
  @check !isempty(A)
  colptr,rowval = first(A).colptr,first(A).rowval
  if (any(A->A.colptr != colptr, A) || any(A->A.rowval != rowval, A))
    throw(DimensionMismatch("Sparsity pattern of element csc matrices is not equal"))
  end
  data = map(nonzeros,A)
  return colptr,rowval,data
end

Base.size(A::MatrixOfSparseMatricesCSC) = (param_length(A),param_length(A))

@inline function ArraysOfArrays.innersize(A::MatrixOfSparseMatricesCSC)
  (A.m,A.n)
end

param_getindex(a::MatrixOfSparseMatricesCSC,i::Integer) = diagonal_getindex(Val(true),a,i)

Base.@propagate_inbounds function Base.getindex(A::MatrixOfSparseMatricesCSC,i::Vararg{Integer,2})
  @boundscheck checkbounds(A,i...)
  iblock = first(i)
  diagonal_getindex(Val(all(i.==iblock)),A,iblock)
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

Base.@propagate_inbounds function Base.setindex!(A::MatrixOfSparseMatricesCSC,v,i::Vararg{Integer,2})
  @boundscheck checkbounds(A,i...)
  iblock = first(i)
  irow==icol && diagonal_setindex!(Val(true),A,v,iblock)
end

Base.@propagate_inbounds function diagonal_setindex!(::Val{true},A::MatrixOfSparseMatricesCSC,v,iblock::Integer)
  setindex!(_nonzeros(A,iblock),v)
end

Base.@propagate_inbounds function diagonal_setindex!(::Val{false},A::MatrixOfSparseMatricesCSC,v,iblock::Integer)
  @notimplemented
end

function Base.similar(
  A::MatrixOfSparseMatricesCSC{Tv,Ti},
  ::Type{<:SparseMatrixCSC{Tv′,Ti′}}
  ) where {Tv,Ti,Tv′,Ti′}

  colptr = similar(A.colptr,Ti′)
  rowval = similar(A.rowval,Ti′)
  nzvals = similar(A.data,Tv′)
  MatrixOfSparseMatricesCSC(A.m,A.n,colptr,rowval,nzvals)
end

function Base.copyto!(A::MatrixOfSparseMatricesCSC,B::MatrixOfSparseMatricesCSC)
  @check size(A) == size(B)
  copyto!(A.colptr,B.colptr)
  copyto!(A.rowval,B.rowval)
  copyto!(A.data,B.data)
  A
end

function Base.zero(A::MatrixOfSparseMatricesCSC{Tv,Ti}) where {Tv,Ti}
  MatrixOfSparseMatricesCSC(A.m,A.n,ones(Ti,length(A.colptr)),Ti[],Tv[])
end
