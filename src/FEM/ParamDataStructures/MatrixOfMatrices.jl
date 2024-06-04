abstract type AbstractMatrixOfMatrices{T} <: AbstractArrayOfSimilarArrays{T,2,2} end

struct MatrixOfMatrices{Tv,Ti,P<:AbstractVector{<:AbstractMatrix{Tv}}} <: AbstractMatrixOfMatrices{Tv,2,2}
  data::P
  colptr::Vector{Ti}
end

function MatrixOfMatrices(A::AbstractVector{<:AbstractMatrix})
  _,n = innersize(A)
  l = length(A)
  colptr = collect(Int32,1:n:(l+1)*n)
  data = reduce(hcat,A)
  MatrixOfMatrices(data,colptr)
end

function ArraysOfArrays.ArrayOfSimilarArrays(A::AbstractVector{<:AbstractMatrix})
  MatrixOfMatrices(A)
end

@inline function ArraysOfArrays.innersize(A::MatrixOfMatrices)
  (size(A.data,1),A.colptr[2]-A.colptr[1])
end

ArraysOfArrays.flatview(A::MatrixOfMatrices) = A.data

Base.size(A::MatrixOfMatrices) = (length(A.colptr),length(A.colptr))

function Base.show(io::IO,::MIME"text/plain",A::MatrixOfMatrices)
  println(io, "Block diagonal matrix of matrices, with the following structure: ")
  show(io,MIME("text/plain"),A[1])
end

Base.@propagate_inbounds function Base.getindex(A::MatrixOfMatrices,iblock::Integer)
  irow = fast_index(iblock,size(A,1))
  icol = slow_index(iblock,size(A,1))
  getindex(A,irow,icol)
end

Base.@propagate_inbounds function Base.getindex(A::MatrixOfMatrices,irow::Integer,icol::Integer)
  diagonal_getindex(Val(irow==icol),A,irow)
end

Base.@propagate_inbounds function diagonal_getindex(
  ::Val{true},
  A::MatrixOfMatrices{T},
  iblock::Integer) where T

  A.data[:,A.colptr[iblock]:A.colptr[iblock+1]-1]
end

Base.@propagate_inbounds function diagonal_getindex(
  ::Val{false},
  A::MatrixOfMatrices{T},
  iblock::Integer) where T

  zeros(T,innersize(A))
end

Base.@propagate_inbounds function Base.setindex!(A::MatrixOfMatrices,v,i::Integer...)
  A[i...] = v
  A
end

function Base.similar(A::MatrixOfMatrices)
  data = similar(A.data)
  colptr = similar(A.colptr)
  MatrixOfMatrices(data,colptr)
end

function Base.similar(
  A::MatrixOfMatrices{Tv},
  ::Type{<:AbstractMatrix{Tv′}}
  ) where {Tv,Tv′}

  data = similar(A.data,Tv′)
  colptr = similar(A.colptr)
  MatrixOfMatrices(data,colptr)
end

function Base.copyto!(A::MatrixOfMatrices,B::MatrixOfMatrices)
  @check size(A) == size(B)
  copyto!(A.data,B.data)
  copyto!(A.colptr,B.colptr)
  A
end
