struct MatrixOfMatrices{Tv,Ti<:Integer,P<:AbstractMatrix{Tv},L} <: AbstractParamArray{Tv,2,L}
  data::P
  colptr::Vector{Ti}
  function MatrixOfMatrices(data::P,colptr::Vector{Ti}) where {Tv,Ti,P<:AbstractMatrix{Tv}}
    L = length(colptr) - 1
    new{Tv,Ti,P,L}(data,colptr)
  end
end

function _get_colptr(A::AbstractVector{<:AbstractMatrix})
  _,n = innersize(A)
  l = length(A)
  collect(Int32,1:n:(l+1)*n)
end

function MatrixOfMatrices(A::AbstractVector{<:AbstractMatrix})
  B = ArrayOfSimilarArrays(A)
  colptr = _get_colptr(A)
  MatrixOfMatrices(B.data,colptr)
end

@inline function ArraysOfArrays.innersize(A::MatrixOfMatrices)
  (size(A.data,1),A.colptr[2]-A.colptr[1])
end

ArraysOfArrays.flatview(A::MatrixOfMatrices) = A.data

Base.size(A::MatrixOfMatrices) = (param_length(A),param_length(A))

function Base.show(io::IO,::MIME"text/plain",A::MatrixOfMatrices)
  println(io, "Block diagonal matrix of matrices, with the following structure: ")
  show(io,MIME("text/plain"),A[1])
end

param_data(A::MatrixOfMatrices) = map(i->param_getindex(A,i),param_eachindex(A))
param_getindex(a::MatrixOfMatrices,i::Integer) = getindex(a,i,i)

Base.@propagate_inbounds function Base.getindex(A::MatrixOfMatrices,i::Integer)
  irow = fast_index(i,size(A,1))
  icol = slow_index(i,size(A,1))
  getindex(A,irow,icol)
end

Base.@propagate_inbounds function Base.getindex(A::MatrixOfMatrices,irow::Integer,icol::Integer)
  diagonal_getindex(Val(irow==icol),A,irow)
end

Base.@propagate_inbounds function diagonal_getindex(
  ::Val{true},
  A::MatrixOfMatrices{T},
  iblock::Integer) where T

  view(A.data,:,A.colptr[iblock]:A.colptr[iblock+1]-1)
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

function Base.similar(A::MatrixOfMatrices{Tv},::Type{<:AbstractMatrix{Tv′}}) where {Tv,Tv′}
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
