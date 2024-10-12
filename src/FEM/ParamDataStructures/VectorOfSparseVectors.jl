"""
    struct VectorOfSparseVectors{Tv,Ti<:Integer,L} <: ParamSparseMatrixCSC{Tv,Ti,L} end

Represents a vector of sparse vectors

"""
struct VectorOfSparseVectors{Tv,Ti<:Integer,L} <: ParamArray{Tv,1,L}
  n::Int64
  nzind::Vector{Ti}
  data::Matrix{Tv}
  function VectorOfSparseVectors(n::Int64,nzind::Vector{Ti},data::Matrix{Tv}) where {Tv,Ti}
    L = size(data,2)
    new{Tv,Ti,L}(n,nzind,data)
  end
end

SparseArrays.nonzeros(A::VectorOfSparseVectors) = A.data
_nonzeros(A::VectorOfSparseVectors,i::Integer) = @inbounds getindex(A.data,:,i)

function VectorOfSparseVectors(A::AbstractVector{SparseVector{Tv,Ti}}) where {Tv,Ti}
  n, = innersize(A)
  nzind,data = innerpattern(A)
  B = ArrayOfSimilarArrays(data)
  VectorOfSparseVectors(n,nzind,B.data)
end

function innerpattern(A::AbstractVector{<:SparseVector})
  @check !isempty(A)
  nzind = first(A).nzind
  if any(A->A.nzind != nzind, A)
    throw(DimensionMismatch("Sparsity pattern of element vectors is not equal"))
  end
  data = map(nonzeros,A)
  return nzind,data
end

Base.size(A::VectorOfSparseVectors) = (param_length(A),)

@inline function ArraysOfArrays.innersize(A::VectorOfSparseVectors)
  (A.n,)
end

@inline function inneraxes(A::VectorOfSparseVectors)
  Base.OneTo.(innersize(A))
end

param_entry(A::VectorOfSparseVectors,i::Integer) = A.data[i,:]

function Base.getindex(A::VectorOfSparseVectors,i::Integer)
  @boundscheck checkbounds(A,i)
  SparseVector(A.n,A.nzind,_nonzeros(A,i))
end

function param_getindex(A::VectorOfSparseVectors,i::Integer)
  SparseVector(A.n,A.nzind,_nonzeros(A,i))
end

function Base.setindex!(A::VectorOfSparseVectors,v,i::Integer)
  @boundscheck checkbounds(A,i)
  setindex!(_nonzeros(A,i),v)
end

function Base.similar(
  A::VectorOfSparseVectors{Tv,Ti},
  ::Type{<:SparseVector{Tv′,Ti′}}
  ) where {Tv,Ti,Tv′,Ti′}

  nzind = similar(A.nzind,Ti′)
  nzvals = similar(A.data,Tv′)
  VectorOfSparseVectors(A.n,nzind,nzvals)
end

function Base.copyto!(A::VectorOfSparseVectors,B::VectorOfSparseVectors)
  @check size(A) == size(B)
  copyto!(A.nzind,B.nzind)
  copyto!(A.data,B.data)
  A
end

function LinearAlgebra.fillstored!(A::VectorOfSparseVectors,z::Number)
  fill!(A.data,z)
  return A
end

# small hack, we shouldn't be able to fill an abstract array with a non-scalar
function LinearAlgebra.fillstored!(A::VectorOfSparseVectors,z::AbstractVector{<:Number})
  @check all(z.==first(z))
  LinearAlgebra.fillstored!(A,first(z))
  return A
end
