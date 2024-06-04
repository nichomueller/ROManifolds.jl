struct VectorOfVectors{T,P<:AbstractMatrix{T},L} <: AbstractParamArray{T,1,L}
  data::P
  function VectorOfVectors(data::P) where {T,P<:AbstractMatrix{T}}
    L = size(data,2)
    new{T,P,L}(data)
  end
end

function VectorOfVectors(A::AbstractVector{<:AbstractVector}) where T
  B = ArrayOfSimilarArrays(A)
  VectorOfVectors(B.data)
end

@inline function ArraysOfArrays.innersize(A::VectorOfVectors)
  (size(A.data,1),)
end

Base.:(==)(A::VectorOfVectors,B::VectorOfVectors) = (A.data == B.data)

ArraysOfArrays.flatview(A::VectorOfVectors) = A.data

Base.size(A::VectorOfVectors) = (param_length(A),)

function Base.show(io::IO,::MIME"text/plain",A::VectorOfVectors)
  println(io, "Parametric collection of vectors, with the following eltypes: ")
  show(io,MIME("text/plain"),A[1])
end

param_data(A::VectorOfVectors) = eachcol(A.data)
param_getindex(A::VectorOfVectors,i::Integer) = getindex(A,i)

Base.@propagate_inbounds function Base.getindex(A::VectorOfVectors,i::Integer)
  view(A.data,:,i)
end

Base.@propagate_inbounds function Base.setindex!(A::VectorOfVectors,v,i::Integer...)
  A[i...] = v
  A
end

function Base.similar(A::VectorOfVectors,::Type{<:AbstractVector{T′}})
  VectorOfVectors(similar(A.data,T′))
end

function Base.copyto!(A::VectorOfVectors,B::VectorOfVectors)
  @check size(A) == size(B)
  copyto!(A.data,B.data)
  A
end
