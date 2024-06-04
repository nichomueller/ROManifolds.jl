struct ArrayOfTrivialArrays{P,N} <: AbstractArrayOfSimilarArrays{P,N,N}
  data::P
  outersize::NTuple{N,Int}
  function ArrayOfTrivialArrays(data::P,outersize::NTuple{N,Int}) where {T<:Number,N,P<:AbstractArray{T,N}}
    new{P,N}(data,outersize)
  end
end

function ArrayOfTrivialArrays(A::AbstractArray{<:Number,N},plength::Int=1) where N
  ArrayOfTrivialArrays(A,ntuple(_ -> plength,Val(N)))
end

function ArrayOfTrivialArrays(A::AbstractArrayOfSimilarArrays,args...)
  A
end

@inline function ArraysOfArrays.innersize(A::ArrayOfTrivialArrays)
  size(A.data)
end

Base.size(A::ArrayOfTrivialArrays) = A.outersize

Base.@propagate_inbounds function Base.getindex(A::ArrayOfTrivialArrays,i::Integer...)
  @boundscheck checkbounds(A,i...)
  A.data
end

Base.@propagate_inbounds function Base.setindex!(A::ArrayOfTrivialArrays,v,i::Integer...)
  A[i...] = v
  A
end

function Base.push!(A::ArrayOfTrivialArrays{P,N},B::AbstractArray{T,N}) where {P,T,N}
  @check A.data == B
  ArrayOfTrivialArrays(A.data,A.outersize .+ 1)
end

function Base.pushfirst!(A::ArrayOfTrivialArrays{P,N},B::AbstractArray{T,N}) where {P,T,N}
  @check A.data == B
  ArrayOfTrivialArrays(A.data,A.outersize .+ 1)
end

function Base.similar(A::ArrayOfTrivialArrays{P,N},::Type{<:AbstractArray{T,N}},dims::Dims) where {P,T,N}
  ArrayOfTrivialArrays(similar(A.data,T,dims...),A.outersize)
end

function Base.copyto!(A::ArrayOfTrivialArrays,B::ArrayOfTrivialArrays)
  @check size(A) == size(B)
  copyto!(B.data,A.data)
  A
end
