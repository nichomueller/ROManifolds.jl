struct ArrayOfTrivialArrays{T,N,L,P<:AbstractArray{T,N}} <: AbstractParamArray{T,N,L}
  data::P
  outersize::NTuple{N,Int}
  function ArrayOfTrivialArrays(data::P,outersize::NTuple{N,Int}) where {T<:Number,N,P<:AbstractArray{T,N}}
    @check all(outersize.==first(outersize))
    L = first(outersize)
    new{T,N,L,P}(data,outersize)
  end
end

function ArrayOfTrivialArrays(A::AbstractArray{<:Number,N},plength::Int=1) where N
  ArrayOfTrivialArrays(A,ntuple(_ -> plength,Val(N)))
end

function ArrayOfTrivialArrays(A::AbstractParamArray,args...)
  A
end

@inline function ArraysOfArrays.innersize(A::ArrayOfTrivialArrays)
  size(A.data)
end

Base.size(A::ArrayOfTrivialArrays) = A.outersize

param_data(A::ArrayOfTrivialArrays) = map(i->param_getindex(A,i),param_eachindex(A))
param_getindex(a::ArrayOfTrivialArrays,i::Integer...) = getindex(a,i...)

Base.@propagate_inbounds function Base.getindex(A::ArrayOfTrivialArrays,i::Integer...)
  @boundscheck checkbounds(A,i...)
  A.data
end

Base.@propagate_inbounds function Base.setindex!(A::ArrayOfTrivialArrays,v,i::Integer...)
  @boundscheck checkbounds(A,i...)
  setindex!(A.data,v,i...)
end

Base.:(==)(A::ArrayOfTrivialArrays,B::ArrayOfTrivialArrays) = (A.data == B.data)

function Base.similar(A::ArrayOfTrivialArrays{T,N},::Type{<:AbstractArray{T′,N}}) where {T,T′,N}
  ArrayOfTrivialArrays(similar(A.data,T′),A.outersize)
end

function Base.copyto!(A::ArrayOfTrivialArrays,B::ArrayOfTrivialArrays)
  @check size(A) == size(B)
  copyto!(B.data,A.data)
  A
end
