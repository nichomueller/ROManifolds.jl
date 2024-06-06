struct ArrayOfTrivialArrays{T,N,L,P<:AbstractArray{T,N}} <: ParamArray{T,N,L}
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

Base.size(A::ArrayOfTrivialArrays) = A.outersize

@inline function ArraysOfArrays.innersize(A::ArrayOfTrivialArrays)
  size(A.data)
end

param_data(A::ArrayOfTrivialArrays) = fill(A.data,param_length(A))
param_getindex(a::ArrayOfTrivialArrays,i::Integer...) = getindex(a,i...)

Base.@propagate_inbounds function Base.getindex(A::ArrayOfTrivialArrays{T,N},i::Integer...) where {T,N}
  @boundscheck checkbounds(A,i...)
  view(A.data,ArraysOfArrays._ncolons(Val(N))...)
end

Base.@propagate_inbounds function Base.setindex!(A::ArrayOfTrivialArrays,v,i::Integer...)
  @boundscheck checkbounds(A,i...)
  setindex!(A.data,v,i...)
end

function Base.similar(A::ArrayOfTrivialArrays{T,N},::Type{<:AbstractArray{T′}}) where {T,T′,N}
  ArrayOfTrivialArrays(similar(A.data,T′),A.outersize)
end

function Base.similar(A::ArrayOfTrivialArrays{T,N},::Type{<:AbstractArray{T′}},dims::Dims{1}) where {T,T′,N}
  ArrayOfTrivialArrays(similar(A.data,T′),dims...)
end

function Base.copyto!(A::ArrayOfTrivialArrays,B::ArrayOfTrivialArrays)
  @check size(A) == size(B)
  copyto!(B.data,A.data)
  A
end

function Base.zero(A::ArrayOfTrivialArrays)
  ArrayOfTrivialArrays(zero(A.data),A.outersize)
end
