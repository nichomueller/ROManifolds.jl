"""
    struct ArrayOfTrivialArrays{T,N,L,P<:AbstractArray{T,N}} <: ParamArray{T,N,L} end

Wrapper for nonparametric arrays that we wish assumed a parametric length.

"""
struct ArrayOfTrivialArrays{T,N,L,P<:AbstractArray{T,N}} <: ParamArray{T,N,L}
  data::P
  plength::Int
  function ArrayOfTrivialArrays(data::P,plength::Int) where {T<:Number,N,P<:AbstractArray{T,N}}
    new{T,N,plength,P}(data,plength)
  end
end

function ArrayOfTrivialArrays(A::AbstractArray{<:Number,N}) where N
  plength = 1
  ArrayOfTrivialArrays(A,plength)
end

function ArrayOfTrivialArrays(A::AbstractParamArray,args...)
  A
end

Base.size(A::ArrayOfTrivialArrays{T,N}) where {T,N} = ntuple(_ -> A.plength,Val{N}())

@inline function ArraysOfArrays.innersize(A::ArrayOfTrivialArrays)
  size(A.data)
end

@inline function inneraxes(A::ArrayOfTrivialArrays)
  axes(A.data)
end

param_data(A::ArrayOfTrivialArrays) = fill(A.data,param_length(A))
param_entry(A::ArrayOfTrivialArrays,i::Integer...) = fill(A.data[i...],A.plength)

Base.@propagate_inbounds function Base.getindex(A::ArrayOfTrivialArrays{T,N},i::Vararg{Integer,N}) where {T,N}
  @boundscheck checkbounds(A,i...)
  iblock = first(i)
  if all(i.==iblock)
    param_getindex(A,iblock)
  else
    fill(zero(T),innersize(A))
  end
end

Base.@propagate_inbounds function param_getindex(A::ArrayOfTrivialArrays{T,N},i::Integer) where {T,N}
  A.data
end

Base.@propagate_inbounds function Base.setindex!(A::ArrayOfTrivialArrays,v,i::Integer...)
  @boundscheck checkbounds(A,i...)
  iblock = first(i)
  all(i.==iblock) && param_setindex!(A,v,iblock)
end

Base.@propagate_inbounds function param_setindex!(A::ArrayOfTrivialArrays{T,N},v,i::Integer) where {T,N}
  copyto(A.data,v)
end

function Base.similar(A::ArrayOfTrivialArrays{T,N},::Type{<:AbstractArray{T′}}) where {T,T′,N}
  ArrayOfTrivialArrays(similar(A.data,T′),A.plength)
end

function Base.similar(A::ArrayOfTrivialArrays{T,N},::Type{<:AbstractArray{T′}},dims::Dims{1}) where {T,T′,N}
  ArrayOfTrivialArrays(similar(A.data,T′),dims...)
end

function Base.copyto!(A::ArrayOfTrivialArrays,B::ArrayOfTrivialArrays)
  @check size(A) == size(B)
  copyto!(B.data,A.data)
  A
end

function Arrays.setsize!(A::ArrayOfCachedArrays{T,N},s::NTuple{N,Int}) where {T,N}
  @inbounds for i in param_eachindex(A)
    setsize!(param_getindex(A,i),s)
  end
end
