"""
    struct ConsecutiveArrayOfArrays{T,N,L,P<:AbstractVector{<:AbstractArray{T,N}}} <: ParamArray{T,N,L} end

Represents conceptually a vector of arrays, but the entries are stored in
consecutive memory addresses. So in practice it simply wraps an AbstractArray,
with a parametric length equal to its last dimension

"""
struct ConsecutiveArrayOfArrays{T,N,L,M,P<:AbstractArray{T,M}} <: ParamArray{T,N,L}
  data::P
  function ConsecutiveArrayOfArrays(data::P) where {T,M,P<:AbstractArray{T,M}}
    N = M - 1
    L = size(data,M)
    new{T,N,L,M,P}(data)
  end
end

function ConsecutiveArrayOfArrays(data::AbstractArray{<:AbstractArray})
  ConsecutiveArrayOfArrays(stack(data))
end

const AbstractConsecutiveParamVector{T,L,A} = ConsecutiveArrayOfArrays{T,1,L,2,A}
const AbstractConsecutiveParamMatrix{T,L,A} = ConsecutiveArrayOfArrays{T,1,L,2,A}

const ConsecutiveVectorOfVectors{T,L} = ConsecutiveArrayOfArrays{T,1,L,2,Array{T,2}}
const ConsecutiveMatrixOfMatrices{T,L} = ConsecutiveArrayOfArrays{T,2,L,3,Array{T,3}}

Base.size(A::ConsecutiveArrayOfArrays{T,N}) where {T,N} = ntuple(_->param_length(A),Val{N}())

@inline function ArraysOfArrays.innersize(A::ConsecutiveArrayOfArrays{T,N}) where {T,N}
  ArraysOfArrays.front_tuple(size(A.data),Val{N}())
end

param_data(A::ConsecutiveArrayOfArrays{T,N}) where {T,N} = eachslice(A.data,dims=N+1)
param_getindex(A::ConsecutiveArrayOfArrays{T,N},i::Integer) where {T,N} = view(A.data,ArraysOfArrays._ncolons(Val{N}())...,i)

function param_entry(A::ConsecutiveArrayOfArrays{T,N},i::Vararg{Integer,N}) where {T,N}
  A.data[i...,:]
end

function Base.getindex(A::ConsecutiveArrayOfArrays{T,N},i::Vararg{Integer,N}) where {T,N}
  @boundscheck checkbounds(A,i...)
  iblock = first(i)
  if all(i.==iblock)
    getindex(A.data,ArraysOfArrays._ncolons(Val{N}())...,i...)
  else
    fill(zero(T),innersize(A))
  end
end

function Base.setindex!(A::ConsecutiveArrayOfArrays{T,N},v,i::Vararg{Integer,N}) where {T,N}
  @boundscheck checkbounds(A,i...)
  iblock = first(i)
  all(i.==iblock) && param_setindex!(A,v,iblock)
end

function param_setindex!(A::ConsecutiveArrayOfArrays{T,N},v,i::Integer) where {T,N}
  setindex!(A.data,v,ArraysOfArrays._ncolons(Val{N}())...,i)
end

function Base.similar(A::ConsecutiveArrayOfArrays{T,N},::Type{<:AbstractArray{T′}}) where {T,T′,N}
  ConsecutiveArrayOfArrays(similar(A.data,T′))
end

function Base.similar(A::ConsecutiveArrayOfArrays{T,N},::Type{<:AbstractArray{T′}},dims::Dims{N}) where {T,T′,N}
  ConsecutiveArrayOfArrays(similar(A.data,T′,dims))
end

function Base.copyto!(A::ConsecutiveArrayOfArrays,B::ConsecutiveArrayOfArrays)
  @check size(A) == size(B)
  copyto!(A.data,B.data)
  A
end

Base.:(==)(A::ConsecutiveArrayOfArrays,B::ConsecutiveArrayOfArrays) = A.data == B.data
Base.:(≈)(A::ConsecutiveArrayOfArrays,B::ConsecutiveArrayOfArrays) = A.data ≈ B.data

function (+)(A::ConsecutiveArrayOfArrays,B::ConsecutiveArrayOfArrays)
  AB = (+)(A.data,B.data)
  ConsecutiveArrayOfArrays(AB)
end

function (-)(A::ConsecutiveArrayOfArrays,B::ConsecutiveArrayOfArrays)
  AB = (-)(A.data,B.data)
  ConsecutiveArrayOfArrays(AB)
end

function (+)(A::ConsecutiveArrayOfArrays{T,N},b::AbstractArray{<:Number}) where {T,N}
  B = copy(A.data)
  @inbounds for i in param_eachindex(A)
    B[ArraysOfArrays._ncolons(Val{N}())...,i] += b
  end
  return B
end

function (-)(A::ConsecutiveArrayOfArrays{T,N},b::AbstractArray{<:Number}) where {T,N}
  B = copy(A.data)
  @inbounds for i in param_eachindex(A)
    B[ArraysOfArrays._ncolons(Val{N}())...,i] -= b
  end
  return B
end

function (+)(b::AbstractArray{<:Number},A::ConsecutiveArrayOfArrays{T,N}) where {T,N}
  (+)(A,b)
end

function (-)(b::AbstractArray{<:Number},A::ConsecutiveArrayOfArrays{T,N}) where {T,N}
  (-)((-)(A,b))
end

function (*)(A::ConsecutiveArrayOfArrays{T,N},b::Number) where {T,N}
  return ConsecutiveArrayOfArrays(A.data*b)
end

function (/)(A::ConsecutiveArrayOfArrays{T,N},b::Number) where {T,N}
  return ConsecutiveArrayOfArrays(A.data/b)
end

for f in (:(Base.maximum),:(Base.minimum))
  @eval begin
    $f(A::ConsecutiveArrayOfArrays) = $f(A.data)
    $f(g,A::ConsecutiveArrayOfArrays) = $f(g,A.data)
  end
end

function Base.fill!(A::ConsecutiveArrayOfArrays,z::Number)
  fill!(A.data,z)
  return A
end

function LinearAlgebra.rmul!(A::ConsecutiveArrayOfArrays,b::Number)
  rmul!(A.data,b)
  return A
end

function LinearAlgebra.axpy!(α::Number,A::ConsecutiveArrayOfArrays,B::ConsecutiveArrayOfArrays)
  @check size(A) == size(B)
  axpy!(α,A.data,B.data)
  return B
end

function param_view(A::ConsecutiveArrayOfArrays,i::Union{Integer,AbstractVector,Colon}...)
  ConsecutiveArrayOfArrays(view(A.data,i...,:))
end
