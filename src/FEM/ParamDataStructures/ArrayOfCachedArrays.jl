struct ArrayOfCachedArrays{T,N,L,M} <: ParamCachedArray{T,N,L}
  data::CachedArray{T,M}
  function ArrayOfCachedArrays(data::CachedArray{T,M}) where {T,M}
    N = M - 1
    L = size(data,M)
    new{T,N,L,M}(data)
  end
end

const VectorOfCachedVectors{T,L} = ArrayOfCachedArrays{T,1,L,2}
const MatrixOfCachedMatrices{T,L} = ArrayOfCachedArrays{T,2,L,3}

function Arrays.CachedArray(A::AbstractParamArray)
  ArrayOfCachedArrays(CachedArray(all_data(A)))
end

function ArrayOfCachedArrays(A::AbstractVector{<:CachedArray})
  B = ArrayOfSimilarArrays(A)
  Bcache = CachedArray(B.data)
  ArrayOfArrays(Bcache)
end

Base.size(A::ArrayOfCachedArrays{T,N}) where {T,N} = ntuple(_->param_length(A),Val{N}())

@inline function ArraysOfArrays.innersize(A::ArrayOfCachedArrays{T,N}) where {T,N}
  ArraysOfArrays.front_tuple(size(A.data),Val{N}())
end

all_data(A::ArrayOfCachedArrays) = A.data
param_getindex(A::ArrayOfCachedArrays,i::Integer) = diagonal_getindex(A,i)
param_setindex!(A::ArrayOfCachedArrays,v,i::Integer) = diagonal_setindex!(A,v,i)

Base.@propagate_inbounds function Base.getindex(A::ArrayOfCachedArrays{T,N},i::Vararg{Integer,N}) where {T,N}
  @boundscheck checkbounds(A,i...)
  iblock = first(i)
  if all(i.==iblock)
    diagonal_getindex(A,iblock)
  else
    CachedArray(fill(zero(T),innersize(A)))
  end
end

Base.@propagate_inbounds function diagonal_getindex(A::ArrayOfCachedArrays{T,N},iblock::Integer) where {T,N}
  CachedArray(getindex(A.data,ArraysOfArrays._ncolons(Val{N}())...,iblock))
end

Base.@propagate_inbounds function Base.setindex!(A::ArrayOfCachedArrays{T,N},v,i::Vararg{Integer,N}) where {T,N}
  @boundscheck checkbounds(A,i...)
  iblock = first(i)
  all(i.==iblock) && diagonal_setindex!(A,v,iblock)
end

Base.@propagate_inbounds function diagonal_setindex!(A::ArrayOfCachedArrays{T,N},v,iblock::Integer) where {T,N}
  setindex!(A.data,v,ArraysOfArrays._ncolons(Val{N}())...,iblock)
end

function Base.copyto!(A::ArrayOfCachedArrays,B::ArrayOfCachedArrays)
  @check size(A) == size(B)
  copyto!(A.data,B.data)
  A
end

function Arrays.setsize!(A::ArrayOfCachedArrays{T,N},s::NTuple{N,Int}) where {T,N}
  setsize!(A.data,(s...,param_length(A)))
end
