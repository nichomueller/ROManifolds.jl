struct ArrayOfArrays{T,N,L,M,P<:AbstractArray{T,M}} <: AbstractParamArray{T,N,L}
  data::P
  function ArrayOfArrays(data::P) where {T,M,P<:AbstractArray{T,M}}
    N = M - 1
    L = size(data,M)
    new{T,N,L,M,P}(data)
  end
end

const VectorOfVectors{T,L,P} = ArrayOfArrays{T,1,L,2,P}
const MatrixOfMatrices{T,L,P} = ArrayOfArrays{T,2,L,3,P}
const Tensor3DOfTensors3D{T,L,P} = ArrayOfArrays{T,3,L,4,P}

function ArrayOfArrays(A::AbstractVector{<:AbstractArray})
  B = ArrayOfSimilarArrays(A)
  ArrayOfArrays(B.data)
end

Base.:(==)(A::ArrayOfArrays,B::ArrayOfArrays) = (A.data == B.data)

Base.size(A::ArrayOfArrays{T,N}) where {T,N} = ntuple(_->param_length(A),Val(N))

param_data(A::ArrayOfArrays{T,N}) where {T,N} = eachslice(A.data,dims=N+1)
param_getindex(A::ArrayOfArrays,i::Integer) = diagonal_getindex(Val(true),A,i)

Base.@propagate_inbounds function Base.getindex(A::ArrayOfArrays{T,N},i::Vararg{Integer,N}) where {T,N}
  @boundscheck checkbounds(A,i...)
  iblock = first(i)
  diagonal_getindex(Val(all(i.==iblock)),A,iblock)
end

Base.@propagate_inbounds function diagonal_getindex(
  ::Val{true},
  A::ArrayOfArrays{T,N},
  iblock::Integer) where {T,N}

  view(A.data,ArraysOfArrays._ncolons(Val(N))...,iblock)
end

Base.@propagate_inbounds function diagonal_getindex(
  ::Val{false},
  A::ArrayOfArrays{T,N},
  iblock::Integer) where {T,N}

  zeros(T,innersize(A))
end

Base.@propagate_inbounds function Base.setindex!(A::ArrayOfArrays{T,N},v,i::Vararg{Integer,N}) where {T,N}
  @boundscheck checkbounds(A,i...)
  iblock = first(i)
  all(i.==iblock) && diagonal_setindex!(Val(true),A,v,iblock)
end

Base.@propagate_inbounds function diagonal_setindex!(
  ::Val{true},
  A::ArrayOfArrays{T,N},
  v,iblock::Integer) where {T,N}

  setindex!(A.data,v,ArraysOfArrays._ncolons(Val(N))...,iblock)
end

Base.@propagate_inbounds function diagonal_setindex!(::Val{false},A::ArrayOfArrays,v,iblock::Integer)
  @notimplemented
end

function Base.similar(A::ArrayOfArrays{T,N},::Type{<:AbstractArray{T′}}) where {T,T′,N}
  ArrayOfArrays(similar(A.data,T′))
end

function Base.copyto!(A::ArrayOfArrays,B::ArrayOfArrays)
  @check size(A) == size(B)
  copyto!(A.data,B.data)
  A
end
