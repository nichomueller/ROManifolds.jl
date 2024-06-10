struct ArrayOfArrays{T,N,L,M,P<:AbstractArray{T,M}} <: ParamArray{T,N,L}
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

Base.size(A::ArrayOfArrays{T,N}) where {T,N} = ntuple(_->param_length(A),Val{N}())

@inline function ArraysOfArrays.innersize(A::ArrayOfArrays{T,N}) where {T,N}
  ArraysOfArrays.front_tuple(size(A.data),Val{N}())
end

all_data(A::ArrayOfArrays) = A.data
param_data(A::ArrayOfArrays{T,N}) where {T,N} = eachslice(A.data,dims=N+1)
param_getindex(A::ArrayOfArrays,i::Integer) = diagonal_getindex(A,i)
param_setindex!(A::ArrayOfArrays,v,i::Integer) = diagonal_setindex!(A,v,i)
param_view(A::ArrayOfArrays{T,N},i::Integer) where {T,N} = view(A.data,ArraysOfArrays._ncolons(Val{N}())...,i)
param_entry(A::ArrayOfArrays{T,N},i::Vararg{Integer,N}) where {T,N} = ParamNumber(A.data[i...,:])

Base.@propagate_inbounds function Base.getindex(A::ArrayOfArrays{T,N},i::Vararg{Integer,N}) where {T,N}
  @boundscheck checkbounds(A,i...)
  iblock = first(i)
  if all(i.==iblock)
    diagonal_getindex(A,iblock)
  else
    fill(zero(T),innersize(A))
  end
end

Base.@propagate_inbounds function diagonal_getindex(A::ArrayOfArrays{T,N},iblock::Integer) where {T,N}
  getindex(A.data,ArraysOfArrays._ncolons(Val{N}())...,iblock)
end

Base.@propagate_inbounds function Base.setindex!(A::ArrayOfArrays{T,N},v,i::Vararg{Integer,N}) where {T,N}
  @boundscheck checkbounds(A,i...)
  iblock = first(i)
  all(i.==iblock) && diagonal_setindex!(A,v,iblock)
end

Base.@propagate_inbounds function diagonal_setindex!(A::ArrayOfArrays{T,N},v,iblock::Integer) where {T,N}
  setindex!(A.data,v,ArraysOfArrays._ncolons(Val{N}())...,iblock)
end

function Base.similar(A::ArrayOfArrays{T,N},::Type{<:AbstractArray{T′}}) where {T,T′,N}
  ArrayOfArrays(similar(A.data,T′))
end

function Base.similar(A::ArrayOfArrays{T,N},::Type{<:AbstractArray{T′}},dims::Dims{1}) where {T,T′,N}
  ArrayOfArrays(similar(A.data,T′,innersize(A)...,dims...))
end

function Base.copyto!(A::ArrayOfArrays,B::ArrayOfArrays)
  @check size(A) == size(B)
  copyto!(A.data,B.data)
  A
end

function Base.zero(A::ArrayOfArrays)
  ArrayOfArrays(zero(A.data))
end

function all_view(A::ArrayOfArrays{T,N},i::Union{Integer,AbstractVector,Colon}...) where {T,N}
  ArrayOfArrays(view(A.data,i...,:))
end
