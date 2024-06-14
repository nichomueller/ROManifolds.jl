struct ArrayOfArrays{T,N,L,P<:AbstractVector{<:AbstractArray{T,N}}} <: ParamArray{T,N,L}
  data::P
  function ArrayOfArrays(data::P) where {T,N,P<:AbstractVector{<:AbstractArray{T,N}}}
    L = length(data)
    new{T,N,L,P}(data)
  end
end

const VectorOfVectors{T,L} = ArrayOfArrays{T,1,L,Vector{Vector{T}}}
const MatrixOfMatrices{T,L} = ArrayOfArrays{T,2,L,Vector{Matrix{T}}}
const Tensor3DOfTensors3D{T,L} = ArrayOfArrays{T,3,L,Vector{Array{T,3}}}
const ArrayOfCachedArrays{T,N,L,P<:AbstractVector{<:CachedArray{T,N}}} = ArrayOfArrays{T,N,L,P}

Base.size(A::ArrayOfArrays{T,N}) where {T,N} = ntuple(_->param_length(A),Val{N}())

@inline function ArraysOfArrays.innersize(A::ArrayOfArrays{T,N}) where {T,N}
  size(first(A.data))
end

param_data(A::ArrayOfArrays{T,N}) where {T,N} = A.data
param_getindex(A::ArrayOfArrays,i::Integer) = diagonal_getindex(A,i)
param_setindex!(A::ArrayOfArrays,v,i::Integer) = diagonal_setindex!(A,v,i)
param_entry(A::ArrayOfArrays{T,N},i::Vararg{Integer,N}) where {T,N} = ParamNumber(map(a->getindex(a,i...),A.data))

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
  getindex(A.data,iblock)
end

Base.@propagate_inbounds function Base.setindex!(A::ArrayOfArrays{T,N},v,i::Vararg{Integer,N}) where {T,N}
  @boundscheck checkbounds(A,i...)
  iblock = first(i)
  all(i.==iblock) && diagonal_setindex!(A,v,iblock)
end

Base.@propagate_inbounds function diagonal_setindex!(A::ArrayOfArrays{T,N},v,iblock::Integer) where {T,N}
  setindex!(A.data,v,iblock)
end

function Base.similar(A::ArrayOfArrays{T,N},::Type{<:AbstractArray{T′}}) where {T,T′,N}
  param_array(param_data(A)) do a
    similar(a,T′)
  end
end

function Base.similar(A::ArrayOfArrays{T,N},::Type{<:AbstractArray{T′}},dims::Dims{N}) where {T,T′,N}
  param_array(param_data(A)) do a
    similar(a,T′)
  end
end

function Base.copyto!(A::ArrayOfArrays,B::ArrayOfArrays)
  @check size(A) == size(B)
  map(copyto!,A.data,B.data)
  A
end

function all_view(A::ArrayOfArrays{T,N},i::Union{Integer,AbstractVector,Colon}...) where {T,N}
  ArrayOfArrays(map(a -> view.(a,i...),A.data))
end
