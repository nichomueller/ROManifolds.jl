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

all_data(A::ArrayOfTrivialArrays) = A.data
param_data(A::ArrayOfTrivialArrays) = fill(A.data,param_length(A))
param_getindex(A::ArrayOfTrivialArrays,i::Integer...) = diagonal_getindex(A,i...)
param_setindex!(A::ArrayOfTrivialArrays,v,i::Integer...) = diagonal_setindex!(A,v,i...)
param_view(A::ArrayOfTrivialArrays{T,N},i::Integer...) where {T,N} = view(A.data,ArraysOfArrays._ncolons(Val{N}())...)
param_entry(A::ArrayOfTrivialArrays,i::Integer...) = ParamNumber(fill(A.data[i...],A.plength))

Base.@propagate_inbounds function Base.getindex(A::ArrayOfTrivialArrays{T,N},i::Vararg{Integer,N}) where {T,N}
  @boundscheck checkbounds(A,i...)
  iblock = first(i)
  if all(i.==iblock)
    diagonal_getindex(A,iblock)
  else
    fill(zero(T),innersize(A))
  end
end

Base.@propagate_inbounds function diagonal_getindex(A::ArrayOfTrivialArrays{T,N},iblock::Integer) where {T,N}
  A.data
end

Base.@propagate_inbounds function Base.setindex!(A::ArrayOfTrivialArrays,v,i::Integer...)
  @boundscheck checkbounds(A,i...)
  iblock = first(i)
  all(i.==iblock) && diagonal_setindex!(A,v,iblock)
end

Base.@propagate_inbounds function diagonal_setindex!(A::ArrayOfTrivialArrays{T,N},v,iblock::Integer) where {T,N}
  setindex!(A.data,v,ArraysOfArrays._ncolons(Val{N}())...)
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

function Base.zero(A::ArrayOfTrivialArrays)
  ArrayOfTrivialArrays(zero(A.data),A.plength)
end
