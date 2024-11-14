struct ParamVectorWithEntryRemoved{T,A} <: ParamVector{T}
  a::A
  index::Int
  function ParamVectorWithEntryRemoved(a::A,index::Integer) where {T,A<:ConsecutiveParamVector{T}}
    @assert 1 <= index <= innerlength(a)
    new{T,A}(a,index)
  end
end

function Arrays.VectorWithEntryRemoved(a::AbstractParamVector,index::Int)
  ParamVectorWithEntryRemoved(a,index)
end

param_length(v::ParamVectorWithEntryRemoved) = param_length(v.a)

Base.size(v::ParamVectorWithEntryRemoved) = size(v.a)

function Base.getindex(v::ParamVectorWithEntryRemoved,i::Integer)
  VectorWithEntryRemoved(v.a[i],v.index)
end

function Base.sum(v::ParamVectorWithEntryRemoved)
  data = get_all_data(v.a)
  sum(data,dims=1) - data[v.index,:]
end

struct ParamVectorWithEntryInserted{T,A} <: ParamVector{T}
  a::A
  index::Int
  value::Vector{T}
  function ParamVectorWithEntryInserted(a::A,index::Integer,value::Vector{T}) where {T,A<:ConsecutiveParamVector{T}}
    @assert 1 <= index <= innerlength(a)+1
    new{T,A}(a,index,value)
  end
end

function Arrays.VectorWithEntryInserted(a::AbstractParamVector,index::Int,value::AbstractVector)
  ParamVectorWithEntryInserted(a,index,value)
end

param_length(v::ParamVectorWithEntryInserted) = param_length(v.a)

Base.size(v::ParamVectorWithEntryInserted) = size(v.a)

function Base.getindex(v::ParamVectorWithEntryInserted,i::Integer)
  VectorWithEntryInserted(v.a[i],v.index,v.value[i])
end

function Base.sum(v::ParamVectorWithEntryInserted)
  data = get_all_data(v.a)
  sum(data,dims=1) + v.value
end
