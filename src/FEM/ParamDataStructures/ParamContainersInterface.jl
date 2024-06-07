param_length(a) = @abstractmethod
param_length(a::Union{Function,Map}) = 0
param_length(a::AbstractParamFunction) = length(a)
param_length(a::Union{Number,AbstractArray{<:Number}}) = 0
param_data(a) = @abstractmethod
all_data(a) = @abstractmethod
param_getindex(a,i::Integer...) = @abstractmethod
param_setindex!(a,v,i::Integer...) = @abstractmethod
param_view(a,i::Integer...) = @abstractmethod
param_eachindex(a) = Base.OneTo(param_length(a))
array_of_similar_arrays(a,l::Integer) = @abstractmethod
_to_param_quantity(a,plength::Integer) = @abstractmethod
_to_param_quantity(a::Union{Function,Map},plength::Integer) = a

function _find_param_length(a...)
  plengths::Tuple{Vararg{Int}} = filter(!iszero,param_length.(a))
  @check all(plengths .== first(plengths))
  return first(plengths)
end

function _to_param_quantities(a...)
  plength = _find_param_length(a...)
  pa = map(f->_to_param_quantity(f,plength),a)
  return pa
end

abstract type AbstractParamContainer{T,N,L} <: AbstractArray{T,N} end

param_length(::Type{<:AbstractParamContainer{T,N,L}}) where {T,N,L} = L
param_length(::T) where {T<:AbstractParamContainer} = param_length(T)

param_data(A::AbstractParamContainer) = map(i->param_getindex(A,i),param_eachindex(A))

struct ParamContainer{T,L} <: AbstractParamContainer{T,1,L}
  data::Vector{T}
  ParamContainer(data::Vector{T}) where T = new{T,length(data)}(data)
end

ParamContainer(a::AbstractArray{<:Number}) = VectorOfScalars(a)
ParamContainer(a::AbstractArray{<:AbstractArray}) = ParamArray(a)

param_getindex(a::ParamContainer,i::Integer) = getindex(a,i)
param_getindex(a::ParamContainer,v,i::Integer) = setindex!(a,v,i)

Base.size(a::ParamContainer) = (param_length(a),)
Base.getindex(a::ParamContainer,i::Integer) = getindex(a.data,i)
Base.setindex!(a::ParamContainer,v,i::Integer) = setindex!(a.data,v,i)

struct VectorOfScalars{T<:Number,L} <: AbstractParamContainer{T,1,L}
  data::Vector{T}
  VectorOfScalars(data::Vector{T}) where T = new{T,length(data)}(data)
end

all_data(a::VectorOfScalars) = a.data
param_getindex(a::VectorOfScalars,i::Integer) = getindex(a,i)
param_setindex!(a::VectorOfScalars,v,i::Integer) = setindex!(a,v,i)
param_view(a::VectorOfScalars,i::Integer) = getindex(a,i)

_to_param_quantity(a::VectorOfScalars,plength::Integer) = a

Base.size(a::VectorOfScalars) = (param_length(a),)
Base.getindex(a::VectorOfScalars,i::Integer) = getindex(a.data,i)
Base.setindex!(a::VectorOfScalars,v,i::Integer) = setindex!(a.data,v,i)

function array_of_similar_arrays(a::Union{Number,AbstractArray{<:Number,0}},l::Integer)
  VectorOfScalars(fill(zero(eltype(a)),l))
end
