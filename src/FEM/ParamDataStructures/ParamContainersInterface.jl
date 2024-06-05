param_length(a) = @abstractmethod
param_length(a::Union{Function,Map}) = 0
param_length(a::AbstractParamFunction) = length(a)
param_length(a::AbstractArray) = 0
param_data(a) = @abstractmethod
param_getindex(a,i::Integer...) = @abstractmethod
param_eachindex(a) = Base.OneTo(param_length(a))
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

Base.size(a::ParamContainer) = (param_length(a),)
Base.getindex(a::ParamContainer,i::Integer) = getindex(a.data,i)

struct VectorOfScalars{T<:Number,L} <: AbstractParamContainer{T,1,L}
  data::Vector{T}
  VectorOfScalars(data::Vector{T}) where T = new{T,length(data)}(data)
end

param_getindex(a::VectorOfScalars,i::Integer) = getindex(a,i)

Base.size(a::VectorOfScalars) = (param_length(a),)
Base.getindex(a::VectorOfScalars,i::Integer) = getindex(a.data,i)
Base.setindex!(a::VectorOfScalars,v,i::Integer) = setindex!(a.data,v,i)
