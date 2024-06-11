param_length(a) = @abstractmethod
param_length(a::Union{Nothing,Function,Map}) = 0
param_length(a::AbstractParamFunction) = length(a)
param_length(a::Union{Number,AbstractArray{<:Number}}) = 0
param_data(a) = @abstractmethod
all_data(a) = @abstractmethod
param_getindex(a,i::Integer...) = @abstractmethod
param_getindex(a::Union{Nothing,Function,Map},i::Integer...) = a
param_setindex!(a,v,i::Integer...) = @abstractmethod
param_view(a,i::Integer...) = @abstractmethod
param_entry(a,i::Integer...) = @abstractmethod
param_eachindex(a) = Base.OneTo(param_length(a))
array_of_similar_arrays(a,l::Integer) = @abstractmethod
to_param_quantity(a,plength::Integer) = @abstractmethod
to_param_quantity(a::Union{Nothing,Function,Map},plength::Integer) = a

function find_param_length(a...)
  plengths::Tuple{Vararg{Int}} = filter(!iszero,param_length.(a))
  @check all(plengths .== first(plengths))
  return first(plengths)
end

function to_param_quantities(a...)
  plength = find_param_length(a...)
  pa = map(f->to_param_quantity(f,plength),a)
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

ParamContainer(a::AbstractArray{<:Number}) = ParamNumber(a)
ParamContainer(a::AbstractArray{<:AbstractArray}) = ParamArray(a)

param_getindex(a::ParamContainer,i::Integer) = getindex(a,i)
param_getindex(a::ParamContainer,v,i::Integer) = setindex!(a,v,i)

Base.size(a::ParamContainer) = (param_length(a),)
Base.getindex(a::ParamContainer,i::Integer) = getindex(a.data,i)
Base.setindex!(a::ParamContainer,v,i::Integer) = setindex!(a.data,v,i)

struct ParamNumber{T<:Number,L} <: AbstractParamContainer{T,1,L}
  data::Vector{T}
  ParamNumber(data::Vector{T}) where T<:Number = new{T,length(data)}(data)
end

all_data(a::ParamNumber) = a.data
param_getindex(a::ParamNumber,i::Integer) = getindex(a,i)
param_setindex!(a::ParamNumber,v,i::Integer) = setindex!(a,v,i)
param_view(a::ParamNumber,i::Integer) = getindex(a,i)
param_entry(a::ParamNumber,i::Integer) = getindex(a,i)

to_param_quantity(a::Number,plength::Integer) = ParamNumber(fill(a,plength))
to_param_quantity(a::ParamNumber,plength::Integer) = a

Base.size(a::ParamNumber) = (param_length(a),)
Base.getindex(a::ParamNumber,i::Integer) = getindex(a.data,i)
Base.setindex!(a::ParamNumber,v,i::Integer) = setindex!(a.data,v,i)

function array_of_similar_arrays(a::Union{Number,AbstractArray{<:Number,0}},l::Integer)
  ParamNumber(fill(zero(eltype(a)),l))
end

for op in (:+,:-)
  @eval begin
    function ($op)(A::ParamNumber,b::Number)
      ParamNumber(Base.broadcast($op,A,b))
    end

    function ($op)(a::Number,B::ParamNumber)
      ParamNumber(Base.broadcast($op,a,B))
    end
  end
end
