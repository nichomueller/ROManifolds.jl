"""
    param_length(a) -> Int

Returns the parametric length of `a`

"""
param_length(a) = @abstractmethod
param_length(a::Union{Nothing,Function,Map}) = 0
param_length(a::Union{AbstractParamRealization,AbstractParamFunction}) = length(a)
param_length(a::Union{Number,AbstractArray{<:Number}}) = 0

"""
    param_data(a) -> Int

Returns the parametric data of `a`

"""
param_data(a) = @abstractmethod

"""
    param_getindex(a,i::Integer) -> Any

Returns the parametric entry of `a` at the index `i` ∈ {1,...,`param_length`(`a`)}

"""
param_getindex(a,i::Integer) = @abstractmethod
param_getindex(a::Union{Nothing,Function,Map},i::Integer) = a

param_setindex!(a,v,i::Integer) = @abstractmethod

"""
    param_entry(a,i::Integer...) -> AbstractVector{eltype(a)}

Same as getindex(a,i::Integer...), but across every parameter defining `a`. The
result is an abstract vector of length `param_length`(`a`). It often outputs a
[`ParamNumber`]

"""
param_entry(a,i::Integer...) = @abstractmethod

param_eachindex(a) = Base.OneTo(param_length(a))

"""
    array_of_similar_arrays(a,plength::Integer) -> AbstractArray{typeof(a),ndims(a)}

Creates an instance of `AbstractArray`{typeof(`a`),ndims(`a`)} of parametric
length `plength` from `a`.

"""
array_of_similar_arrays(a,plength::Integer) = @abstractmethod

function array_of_zero_arrays(a,plength::Integer)
  A = array_of_similar_arrays(a,plength)
  fill!(A,zero(eltype(a)))
  return A
end

"""
    array_of_consecutive_arrays(a,plength::Integer) -> AbstractArray{typeof(a),ndims(a)}

Like [`array_of_similar_arrays`](@ref), but the result has entries stored in
consecutive memory cells

"""
array_of_consecutive_arrays(a,plength::Integer) = @abstractmethod

function array_of_consecutive_zero_arrays(a,plength::Integer)
  A = array_of_consecutive_arrays(a,plength)
  fill!(A,zero(eltype(a)))
  return A
end

"""
    to_param_quantity(a,plength::Integer) -> Any

Returns a quantity with parametric length `plength` from `a`. When `a` already
possesses a parametric length, i.e. it is a parametrized quantity, it returns `a`

"""
to_param_quantity(a,plength::Integer) = @abstractmethod
to_param_quantity(a::Union{Nothing,Function,Map},plength::Integer) = a

"""
    find_param_length(a...) -> Int

Returns the parametric length of all parametric quantities. An error is thrown
if there are no parametric quantities or if at least two quantities have different
parametric length

"""
function find_param_length(a...)
  plengths::Tuple{Vararg{Int}} = filter(!iszero,param_length.(a))
  @check all(plengths .== first(plengths))
  return first(plengths)
end

"""
    to_param_quantities(a...;plength=find_param_length(a...)) -> Any

Converts the input quantities to parametric quantities

"""
function to_param_quantities(a...;plength=find_param_length(a...))
  pa = map(f->to_param_quantity(f,plength),a)
  return pa
end

"""
    abstract type AbstractParamContainer{T,N,L} <: AbstractArray{T,N} end

Type representing generic parametric quantities. L encodes the parametric length.
Subtypes:
- [`ParamContainer`](@ref).
- [`ParamNumber`](@ref).
- [`AbstractParamArray`](@ref).
- [`AbstractSnapshots`](@ref).

"""
abstract type AbstractParamContainer{T,N,L} <: AbstractArray{T,N} end

param_length(::Type{<:AbstractParamContainer{T,N,L}}) where {T,N,L} = L
param_length(::T) where {T<:AbstractParamContainer} = param_length(T)

param_data(A::AbstractParamContainer) = map(i->param_getindex(A,i),param_eachindex(A))

function to_param_quantity(a::AbstractParamContainer,plength::Integer)
  @check param_length(a) == plength
  return a
end

"""
    struct ParamContainer{T,L} <: AbstractArray{T,1,L} end

Used as a wrapper for non-array structures, e.g. factorizations

"""
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

"""
    struct ParamNumber{T<:Number,L} <: AbstractParamContainer{T,1,L} end

Represents parametric scalars, e.g. entries of parametric arrays across all
parameters.

"""
struct ParamNumber{T<:Number,L} <: AbstractParamContainer{T,1,L}
  data::Vector{T}
  ParamNumber(data::Vector{T}) where T<:Number = new{T,length(data)}(data)
end

param_getindex(a::ParamNumber,i::Integer) = getindex(a,i)
param_setindex!(a::ParamNumber,v,i::Integer) = setindex!(a,v,i)
param_entry(a::ParamNumber,i::Integer) = getindex(a,i)

to_param_quantity(a::Number,plength::Integer) = ParamNumber(fill(a,plength))

Base.size(a::ParamNumber) = (param_length(a),)
Base.getindex(a::ParamNumber,i::Integer) = getindex(a.data,i)
Base.setindex!(a::ParamNumber,v,i::Integer) = setindex!(a.data,v,i)

function array_of_similar_arrays(a::Union{Number,AbstractArray{<:Number,0}},l::Integer)
  ParamNumber(fill(zero(eltype(a)),l))
end

function array_of_consecutive_arrays(a::Union{Number,AbstractArray{<:Number,0}},l::Integer)
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
