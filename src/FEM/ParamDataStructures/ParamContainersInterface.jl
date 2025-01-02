"""
    param_length(a) -> Int

Returns the parametric length of `a`
"""
param_length(a) = @abstractmethod
param_length(a::Union{Nothing,Function,Map,Field,AbstractArray{<:Field}}) = 0
param_length(a::Union{AbstractRealization,AbstractParamFunction}) = length(a)
param_length(a::Union{Number,AbstractArray{<:Number}}) = 0
param_length(a::CellField) = param_length(testitem(get_data(a)))

"""
    get_param_data(a) -> Any

Returns the parametric data of `a`, usually in the form of a AbstractVector or
a NTuple
"""
get_param_data(a) = @abstractmethod

"""
    param_getindex(a,i::Integer) -> Any

Returns the parametric entry of `a` at the index `i` ∈ {1,...,`param_length(a)`}
"""
param_getindex(a,i::Integer) = @abstractmethod

"""
    param_setindex!(a,v,i::Integer) -> Any

Sets the parametric entry of `a` to `v` at the index `i` ∈ {1,...,`param_length(a)`}
"""
param_setindex!(a,v,i::Integer) = @abstractmethod

"""
    param_eachindex(a,i::Integer) -> Any

Returns the parametric range of `a` 1:`param_length(a)`
"""
param_eachindex(a) = Base.OneTo(param_length(a))

"""
    to_param_quantity(a,plength::Integer) -> Any

Returns a quantity with parametric length `plength` from `a`. When `a` already
possesses a parametric length, i.e. it is a parametrized quantity, it returns `a`
"""
to_param_quantity(a,plength::Integer) = @abstractmethod

function to_param_quantity(a::AbstractParamFunction,plength::Integer)
  @check param_length(a) == plength
  return a
end

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

param_typeof(a) = typeof(a)

"""
    abstract type AbstractParamContainer{T,N} <: AbstractArray{T,N} end

Type representing generic parametric quantities.
Subtypes:
- [`ParamContainer`](@ref)
- [`AbstractParamArray`](@ref)
- [`AbstractSnapshots`](@ref)
"""
abstract type AbstractParamContainer{T,N} <: AbstractArray{T,N} end

get_param_data(a::AbstractParamContainer) = (param_getindex(a,i) for i in param_eachindex(a))

function to_param_quantity(a::AbstractParamContainer,plength::Integer)
  @check param_length(a) == plength
  return a
end

"""
    param_typeof(a::AbstractParamContainer) -> Any

Returns a type-like structure thanks to which we can access the parametric
length of `a`` without `a` itself
"""
param_typeof(a::AbstractParamContainer) = ParamType{typeof(a),param_length(a)}

abstract type ParamType{T<:AbstractParamContainer,L} <: Core.Any end

const PType{T,L} = Union{ParamType{T,L},Type{ParamType{T,L}}}
Base.eltype(::PType{T,L}) where {T,L} = eltype(T)
param_length(::PType{T,L}) where {T,L} = L

"""
    struct ParamContainer{T,A<:AbstractVector{T}} <: AbstractParamContainer{T,1}
      data::A
    end

Used as a wrapper for non-array structures, e.g. factorizations or numbers
"""
struct ParamContainer{T,A<:AbstractVector{T}} <: AbstractParamContainer{T,1}
  data::A
end

ParamContainer(a::AbstractArray{<:Number}) = ParamNumber(a)
ParamContainer(a::AbstractArray{<:AbstractArray}) = ParamArray(a)

param_length(a::ParamContainer) = length(a.data)
param_getindex(a::ParamContainer,i::Integer) = getindex(a,i)
param_getindex(a::ParamContainer,v,i::Integer) = setindex!(a,v,i)

to_param_quantity(a::Union{Function,Map,Nothing},plength::Integer) = ParamContainer(Fill(a,plength))

Base.size(a::ParamContainer) = (param_length(a),)
Base.getindex(a::ParamContainer,i::Integer) = getindex(a.data,i)
Base.setindex!(a::ParamContainer,v,i::Integer) = setindex!(a.data,v,i)

const ParamNumber = ParamContainer

to_param_quantity(a::Number,plength::Integer) = ParamNumber(fill(a,plength))

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
