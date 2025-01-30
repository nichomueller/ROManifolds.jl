"""
    abstract type ParamField <: Field end

Represents a parametric field.
Subtypes:
- [`TrivialParamField`](@ref)
- [`GenericParamField`](@ref)
- [`ParamFieldGradient`](@ref)
- [`OperationParamField`](@ref)
"""
abstract type ParamField <: Field end

param_length(f::ParamField) = length(get_param_data(f))
param_getindex(f::ParamField,i::Integer) = getindex(get_param_data(f),i)

Base.length(f::ParamField) = param_length(f)
Base.size(f::ParamField) = (length(f),)
Base.axes(f::ParamField) = (Base.OneTo(length(f)),)
Base.eltype(f::ParamField) = typeof(testitem(f))
Base.getindex(f::ParamField,i::Integer) = param_getindex(f,i)
Base.iterate(f::ParamField,i...) = iterate(get_param_data(f),i...)

Arrays.testitem(f::ParamField) = param_getindex(f,1)
Arrays.testargs(f::ParamField,x::Point) = testargs(testitem(f),x)
Arrays.testargs(f::ParamField,x::AbstractArray{<:Point}) = testargs(testitem(f),x)
Arrays.return_value(b::Broadcasting{<:Function},f::ParamField,x...) = evaluate(b.f,f,x...)

to_param_quantity(f::ParamField,plength::Integer) = f
to_param_quantity(f::Union{Field,AbstractArray{<:Field}},plength::Integer) = TrivialParamField(f,plength)

"""
    struct TrivialParamField{F} <: ParamField
      data::F
      plength::Int
    end

Wrapper for a non-parametric field `data` that we wish assumed a parametric length `plength`
"""
struct TrivialParamField{F} <: ParamField
  data::F
  plength::Int
end

TrivialParamField(f::ParamField,plength::Int) = f

get_param_data(f::TrivialParamField) = Fill(f.data,f.plength)

Arrays.evaluate(f::TrivialParamField,x::Point) = fill(evaluate(f.data,x),f.plength)

"""
    struct GenericParamField{T<:GenericField} <: ParamField
      data::Vector{T}
    end

Parametric extension of a [`GenericField`](@ref) in [`Gridap`](@ref)
"""
struct GenericParamField{T<:GenericField} <: ParamField
  data::Vector{T}
end

Fields.GenericField(f::AbstractParamFunction) = GenericParamField(map(i -> GenericField(f[i]),1:length(f)))

get_param_data(f::GenericParamField) = f.data

Arrays.return_value(f::GenericParamField,x::Point) = fill(return_value(f.data[1],x),param_length(f))
Arrays.return_cache(f::GenericParamField,x::Point) = fill(return_cache(f.data[1],x),param_length(f))
Arrays.evaluate!(cache,f::GenericParamField,x::Point) = map(o->evaluate!(cache,o,x),f.data)

"""
    struct ParamFieldGradient{N,F} <: ParamField
      object::F
    end

Parametric extension of a [`FieldGradient`](@ref) in [`Gridap`](@ref)
"""
struct ParamFieldGradient{N,F} <: ParamField
  object::F
  ParamFieldGradient{N}(object::F) where {N,F} = new{N,F}(object)
end

Fields.FieldGradient{N}(f::ParamField) where N = ParamFieldGradient{N}(f)
Fields.FieldGradient{N}(f::ParamFieldGradient{N}) where N = ParamFieldGradient{N+1}(f.object)
get_param_data(f::ParamFieldGradient) = get_param_data(f.object)
param_getindex(f::ParamFieldGradient{N,<:ParamField},i::Integer) where N = FieldGradient{N}(param_getindex(f.object,i))

Arrays.return_value(f::ParamFieldGradient,x::Point) = evaluate(f,testargs(f,x)...)
Arrays.return_cache(f::ParamFieldGradient,x::Point) = nothing
Arrays.evaluate!(cache,f::ParamFieldGradient,x::Point) = @abstractmethod
Arrays.testvalue(::Type{ParamFieldGradient{N,T}}) where {N,T} = ParamFieldGradient{N}(testvalue(T))

function Arrays.return_cache(f::ParamFieldGradient{N,<:GenericParamField},x::Point) where N
  return_cache(ParamFieldGradient{N}(f.object.object),x)
end

function Arrays.evaluate!(c,f::ParamFieldGradient{N,<:GenericParamField},x::Point) where N
  evaluate!(c,ParamFieldGradient{N}(f.object.object),x)
end

"""
    struct OperationParamField{T<:Fields.OperationField} <: ParamField
      data::Vector{T}
    end

Parametric extension of a [`OperationField`](@ref) in [`Gridap`](@ref)
"""
struct OperationParamField{T<:Fields.OperationField} <: ParamField
  data::Vector{T}
end

function Fields.OperationField(op,fields::Tuple{Vararg{Any}})
  T = Union{ParamField,ParamContainer{<:Field}}
  if any(isa.(fields,T)) || isa(op,T)
    pop,pfields... = to_param_quantities(op,fields...)
    plength = find_param_length(pop,pfields...)
    data = map(1:plength) do i
      op = param_getindex(pop,i)
      field = map(f->param_getindex(f,i),pfields)
      Fields.OperationField(op,field)
    end
    OperationParamField(data)
  else
    Fields.OperationField{typeof(op),typeof(fields)}(op,fields)
  end
end

get_param_data(f::OperationParamField) = f.data

function Fields.gradient(f::OperationParamField)
  @notimplemented
end

# lazy maps

function Arrays.return_value(f::ParamField,x::AbstractArray{<:Point})
  fi = testitem(f)
  vi = return_value(fi,x)
  array = Vector{typeof(vi)}(undef,param_length(f))
  for i in param_eachindex(f)
    array[i] = return_value(param_getindex(f,i),x)
  end
  ParamArray(array)
end

function Arrays.return_cache(f::ParamField,x::AbstractArray{<:Point})
  fi = testitem(f)
  li = return_cache(fi,x)
  fix = evaluate!(li,fi,x)
  l = Vector{typeof(li)}(undef,param_length(f))
  g = param_array(fix,param_length(f))
  for i in param_eachindex(f)
    l[i] = return_cache(param_getindex(f,i),x)
  end
  l,g
end

function Arrays.evaluate!(cache,f::ParamField,x::AbstractArray{<:Point})
  l,g = cache
  @inbounds for i in param_eachindex(f)
    g[i] = evaluate!(l[i],param_getindex(f,i),x)
  end
  g
end

# used to correctly deal with parametric FEFunctions

function Arrays.return_value(
  k::Broadcasting{<:Operation},
  args::Union{Field,ParamContainer{<:Field}}...)

  pargs = to_param_quantities(args...)
  Fields.OperationField(k.f.op,pargs)
end

function Arrays.evaluate!(
  cache,
  k::Broadcasting{<:Operation},
  args::Union{Field,ParamContainer{<:Field}}...)

  pargs = to_param_quantities(args...)
  Fields.OperationField(k.f.op,pargs)
end
