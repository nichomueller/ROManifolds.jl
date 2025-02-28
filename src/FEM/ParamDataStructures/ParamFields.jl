"""
    abstract type ParamField <: Field end

Represents a parametric field.
Subtypes:
- [`TrivialParamField`](@ref)
- [`GenericParamField`](@ref)
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
# Arrays.testargs(f::ParamField,x::Point) = testargs(testitem(f),x)
# Arrays.testargs(f::ParamField,x::AbstractArray{<:Point}) = testargs(testitem(f),x)

to_param_quantity(f::ParamField,plength::Integer) = f
to_param_quantity(f::Union{Field,AbstractArray{<:Field}},plength::Integer) = TrivialParamField(f,plength)
parameterize(f::ParamField,plength::Integer) = f

function Base.zero(::Type{<:ParamField})
  @notimplemented "Must provide a parametric length"
end

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

Fields.gradient(f::ParamField) = TrivialParamField(gradient(f.data),f.plength)
Fields.∇∇(f::ParamField) = TrivialParamField(Fields.∇∇(f.data),f.plength)

"""
    struct GenericParamField{F<:GenericField} <: ParamField
      data::Vector{F}
    end

Wrapper for a generic-parametric field
"""
struct GenericParamField{F<:Field} <: ParamField
  data::Vector{F}
end

get_param_data(f::GenericParamField) = f.data

Fields.gradient(f::GenericParamField) = GenericParamField(map(gradient,f.field))
Fields.∇∇(f::GenericParamField) = GenericParamField(map(∇∇,f.field))

Fields.GenericField(f::AbstractParamFunction) = GenericParamField(map(i -> GenericField(f[i]),1:length(f)))

function _is_any_param(op,fields)
  T = Union{ParamField,ParamContainer{<:Field}}
  isa(op,T) && return true
  for f in fields
    isa(f,T) && return true
  end
  return false
end

function Fields.OperationField(op,fields::Tuple{Vararg{Any}})
  opfield(op,fields) = Fields.OperationField{typeof(op),typeof(fields)}(op,fields)
  if _is_any_param(op,fields)
    pop,pfields... = to_param_quantities(op,fields...)
    plength = find_param_length(pop,pfields...)
    T = typeof(opfield(testitem(pop),map(testitem,pfields)))
    data = Vector{T}(undef,plength)
    for i in 1:plength
      op = param_getindex(pop,i)
      field = map(f->param_getindex(f,i),pfields)
      data[i] = opfield(op,field)
    end
    GenericParamField(data)
  else
    opfield(op,fields)
  end
end

for T in (:(Fields.ZeroField),:(Fields.ConstantField),:(Fields.VoidField),:(Fields.InverseField))
  @eval begin
    $T(f::TrivialParamField) = TrivialParamField($T(f.data),f.plength)
    $T(f::GenericParamField) = GenericParamField(map($T,f.data))
    $T(data::Vector{<:$T}) = GenericParamField(data)
  end
end

function Fields.AffineField(
  gradient::ParamContainer{<:TensorValue},
  origin::ParamContainer{<:Point})

  data = map(AffineField,get_param_data(gradient),get_param_data(origin))
  GenericParamField(data)
end

const AffineParamField = GenericParamField{<:AffineField}

function Fields.push_∇∇(∇∇a::Field,ϕ::AffineParamField)
  Jt = ∇(ϕ)
  Jt_inv = pinvJt(Jt)
  Operation(Fields.push_∇∇)(∇∇a,Jt_inv)
end

function Fields.inverse_map(f::AffineParamField)
  AffineParamField(map(inverse_map,f.data))
end

function Fields.linear_combination(A::AbstractParamArray,b::AbstractVector{<:Field})
  ab = linear_combination(testitem(A),b)
  data = Vector{typeof(ab)}(undef,param_length(A))
  @inbounds for i in param_eachindex(A)
    data[i] = linear_combination(param_getindex(A,i),b)
  end
  ParamContainer(data)
end

# lazy maps

for T in (:Point,:(AbstractArray{<:Point}))
  @eval begin
    function Arrays.return_value(f::TrivialParamField,x::$T)
      v = return_value(f.data,x)
      TrivialParamArray(v,param_length(f))
    end

    function Arrays.return_cache(f::TrivialParamField,x::$T)
      c = return_cache(f.data,x)
      v = evaluate!(c,f.data,x)
      array = TrivialParamArray(v,param_length(f))
      c,array
    end

    function Arrays.evaluate!(cache,f::TrivialParamField,x::$T)
      c,array = cache
      v = evaluate!(c,f.data,x)
      copyto!(array.data,v)
      array
    end
  end
end

for F in (:ParamField,:ParamContainer)

  for G in (:ParamField,:ParamContainer)
    @eval begin
      function Arrays.return_value(k::Broadcasting{typeof(∘)},f::$F,g::$G)
        @check param_length(f) == param_length(g)
        v = Operation(testitem(f))(testitem(g))
        pv = Vector{typeof(v)}(undef,param_length(f))
        ParamContainer(pv)
      end

      function Arrays.return_cache(k::Broadcasting{typeof(∘)},f::$F,g::$G)
        return_value(k,f,g)
      end

      function Arrays.evaluate!(cache,::Broadcasting{typeof(∘)},f::$F,g::$G)
        @check param_length(f) == param_length(g)
        for i in param_eachindex(f)
          value = param_getindex(f,i)∘param_getindex(g,i)
          param_setindex!(cache,value,i)
        end
        cache
      end
    end
  end

  for T in (:Point,:(AbstractArray{<:Point}))
    @eval begin
      function Arrays.return_value(f::$F,x::$T)
        fi = testitem(f)
        vi = return_value(fi,x)
        array = Vector{typeof(vi)}(undef,param_length(f))
        for i in param_eachindex(f)
          array[i] = return_value(param_getindex(f,i),x)
        end
        ParamArray(array)
      end

      function Arrays.return_cache(f::$F,x::$T)
        fi = testitem(f)
        li = return_cache(fi,x)
        fix = evaluate!(li,fi,x)
        l = Vector{typeof(li)}(undef,param_length(f))
        g = parameterize(fix,param_length(f))
        for i in param_eachindex(f)
          l[i] = return_cache(param_getindex(f,i),x)
        end
        l,g
      end

      function Arrays.evaluate!(cache,f::$F,x::$T)
        l,g = cache
        @inbounds for i in param_eachindex(f)
          g[i] = evaluate!(l[i],param_getindex(f,i),x)
        end
        g
      end
    end
  end

  @eval begin
    function Arrays.return_value(k::Broadcasting{typeof(∘)},f::$F,g::Field)
      v = Operation(testitem(f))(g)
      pv = Vector{typeof(v)}(undef,param_length(f))
      ParamContainer(pv)
    end

    function Arrays.return_cache(k::Broadcasting{typeof(∘)},f::$F,g::Field)
      return_value(k,f,g)
    end

    function Arrays.evaluate!(cache,::Broadcasting{typeof(∘)},f::$F,g::Field)
      for i in param_eachindex(f)
        value = param_getindex(f,i)∘g
        param_setindex!(cache,value,i)
      end
      cache
    end

    function Arrays.return_value(k::Broadcasting{typeof(∘)},f::Field,g::$F)
      v = Operation(f)(testitem(g))
      pv = Vector{typeof(v)}(undef,param_length(g))
      ParamContainer(pv)
    end

    function Arrays.return_cache(k::Broadcasting{typeof(∘)},f::Field,g::$F)
      return_value(k,f,g)
    end

    function Arrays.evaluate!(cache,::Broadcasting{typeof(∘)},f::Field,g::$F)
      for i in param_eachindex(g)
        value = f∘param_getindex(g,i)
        param_setindex!(cache,value,i)
      end
      cache
    end

    function Arrays.return_value(f::$F,x::AbstractParamArray{<:Point})
      @check param_length(f) == param_length(x)
      fi = testitem(f)
      xi = testitem(x)
      vi = return_value(fi,xi)
      array = Vector{typeof(vi)}(undef,param_length(f))
      for i in param_eachindex(f)
        array[i] = return_value(param_getindex(f,i),param_getindex(x,i))
      end
      ParamArray(array)
    end

    function Arrays.return_cache(f::$F,x::AbstractParamArray{<:Point})
      @check param_length(f) == param_length(x)
      fi = testitem(f)
      xi = testitem(x)
      li = return_cache(fi,xi)
      fix = evaluate!(li,fi,xi)
      l = Vector{typeof(li)}(undef,param_length(f))
      g = parameterize(fix,param_length(f))
      for i in param_eachindex(f)
        l[i] = return_cache(param_getindex(f,i),param_getindex(x,i))
      end
      l,g
    end

    function Arrays.evaluate!(cache,f::$F,x::AbstractParamArray{<:Point})
      @check param_length(f) == param_length(x)
      l,g = cache
      @inbounds for i in param_eachindex(f)
        g[i] = evaluate!(l[i],param_getindex(f,i),param_getindex(x,i))
      end
      g
    end
  end

end

function Arrays.lazy_map(
  k::Broadcasting{typeof(Fields.push_∇∇)},
  cell_∇∇a::AbstractArray,
  cell_map::AbstractArray{<:AffineParamField})

  cell_Jt = lazy_map(∇,cell_map)
  cell_invJt = lazy_map(Operation(pinvJt),cell_Jt)
  lazy_map(Broadcasting(Operation(Fields.push_∇∇)),cell_∇∇a,cell_invJt)
end

function Arrays.lazy_map(
  k::Broadcasting{typeof(Fields.push_∇∇)},
  cell_∇∇at::LazyArray{<:Fill{typeof(transpose)}},
  cell_map::AbstractArray{<:AffineParamField})

  cell_∇∇a = cell_∇∇at.args[1]
  cell_∇∇b = lazy_map(k,cell_∇∇a,cell_map)
  cell_∇∇bt = lazy_map(transpose,cell_∇∇b)
  cell_∇∇bt
end

# function Arrays.return_value(b::Broadcasting{<:Function},f::ParamField,x...)
#   println(length(x))
#   error("stop")
#   evaluate(b.f,f,x...)
# end

function Arrays.return_value(k::Operation,args::Union{Field,ParamContainer{<:Field}}...)
  pargs = to_param_quantities(args...)
  Fields.OperationField(k.op,pargs)
end

function Arrays.evaluate!(cache,k::Operation,args::Union{Field,ParamContainer{<:Field}}...)
  pargs = to_param_quantities(args...)
  Fields.OperationField(k.op,pargs)
end

function Arrays.return_value(k::Broadcasting{<:Operation},args::Union{Field,ParamContainer{<:Field}}...)
  pargs = to_param_quantities(args...)
  Fields.OperationField(k.f.op,pargs)
end

function Arrays.evaluate!(cache,k::Broadcasting{<:Operation},args::Union{Field,ParamContainer{<:Field}}...)
  pargs = to_param_quantities(args...)
  Fields.OperationField(k.f.op,pargs)
end
