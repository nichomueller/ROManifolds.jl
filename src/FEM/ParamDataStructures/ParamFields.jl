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

for op in (:+,:-,:*,:/,:⋅,:⊙,:⊗)
  @eval ($op)(a::ParamField,b::Field) = Operation($op)(a,TrivialParamField(b,param_length(a)))
  @eval ($op)(a::Field,b::ParamField) = Operation($op)(TrivialParamField(a,param_length(b)),b)
end

function Arrays.return_value(k::Operation,fields::ParamField...)
  plength = find_param_length(fields...)
  fi = map(testitem,fields)
  vi = return_value(k,fi...)
  v = Vector{typeof(vi)}(undef,plength)
  for i in 1:plength
    fi = map(f -> param_getindex(f,i),fields)
    v[i] = return_value(k,fi...)
  end
  return GenericParamField(v)
end

function Arrays.return_cache(k::Operation,fields::ParamField...)
  v = return_value(k,fields...)
  c = copy(v.data)
  return c,v
end

function Arrays.evaluate!(cache,k::Operation,fields::ParamField...)
  c,v = cache
  @inbounds for i in 1:param_length(v)
    fi = map(f -> param_getindex(f,i),fields)
    v.data[i] = evaluate!(c[i],k,fi...)
  end
  v
end

const ParamOperation = Operation{<:ParamField}

param_length(k::ParamOperation) = param_length(k.op)
param_getindex(k::ParamOperation,i::Integer) = Operation(param_getindex(k.op,i))
Arrays.testitem(k::ParamOperation) = param_getindex(k,1)

function Arrays.return_value(k::ParamOperation,fields::Field...)
  plength = param_length(k)
  pfields = to_param_quantities(fields...;plength)
  ki = testitem(k)
  fi = map(testitem,pfields)
  vi = return_value(ki,fi...)
  v = Vector{typeof(vi)}(undef,plength)
  for i in 1:plength
    ki = param_getindex(k,i)
    fi = map(f -> param_getindex(f,i),pfields)
    v[i] = return_value(ki,fi...)
  end
  return GenericParamField(v)
end

function Arrays.return_cache(k::ParamOperation,fields::Field...)
  v = return_value(k,fields...)
  c = copy(v.data)
  return c,v
end

function Arrays.evaluate!(cache,k::ParamOperation,fields::Field...)
  c,v = cache
  plength = param_length(k)
  pfields = to_param_quantities(fields...;plength)
  @inbounds for i in 1:param_length(v)
    ki = param_getindex(k,i)
    fi = map(f -> param_getindex(f,i),pfields)
    v.data[i] = evaluate!(c[i],ki,fi...)
  end
  v
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

function Fields.linear_combination(A::AbstractParamVector,b::AbstractVector{<:Field})
  ab = linear_combination(testitem(A),b)
  data = Vector{typeof(ab)}(undef,param_length(A))
  @inbounds for i in param_eachindex(A)
    data[i] = linear_combination(param_getindex(A,i),b)
  end
  GenericParamField(data)
end

# parametric field arrays

function Fields.linear_combination(A::AbstractParamMatrix,b::AbstractVector{<:Field})
  ab = linear_combination(testitem(A),b)
  data = Vector{typeof(ab)}(undef,param_length(A))
  @inbounds for i in param_eachindex(A)
    data[i] = linear_combination(param_getindex(A,i),b)
  end
  ParamContainer(data)
end

for op in (:(Fields.∇),:(Fields.∇∇))
  @eval begin
    function $op(A::ParamContainer)
      ParamContainer(map($op,get_param_data(A)))
    end
  end
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

for T in (:∇,:∇∇)
  @eval begin
    function Arrays.return_value(k::Broadcasting{typeof($T)},A::ParamContainer)
      v = return_value(k,testitem(A))
      pv = Vector{typeof(v)}(undef,param_length(A))
      ParamContainer(pv)
    end

    function Arrays.return_cache(k::Broadcasting{typeof($T)},A::ParamContainer)
      return_value(k,A)
    end

    function Arrays.evaluate!(cache,k::Broadcasting{typeof($T)},A::ParamContainer)
      for i in param_eachindex(A)
        value = evaluate(k,param_getindex(A,i))
        param_setindex!(cache,value,i)
      end
      cache
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

function Arrays.return_value(k::Broadcasting{<:Operation},args::Union{ParamField,ParamContainer}...)
  plength = find_param_length(args...)
  ai = map(testitem,args)
  vi = return_value(k,ai...)
  v = Vector{typeof(vi)}(undef,plength)
  for i in 1:plength
    ai = map(a -> param_getindex(a,i),args)
    v[i] = return_value(k,fi...)
  end
  return ParamContainer(v)
end

function Arrays.return_cache(k::Broadcasting{<:Operation},args::Union{ParamField,ParamContainer}...)
  v = return_value(k,args...)
  c = copy(v.data)
  return c,v
end

function Arrays.evaluate!(cache,k::Broadcasting{<:Operation},args::Union{ParamField,ParamContainer}...)
  c,v = cache
  @inbounds for i in 1:param_length(v)
    ai = map(a -> param_getindex(a,i),args)
    v.data[i] = evaluate!(c[i],k,ai...)
  end
  v
end

function Arrays.return_value(k::Broadcasting{<:ParamOperation},args::Union{Field,AbstractArray{<:Field}}...)
  plength = param_length(k)
  pfields = to_param_quantities(args...;plength)
  ki = testitem(k)
  ai = map(testitem,pfields)
  vi = return_value(ki,ai...)
  v = Vector{typeof(vi)}(undef,plength)
  for i in 1:plength
    ki = param_getindex(k,i)
    ai = map(a -> param_getindex(a,i),args)
    v[i] = return_value(ki,ai...)
  end
  return GenericParamField(v)
end

function Arrays.return_cache(k::Broadcasting{<:ParamOperation},args::Union{Field,AbstractArray{<:Field}}...)
  v = return_value(k,args...)
  c = copy(v.data)
  return c,v
end

function Arrays.evaluate!(cache,k::Broadcasting{<:ParamOperation},args::Union{Field,AbstractArray{<:Field}}...)
  c,v = cache
  plength = param_length(k)
  pfields = to_param_quantities(args...;plength)
  @inbounds for i in 1:param_length(v)
    ki = param_getindex(k,i)
    ai = map(a -> param_getindex(a,i),pfields)
    v.data[i] = evaluate!(c[i],ki,ai...)
  end
  v
end
