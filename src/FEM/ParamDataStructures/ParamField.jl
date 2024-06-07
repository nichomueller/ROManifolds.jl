abstract type ParamField <: Field end

Base.length(f::ParamField) = param_length(f)
Base.size(f::ParamField) = (length(f),)
Base.eltype(f::ParamField) = typeof(testitem(f))
Base.IteratorSize(::Type{<:ParamField}) = Base.HasShape{length(f)}()

Arrays.testitem(f::ParamField) = param_getindex(f,1)
Arrays.testargs(f::ParamField,x::Point) = testargs(testitem(f),x)
Arrays.testargs(f::ParamField,x::AbstractArray{<:Point}) = testargs(testitem(f),x)
Arrays.return_value(b::Broadcasting{<:Function},f::ParamField,x...) = evaluate(b.f,f,x...)

_to_param_quantity(f::ParamField,plength::Integer) = f
_to_param_quantity(f::Field,plength::Integer) = TrivialParamField(f,plength)

# this aims to make a field of type F behave like a pfield of length plength
struct TrivialParamField{F<:Field} <: ParamField
  field::F
  plength::Int
  function TrivialParamField(field::F,::Val{L}) where {F,L}
    new{F}(field,L)
  end
end

TrivialParamField(f::Field,plength::Int) = TrivialParamField(f,Val(plength))
TrivialParamField(f::ParamField,plength::Int) = f
TrivialParamField(f,args...) = f

param_length(f::TrivialParamField) = f.plength
param_getindex(f::TrivialParamField,i::Integer) = f.field

Arrays.evaluate(f::TrivialParamField,x::Point) = fill(evaluate(f.field,x),f.plength)

struct GenericParamField{T<:AbstractParamFunction} <: ParamField
  object::T
end

Fields.GenericField(f::AbstractParamFunction) = GenericParamField(f)
param_length(f::GenericParamField) = length(f.object)
param_getindex(f::GenericParamField,i::Integer) = GenericField(f.object[i])

Arrays.return_value(f::GenericParamField,x::Point) = return_value(f.object,x)
Arrays.return_cache(f::GenericParamField,x::Point) = return_cache(f.object,x)
Arrays.evaluate!(cache,f::GenericParamField,x::Point) = evaluate!(cache,f.object,x)

struct ParamFieldGradient{N,F} <: ParamField
  object::F
  ParamFieldGradient{N}(object::F) where {N,F} = new{N,F}(object)
end

Fields.FieldGradient{N}(f::ParamField) where N = ParamFieldGradient{N}(f)
Fields.FieldGradient{N}(f::ParamFieldGradient{N}) where N = ParamFieldGradient{N+1}(f.object)
param_length(f::ParamFieldGradient) = param_length(f.object)
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

Arrays.return_cache(f::ParamFieldGradient{N,<:Function},x::Point) where N = gradient(f.object,Val{N}())
Arrays.evaluate!(c,f::ParamFieldGradient{N,<:Function},x::Point) where N = c(x)

struct OperationParamField{O,F} <: ParamField
  op::O
  fields::F
end

function Fields.OperationField(op,fields::Tuple{Vararg{Field}})
  try
    pop,pfields... = _to_param_quantities(op,fields...)
    OperationParamField(pop,pfields)
  catch
    Fields.OperationField{typeof(op),typeof(fields)}(op,fields)
  end
end

param_length(f::OperationParamField) = _find_param_length(f.op,f.fields...)
param_getindex(f::OperationParamField,i::Integer) = Fields.OperationField(f.op,param_getindex.(f.fields,i))
param_getindex(f::OperationParamField{<:ParamField},i::Integer) = Fields.OperationField(param_getindex(f.op,i),param_getindex.(f.fields,i))

function Arrays.return_value(c::OperationParamField,x::Point)
  map(i->return_value(param_getindex(c,i),x),param_eachindex(c))
end

function Arrays.return_cache(c::OperationParamField,x::Point)
  map(i->return_cache(param_getindex(c,i),x),param_eachindex(c))
end

function Arrays.evaluate!(cache,c::OperationParamField,x::Point)
  map(i->evaluate!(cache[i],param_getindex(c,i),x),param_eachindex(c))
end

for op in (:+,:-)
  @eval begin
    function Fields.gradient(f::OperationParamField{typeof($op)})
      g = map(gradient,f.fields)
      $op(g...)
    end
  end
end

for op in (:*,:⋅)
  @eval begin
     function Fields.product_rule(::typeof($op),args::AbstractVector{<:Number}...)
       map((x...) -> Fields.product_rule($op,x...),args...)
     end
  end
end

for op in (:*,:⋅,:⊙,:⊗)
  @eval begin
    function Fields.gradient(f::OperationParamField{typeof($op)})
      @notimplementedif length(f.fields) != 2
      f1,f2 = f.fields
      g1,g2 = map(gradient,f.fields)
      k(F1,F2,G1,G2) = Fields.product_rule($op,F1,F2,G1,G2)
      Operation(k)(f1,f2,g1,g2)
    end
  end
end

function Fields.gradient(f::OperationParamField{<:Field})
  a = f.op
  @notimplementedif length(f.fields) != 1
  b, = f.fields
  x = ∇(a)∘b
  y = ∇(b)
  y⋅x
end

struct InverseParamField{F} <: ParamField
  original::F
end

InverseParamField(a::InverseParamField) = a
Fields.InverseField(a::ParamField) = InverseParamField(a)
param_length(f::InverseParamField) = param_length(f.original)
param_getindex(f::InverseParamField,i::Integer) = Fields.InverseField(param_getindex(f.original,i))

function Arrays.return_cache(c::InverseParamField,x::Point)
  map(i->return_cache(param_getindex(c,i),x),param_eachindex(c))
end

function Arrays.evaluate!(cache,c::InverseParamField,x::Point)
  map(i->evaluate!(cache[i],param_getindex(c,i),x),param_eachindex(c))
end

struct BroadcastOpParamFieldArray{O,T,N,A} <: AbstractVector{BroadcastOpFieldArray{O,T,N,A}}
  array::Vector{BroadcastOpFieldArray{O,T,N,A}}
end

function Fields.BroadcastOpFieldArray(op,args...)
  BroadcastOpParamFieldArray(map(a->BroadcastOpFieldArray(op,a),args...))
end

param_length(a::BroadcastOpParamFieldArray) = length(a.array)
param_getindex(a::BroadcastOpParamFieldArray,i::Integer) = a.array[i]

Base.size(a::BroadcastOpParamFieldArray) = param_length(a)
Base.getindex(a::BroadcastOpParamFieldArray,i::Integer) = param_getindex(a,i)
Arrays.testitem(a::BroadcastOpParamFieldArray) = param_getindex(a,1)

# lazy maps

function Arrays.return_value(f::Union{ParamField,BroadcastOpParamFieldArray},x::AbstractArray{<:Point})
  fi = testitem(f)
  vi = return_value(fi,x)
  array = Vector{typeof(vi)}(undef,param_length(f))
  for i in param_eachindex(f)
    array[i] = return_value(param_getindex(f,i),x)
  end
  ParamArray(array)
end

function Arrays.return_cache(f::Union{ParamField,BroadcastOpParamFieldArray},x::AbstractArray{<:Point})
  fi = testitem(f)
  li = return_cache(fi,x)
  fix = evaluate!(li,fi,x)
  l = Vector{typeof(li)}(undef,param_length(f))
  g = array_of_similar_arrays(fix,param_length(f))
  for i in param_eachindex(f)
    l[i] = return_cache(param_getindex(f,i),x)
  end
  l,g
end

function Arrays.evaluate!(cache,f::Union{ParamField,BroadcastOpParamFieldArray},x::AbstractArray{<:Point})
  l,g = cache
  for i in param_eachindex(f)
    g[i] = evaluate!(l[i],param_getindex(f,i),x)
  end
  g
end

function Arrays.return_cache(f::Broadcasting{typeof(∇)},a::BroadcastOpParamFieldArray)
  ci = return_cache(f,testitem(a))
  bi = evaluate!(ci,f,testitem(a))
  cache = Vector{typeof(ci)}(undef,param_length(a))
  array = Vector{typeof(bi)}(undef,param_length(a))
  @inbounds for i = param_eachindex(a)
    cache[i] = return_cache(f,param_getindex(a,i))
  end
  cache,ParamContainer(array)
end

function Arrays.evaluate!(cache,f::Broadcasting{typeof(∇)},a::BroadcastOpParamFieldArray)
  cx,array = cache
  @inbounds for i = param_eachindex(array)
    array[i] = evaluate!(cx[i],f,param_getindex(a,i))
  end
  array
end

function Arrays.return_cache(f::ParamContainer{Union{Field,ParamField}},x::AbstractArray{<:Point})
  c = return_cache(testitem(f),x)
  cache = Vector{typeof(c)}(undef,param_length(f))
  @inbounds for i = param_eachindex(cache)
    cache[i] = return_cache(param_getindex(f,i),x)
  end
  return c,ParamContainer(cache)
end

function Arrays.evaluate!(cache,f::ParamContainer{Union{Field,ParamField}},x::AbstractArray{<:Point})
  c,pcache = cache
  @inbounds for i = param_eachindex(pcache)
    pcache[i] = evaluate!(c,param_getindex(f,i),x)
  end
  return pcache
end
