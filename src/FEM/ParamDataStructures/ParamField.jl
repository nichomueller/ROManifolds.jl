abstract type ParamField <: Field end

Arrays.testitem(f::ParamField) = param_getindex(f,1)

function _find_param_length(f::Union{ParamField,Field}...)
  pf::Tuple{Vararg{ParamField}} = filter(g->isa(g,ParamField),f)
  @check all(param_length(first(pf)) .== param_length.(pf))
  return param_length(first(pf))
end

function _to_param_quantities(A::Union{ParamField,Field}...)
  plength = _find_param_length(A...)
  pA = map(f->TrivialParamField(f,plength),A)
  return pA
end

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

struct GenericParamField{T<:AbstractParamFunction} <: ParamField
  object::T
end

Fields.GenericField(f::AbstractParamFunction) = GenericParamField(f)
param_length(f::GenericParamField) = length(f.object)
param_getindex(f::GenericParamField,i::Integer) = GenericField(f.object[i])

struct ParamFieldGradient{N,F} <: ParamField
  object::F
  ParamFieldGradient{N}(object::F) where {N,F} = new{N,F}(object)
end

Fields.FieldGradient{N}(f::ParamField) where N = ParamFieldGradient{N}(f)
Fields.FieldGradient{N}(f::ParamFieldGradient{N}) where N = ParamFieldGradient{N+1}(f.object)
param_length(f::ParamFieldGradient) = param_length(f.object)
param_getindex(f::ParamFieldGradient{N},i::Integer) where N = FieldGradient{N}(f.object[i])
Arrays.testvalue(::Type{ParamFieldGradient{N,T}}) where {N,T} = ParamFieldGradient{N}(testvalue(T))

struct OperationParamField{O,F} <: ParamField
  op::O
  fields::F
end

function Fields.OperationField(op,fields::Tuple{Vararg{Field}})
  try pfields = _to_param_quantities(op,fields...)
    OperationParamField(pfields...)
  catch
    OperationField{typeof(op),typeof(fields)}(op,fields)
  end
end

param_length(f::OperationParamField) = param_length(f.fields)
param_getindex(f::OperationParamField,i::Integer) = OperationField(f.op,param_getindex.(f.fields,i))
param_getindex(f::OperationParamField{<:ParamField},i::Integer) = OperationField(param_getindex(f.op,i),param_getindex.(f.fields,i))

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

# common functions among ParamFields
Base.size(f::ParamField) = (length(f),)
Base.eltype(f::ParamField) = typeof(testitem(f))

Arrays.return_value(b::Broadcasting{<:Function},f::ParamField,x...) = evaluate(b.f,f,x...)

for T in (:Point,:(AbstractArray{<:Point}))
  @eval begin
    Arrays.testargs(f::ParamField,x::$T) = testargs(testitem(f),x)

    function Arrays.return_value(f::ParamField,x::$T)
      fi = testitem(f)
      vi = return_value(fi,x)
      array = Vector{typeof(vi)}(undef,param_length(f))
      for i in param_eachindex(f)
        array[i] = return_value(param_getindex(f,i),x)
      end
      ArrayOfSimilarArrays(array)
    end

    function Arrays.return_cache(f::ParamField,x::$T)
      fi = testitem(f)
      li = return_cache(fi,x)
      fix = evaluate!(li,fi,x)
      l = Vector{typeof(li)}(undef,param_length(f))
      g = Vector{typeof(fix)}(undef,param_length(f))
      for i in param_eachindex(f)
        l[i] = return_cache(param_getindex(f,i),x)
      end
      l,ArrayOfSimilarArrays(g)
    end

    function Arrays.evaluate!(cache,f::ParamField,x::$T)
      l,g = cache
      for i in param_eachindex(f)
        g[i] = evaluate!(param_getindex(l,i),param_getindex(f,i),x)
      end
      g
    end
  end
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

for T in (:(Point),:(AbstractArray{<:Point}))
  @eval begin
    function Arrays.return_cache(f::BroadcastOpParamFieldArray,x::$T)
      ci = return_cache(testitem(f),x)
      bi = evaluate!(ci,testitem(f),x)
      cache = Vector{typeof(ci)}(undef,param_length(f))
      array = Vector{typeof(bi)}(undef,param_length(f))
      @inbounds for i = param_eachindex(f)
        cache[i] = return_cache(param_getindex(f,i),x)
      end
      cache,ArrayOfSimilarArrays(array)
    end

    function Arrays.evaluate!(cache,f::BroadcastOpParamFieldArray,x::$T)
      cx,array = cache
      @inbounds for i = param_eachindex(array)
        array[i] = evaluate!(param_getindex(cx,i),param_getindex(f,i),x)
      end
      array
    end
  end
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
    array[i] = evaluate!(param_getindex(cx,i),f,param_getindex(a,i))
  end
  array
end
