abstract type ParamField <: Field end

struct GenericParamField{T<:AbstractParamFunction} <: ParamField
  object::T
end

Fields.GenericField(f::AbstractParamFunction) = GenericParamField(f)
Base.length(f::GenericParamField) = length(f.object)
Arrays.testitem(f::GenericParamField) = GenericField(testitem(f.object))

function Base.iterate(f::GenericParamField,oldstate...)
  it = iterate(f.object,oldstate...)
  if isnothing(it)
    return nothing
  end
  fit,nextstate = it
  GenericField(fit),nextstate
end

struct ParamFieldGradient{N,F} <: ParamField
  object::F
  ParamFieldGradient{N}(object::F) where {N,F} = new{N,F}(object)
end

Fields.FieldGradient{N}(f::ParamField) where N = ParamFieldGradient{N}(f)
Fields.FieldGradient{N}(f::ParamFieldGradient{N}) where N = ParamFieldGradient{N+1}(f.object)
Base.length(f::ParamFieldGradient) = length(f.object)
Arrays.testitem(f::ParamFieldGradient{N}) where N = FieldGradient{N}(testitem(f.object))
Arrays.testvalue(::Type{ParamFieldGradient{N,T}}) where {N,T} = ParamFieldGradient{N}(testvalue(T))

function Base.iterate(f::ParamFieldGradient{N},oldstate...) where N
  it = iterate(f.object,oldstate...)
  if isnothing(it)
    return nothing
  end
  fit,nextstate = it
  FieldGradient{N}(fit),nextstate
end

struct ZeroParamField{F} <: ParamField
  field::F
end

Base.zero(f::ParamField) = ZeroParamField(f)
Fields.ZeroField(f::ParamField) = ZeroParamField(f)
Base.length(f::ZeroParamField) = length(f.field)
Arrays.testitem(f::ZeroParamField) = ZeroField(testitem(f.field))
Arrays.testvalue(::Type{ZeroParamField{F}}) where F = ZeroParamField(testvalue(F))
Fields.gradient(f::ZeroParamField) = ZeroParamField(gradient(f.field))

function Base.iterate(f::ZeroParamField,oldstate...)
  it = iterate(f.field,oldstate...)
  if isnothing(it)
    return nothing
  end
  fit,nextstate = it
  ZeroField(fit),nextstate
end

struct ConstantParamField{T<:Number} <: ParamField
  value::AbstractVector{T}
end

Fields.ConstantField(a::AbstractVector{T}) where T<:Number = ConstantParamField(a)
Base.zero(::Type{ConstantParamField{T}}) where T = ConstantField(zero.(T))
Base.length(f::ConstantParamField) = length(f.value)
Arrays.testitem(f::ConstantParamField) = ConstantField(testitem(f.value))

function Base.iterate(f::ConstantParamField,oldstate...)
  it = iterate(f.value,oldstate...)
  if isnothing(it)
    return nothing
  end
  fit,nextstate = it
  ConstantField(fit),nextstate
end

struct OperationParamField{O,F} <: ParamField
  op::O
  fields::F
end

function _find_length(op,fields)
  pfields = filter(x->isa(x,ParamField),fields)
  if isempty(pfields)
    @check isa(op,ParamField)
    L = length(op)
  else
    L = length.(pfields)
    @check all(L .== first(L))
    L = first(L)
    if isa(op,ParamField)
      @check length(op) == L
    end
  end
  return L
end

function Fields.OperationField(op,fields::Tuple{Vararg{Field}})
  if any(isa.(fields,ParamField)) || isa(op,ParamField)
    L = _find_length(op,fields)
    OperationParamField(FieldToParamField(op,L),FieldToParamField.(fields,L))
  else
    OperationField{typeof(op),typeof(fields)}(op,fields)
  end
end

function Base.length(f::OperationParamField)
  L = map(length,f.fields)
  first(L)
end
Arrays.testitem(f::OperationParamField) = Fields.OperationField(f.op,map(testitem,f.fields))
Arrays.testitem(f::OperationParamField{<:Field}) = Fields.OperationField(testitem(f.op),map(testitem,f.fields))

function Base.iterate(f::OperationParamField,oldstate...)
  itf = iterate.(f.fields,oldstate...)
  if all(isnothing.(itf))
    return nothing
  end
  fit,nextstate = itf |> tuple_of_arrays
  Fields.OperationField(f.op,fit),nextstate
end

function Base.iterate(f::OperationParamField{<:ParamField})
  ito = iterate(f.op)
  itf = iterate.(f.fields)
  if isnothing(ito) && all(isnothing.(itf))
    return nothing
  end
  oit,nextstateo = ito
  fit,nextstatef = itf |> tuple_of_arrays
  Fields.OperationField(oit,fit),(nextstateo,nextstatef)
end

function Base.iterate(f::OperationParamField{<:ParamField},oldstate)
  oldstateo,oldstatef = oldstate
  ito = iterate(f.op,oldstateo)
  itf = iterate.(f.fields,oldstatef)
  if isnothing(ito) && all(isnothing.(itf))
    return nothing
  end
  oit,nextstateo = ito
  fit,nextstatef = itf |> tuple_of_arrays
  Fields.OperationField(oit,fit),(nextstateo,nextstatef)
end

for op in (:+,:-)
  @eval begin
    function Fields.gradient(a::OperationParamField{typeof($op)})
      f = a.fields
      g = map(gradient,f)
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
    function Fields.gradient(a::OperationParamField{typeof($op)})
      f = a.fields
      @notimplementedif length(f) != 2
      f1,f2 = f
      g1,g2 = map(gradient,f)
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

struct VoidParamField{F} <: ParamField
  field::F
  isvoid::Bool
end

Fields.VoidField(field::ParamField,isvoid::Bool) = VoidParamField(field,isvoid)
Base.length(f::VoidParamField) = length(f.field)
Arrays.testitem(f::VoidParamField) = VoidField(testitem(f.field),f.isvoid)
Fields.gradient(f::VoidParamField) = VoidParamField(gradient(f.field),f.isvoid)

function Base.iterate(f::VoidParamField,oldstate...)
  it = iterate(f.field,oldstate...)
  if isnothing(it)
    return nothing
  end
  fit,nextstate = it
  VoidField(fit,f.isvoid),nextstate
end

struct InverseParamField{F} <: ParamField
  original::F
end

InverseParamField(a::InverseParamField) = a
Fields.InverseField(a::ParamField) = InverseParamField(a)
Base.length(f::InverseParamField) = length(f.original)
Arrays.testitem(f::InverseParamField) = Fields.InverseField(testitem(f.original))

function Base.iterate(f::InverseParamField,oldstate...)
  it = iterate(f.original,oldstate...)
  if isnothing(it)
    return nothing
  end
  fit,nextstate = it
  Fields.InverseField(fit),nextstate
end

# this aims to make a field of type F behave like a pfield of length L
struct FieldToParamField{F,L} <: ParamField
  field::F
  FieldToParamField(field::F,::Val{L}) where {F,L} = new{F,L}(field)
end

FieldToParamField(f::Field,L::Integer) = FieldToParamField(f,Val(L))
FieldToParamField(f::ParamField,L::Integer) = f
FieldToParamField(f,args...) = f
Base.length(f::FieldToParamField{F,L} where F) where L = L
Arrays.testitem(f::FieldToParamField) = f.field

function Base.iterate(f::FieldToParamField)
  f.field,1
end
function Base.iterate(f::FieldToParamField{F,L} where F,it) where L
  if it >= L
    return nothing
  end
  f.field,it+1
end

# common functions among ParamFields
Base.size(f::ParamField) = (length(f),)
Base.eltype(f::ParamField) = typeof(testitem(f))

function Base.map(f::ParamField,x::AbstractArray{<:Point})
  fi = testitem(f)
  vi = map(fi,x)
  array = Vector{typeof(vi)}(undef,length(f))
  for (i,fi) in enumerate(f)
    array[i] = map(fi,x)
  end
  ParamArray(array)
end

Base.broadcasted(f::ParamField,x::AbstractArray{<:Point}) = map(f,x)

Arrays.return_value(b::Broadcasting{<:Function},f::ParamField,x...) = evaluate(b.f,f,x...)

for T in (:Point,:(AbstractArray{<:Point}))
  @eval begin
    Arrays.testargs(f::ParamField,x::$T) = testargs(testitem(f),x)

    function Arrays.return_value(f::ParamField,x::$T)
      fi = testitem(f)
      vi = return_value(fi,x)
      array = Vector{typeof(vi)}(undef,length(f))
      for (i,fi) in enumerate(f)
        array[i] = return_value(fi,x)
      end
      ParamArray(array)
    end

    function Arrays.return_cache(f::ParamField,x::$T)
      fi = testitem(f)
      li = return_cache(fi,x)
      fix = evaluate!(li,fi,x)
      l = Vector{typeof(li)}(undef,length(f))
      g = Vector{typeof(fix)}(undef,length(f))
      for (i,fi) in enumerate(f)
        l[i] = return_cache(fi,x)
      end
      l,ParamArray(g)
    end

    function Arrays.evaluate!(cache,f::ParamField,x::$T)
      l,g = cache
      for (i,fi) in enumerate(f)
        g[i] = evaluate!(l[i],fi,x)
      end
      g
    end
  end
end
