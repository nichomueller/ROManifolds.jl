abstract type ParamField <: Field end

Arrays.testitem(f::ParamField) = f[1]

struct GenericParamField{T<:AbstractParamFunction} <: ParamField
  object::T
end

Fields.GenericField(f::AbstractParamFunction) = GenericParamField(f)
Base.length(f::GenericParamField) = length(f.object)
Base.getindex(f::GenericParamField,i::Integer) = GenericField(f.object[i])

struct ParamFieldGradient{N,F} <: ParamField
  object::F
  ParamFieldGradient{N}(object::F) where {N,F} = new{N,F}(object)
end

Fields.FieldGradient{N}(f::ParamField) where N = ParamFieldGradient{N}(f)
Fields.FieldGradient{N}(f::ParamFieldGradient{N}) where N = ParamFieldGradient{N+1}(f.object)
Base.length(f::ParamFieldGradient) = length(f.object)
Base.getindex(f::ParamFieldGradient{N},i::Integer) where N = FieldGradient{N}(f.object[i])
Arrays.testvalue(::Type{ParamFieldGradient{N,T}}) where {N,T} = ParamFieldGradient{N}(testvalue(T))

struct ZeroParamField{F} <: ParamField
  field::F
end

Base.zero(f::ParamField) = ZeroParamField(f)
Fields.ZeroField(f::ParamField) = ZeroParamField(f)
Base.length(f::ZeroParamField) = length(f.field)
Base.getindex(f::ZeroParamField,i::Integer) = ZeroField(f.field[i])
Arrays.testvalue(::Type{ZeroParamField{F}}) where F = ZeroParamField(testvalue(F))
Fields.gradient(f::ZeroParamField) = ZeroParamField(gradient(f.field))

struct ConstantParamField{T<:Number,V} <: ParamField
  value::AbstractVector{T}
  function ConstantParamField(value::AbstractVector{T}) where T
    V = typeof(value)
    new{T,V}(value)
  end
end

Fields.ConstantField(a::AbstractVector{T}) where T<:Number = ConstantParamField(a)
Base.zero(::Type{<:ConstantParamField{T}}) where T = ConstantField(zero.(T))
Base.length(f::ConstantParamField) = length(f.value)
Base.getindex(f::ConstantParamField,i::Integer) = ConstantField(f.value[i])

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

Base.length(f::OperationParamField) = length(f.fields[1])
Base.getindex(f::OperationParamField,i::Integer) = OperationField(f.op,map(x->getindex(x,i),f.fields))
Base.getindex(f::OperationParamField{<:ParamField},i::Integer) = OperationField(f.op[i],map(x->getindex(x,i),f.fields))

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

struct InverseParamField{F} <: ParamField
  original::F
end

InverseParamField(a::InverseParamField) = a
Fields.InverseField(a::ParamField) = InverseParamField(a)
Base.length(f::InverseParamField) = length(f.original)
Base.getindex(f::InverseParamField,i::Integer) = Fields.InverseField(f.original[i])

# this aims to make a field of type F behave like a pfield of length L
struct FieldToParamField{F,L} <: ParamField
  field::F
  FieldToParamField(field::F,::Val{L}) where {F,L} = new{F,L}(field)
end

FieldToParamField(f::Field,L::Integer) = FieldToParamField(f,Val(L))
FieldToParamField(f::ParamField,L::Integer) = f
FieldToParamField(f,args...) = f
Base.length(f::FieldToParamField{F,L} where F) where L = L
Base.getindex(f::FieldToParamField,i::Integer) = f.field

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
      array = Vector{typeof(vi)}(undef,length(f))
      for i in 1:length(f)
        array[i] = return_value(f[i],x)
      end
      ParamArray(array)
    end

    function Arrays.return_cache(f::ParamField,x::$T)
      fi = testitem(f)
      li = return_cache(fi,x)
      fix = evaluate!(li,fi,x)
      l = Vector{typeof(li)}(undef,length(f))
      g = Vector{typeof(fix)}(undef,length(f))
      for i in 1:length(f)
        l[i] = return_cache(f[i],x)
      end
      l,ParamArray(g)
    end

    function Arrays.evaluate!(cache,f::ParamField,x::$T)
      l,g = cache
      for i in 1:length(f)
        g[i] = evaluate!(l[i],f[i],x)
      end
      g
    end
  end
end
