abstract type PField <: Field end

struct GenericPField{T<:AbstractPFunction} <: PField
  object::T
end

Fields.GenericField(f::AbstractPFunction) = GenericPField(f)
Base.length(f::GenericPField) = length(f.object)
Arrays.testitem(f::GenericPField) = GenericField(testitem(f.object))

function Base.iterate(f::GenericPField,oldstate...)
  it = iterate(f.object,oldstate...)
  if isnothing(it)
    return nothing
  end
  fit,nextstate = it
  GenericField(fit),nextstate
end

struct PFieldGradient{N,F} <: PField
  object::F
  PFieldGradient{N}(object::F) where {N,F} = new{N,F}(object)
end

Fields.FieldGradient{N}(f::PField) where N = PFieldGradient{N}(f)
Fields.FieldGradient{N}(f::PFieldGradient{N}) where N = PFieldGradient{N+1}(f.object)
Base.length(f::PFieldGradient) = length(f.object)
Arrays.testitem(f::PFieldGradient{N}) where N = FieldGradient{N}(testitem(f.object))
Arrays.testvalue(::Type{PFieldGradient{N,T}}) where {N,T} = PFieldGradient{N}(testvalue(T))

function Base.iterate(f::PFieldGradient{N},oldstate...) where N
  it = iterate(f.object,oldstate...)
  if isnothing(it)
    return nothing
  end
  fit,nextstate = it
  FieldGradient{N}(fit),nextstate
end

struct ZeroPField{F} <: PField
  field::F
end

Base.zero(f::PField) = ZeroPField(f)
Fields.ZeroField(f::PField) = ZeroPField(f)
Base.length(f::ZeroPField) = length(f.field)
Arrays.testitem(f::ZeroPField) = ZeroField(testitem(f.field))
Arrays.testvalue(::Type{ZeroPField{F}}) where F = ZeroPField(testvalue(F))
Fields.gradient(f::ZeroPField) = ZeroPField(gradient(f.field))

function Base.iterate(f::ZeroPField,oldstate...)
  it = iterate(f.field,oldstate...)
  if isnothing(it)
    return nothing
  end
  fit,nextstate = it
  ZeroField(fit),nextstate
end

struct ConstantPField{T<:Number} <: PField
  value::AbstractVector{T}
end

Fields.ConstantField(a::AbstractVector{T}) where T<:Number = ConstantPField(a)
Base.zero(::Type{ConstantPField{T}}) where T = ConstantField(zero.(T))
Base.length(f::ConstantPField) = length(f.value)
Arrays.testitem(f::ConstantPField) = ConstantField(testitem(f.value))

function Base.iterate(f::ConstantPField,oldstate...)
  it = iterate(f.value,oldstate...)
  if isnothing(it)
    return nothing
  end
  fit,nextstate = it
  ConstantField(fit),nextstate
end

# const POperation = Operation{T} where T<:Union{AbstractPFunction,PField}
# Base.length(op::POperation) = length(op.op)

struct OperationPField{O,F} <: PField
  op::O
  fields::F
end

Fields.OperationField(op,fields::Tuple{Vararg{PField}}) = OperationPField(op,fields)
Fields.OperationField(op::PField,fields::Tuple{Vararg{Field}}) = OperationPField(op,FieldToPField.(fields,length(op)))
function Fields.OperationField(op,fields::Tuple{Vararg{Field}})
  if any(isa.(fields,PField))
    OperationPField(op,fields)
  else
    OperationField{typeof(op),typeof(fields)}(op,fields)
  end
end
# Fields.OperationField(op::POperation,fields::Field...) = OperationPField(op,FieldToPField.(fields,length(op)))
function Base.length(f::OperationPField)
  L = map(length,f.fields)
  @check all(L .== first(L))
  first(L)
end
Arrays.testitem(f::OperationPField) = Fields.OperationField(f.op,map(testitem,f.fields))
Arrays.testitem(f::OperationPField{<:PField}) = Fields.OperationField(map(testitem,f.op),map(testitem,f.fields))

function Base.iterate(f::OperationPField,oldstate...)
  it = iterate.(f.fields,oldstate...)
  if all(isnothing.(it))
    return nothing
  end
  fit,nextstate = it |> tuple_of_arrays
  Fields.OperationField(f.op,fit),nextstate
end

function Base.iterate(f::OperationPField{<:PField})
  ito = iterate.(f.op)
  itf = iterate.(f.fields)
  if all(isnothing.(ito)) && all(isnothing.(itf))
    return nothing
  end
  oit,nextstateo = ito |> tuple_of_arrays
  fit,nextstatef = itf |> tuple_of_arrays
  Fields.OperationField(oit,fit),(nextstateo,nextstatef)
end

function Base.iterate(f::OperationPField{<:PField},oldstate)
  oldstateo,oldstatef = oldstate
  ito = iterate.(f.op,oldstateo)
  itf = iterate.(f.fields,oldstatef)
  if all(isnothing.(ito)) && all(isnothing.(itf))
    return nothing
  end
  oit,nextstateo = ito |> tuple_of_arrays
  fit,nextstatef = itf |> tuple_of_arrays
  Fields.OperationField(oit,fit),(nextstateo,nextstatef)
end

for op in (:+,:-)
  @eval begin
    function Fields.gradient(a::OperationPField{typeof($op)})
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
    function Fields.gradient(a::OperationPField{typeof($op)})
      f = a.fields
      @notimplementedif length(f) != 2
      f1,f2 = f
      g1,g2 = map(gradient,f)
      k(F1,F2,G1,G2) = Fields.product_rule($op,F1,F2,G1,G2)
      Operation(k)(f1,f2,g1,g2)
    end
  end
end

function Fields.gradient(f::OperationPField{<:Field})
  a = f.op
  @notimplementedif length(f.fields) != 1
  b, = f.fields
  x = ∇(a)∘b
  y = ∇(b)
  y⋅x
end

function Arrays.return_value(k::Broadcasting{typeof(∘)},f::PField,g::PField)
  fi,gi = map(testitem,(f,g))
  vi = return_value(k,fi,gi)
  array = Vector{typeof(vi)}(undef,length(f))
  for (i,(fi,gi)) in enumerate(zip(f,g))
    array[i] = return_value(k,fi,gi)
  end
  PArray(array)
end
function Arrays.return_value(k::Broadcasting{typeof(∘)},f::PField,g::Field)
  return_value(k,f,FieldToPField(g,length(f)))
end
function Arrays.return_value(k::Broadcasting{typeof(∘)},f::Field,g::PField)
  return_value(k,FieldToPField(f,length(g)),g)
end
function Arrays.return_value(k::Broadcasting{<:Operation},f::PField,g::PField)
  fi,gi = map(testitem,(f,g))
  vi = return_value(k,fi,gi)
  array = Vector{typeof(vi)}(undef,length(f))
  for (i,(fi,gi)) in enumerate(zip(f,g))
    array[i] = return_value(k,fi,gi)
  end
  PArray(array)
end
function Arrays.return_value(k::Broadcasting{<:Operation},f::PField,g::Field)
  return_value(k,f,FieldToPField(g,length(f)))
end
function Arrays.return_value(k::Broadcasting{<:Operation},f::Field,g::PField)
  return_value(k,FieldToPField(f,length(g)),g)
end

# function Arrays.evaluate!(cache,k::Broadcasting{typeof(∘)},f::PField,g::PField)
#   f∘g
# end
# function Arrays.evaluate!(cache,k::Broadcasting{typeof(∘)},f::PField,g::Field)
#   evaluate!(cache,k,f,FieldToPField(g,length(f)))
# end
# function Arrays.evaluate!(cache,k::Broadcasting{typeof(∘)},f::Field,g::PField)
#   evaluate!(cache,k,FieldToPField(f,length(g)),g)
# end

struct VoidPField{F} <: PField
  field::F
  isvoid::Bool
end

Fields.VoidField(field::PField,isvoid::Bool) = VoidPField(field,isvoid)
Base.length(f::VoidPField) = length(f.field)
Arrays.testitem(f::VoidPField) = VoidField(testitem(f.field),f.isvoid)
Fields.gradient(f::VoidPField) = VoidPField(gradient(f.field),f.isvoid)

function Base.iterate(f::VoidPField,oldstate...)
  it = iterate(f.field,oldstate...)
  if isnothing(it)
    return nothing
  end
  fit,nextstate = it
  VoidField(fit,f.isvoid),nextstate
end

struct InversePField{F} <: PField
  original::F
end

InversePField(a::InversePField) = a
Fields.InverseField(a::PField) = InversePField(a)
Base.length(f::InversePField) = length(f.original)
Arrays.testitem(f::InversePField) = Fields.InverseField(testitem(f.original))

function Base.iterate(f::InversePField,oldstate...)
  it = iterate(f.original,oldstate...)
  if isnothing(it)
    return nothing
  end
  fit,nextstate = it
  Fields.InverseField(fit),nextstate
end

# this aims to make a field of type F behave like a pfield of length L
struct FieldToPField{F,L} <: PField
  field::F
  FieldToPField(field::F,::Val{L}) where {F,L} = new{F,L}(field)
end

FieldToPField(f::Field,L::Integer) = FieldToPField(f,Val(L))
FieldToPField(f::PField,args...) = f
Base.length(f::FieldToPField{F,L} where F) where L = L
Arrays.testitem(f::FieldToPField) = f.field

function Base.iterate(f::FieldToPField)
  f.field,1
end
function Base.iterate(f::FieldToPField{F,L} where F,it) where L
  if it > L
    return nothing
  end
  f.field,it+1
end

# common functions among PFields
Base.size(f::PField) = (length(f),)
Base.eltype(f::PField) = typeof(testitem(f))

function Base.map(f::PField,x::AbstractArray{<:Point})
  fi = testitem(f)
  vi = map(fi,x)
  array = Vector{typeof(vi)}(undef,length(f))
  for (i,fi) in enumerate(f)
    array[i] = map(fi,x)
  end
  PArray(array)
end

Base.broadcasted(f::PField,x::AbstractArray{<:Point}) = map(f,x)

for T in (:Point,:(AbstractArray{<:Point}))
  @eval begin
    Arrays.testargs(f::PField,x::$T) = testargs(testitem(f),x)

    function Arrays.return_value(f::PField,x::$T)
      fi = testitem(f)
      vi = return_value(fi,x)
      array = Vector{typeof(vi)}(undef,length(f))
      for (i,fi) in enumerate(f)
        array[i] = return_value(fi,x)
      end
      PArray(array)
    end

    function Arrays.return_cache(f::PField,x::$T)
      fi = testitem(f)
      li = return_cache(fi,x)
      fix = evaluate!(li,fi,x)
      l = Vector{typeof(li)}(undef,length(f))
      g = Vector{typeof(fix)}(undef,length(f))
      for (i,fi) in enumerate(f)
        l[i] = return_cache(fi,x)
      end
      l,PArray(g)
    end

    function Arrays.evaluate!(cache,f::PField,x::$T)
      l,g = cache
      for (i,fi) in enumerate(f)
        g[i] = evaluate!(l[i],fi,x)
      end
      g
    end
  end
end

function Arrays.return_value(
  b::LagrangianDofBasis,
  field::OperationPField)

  f1 = OperationField(testitem(field.op),field.fields)
  v1 = return_value(b,f1)
  allocate_parray(v1,length(field.op))
end

function Arrays.return_cache(
  b::LagrangianDofBasis,
  field::OperationPField)

  @error "deprecate"
  f1 = OperationField(testitem(field.op),field.fields)
  c1 = return_cache(b,f1)
  a1 = evaluate!(c1,b,f1)
  cache = Vector{typeof(c1)}(undef,length(field.op))
  array = Vector{typeof(a1)}(undef,length(field.op))
  for (i,opi) = enumerate(field.op)
    fi = OperationField(opi,field.fields)
    cache[i] = return_cache(b,fi)
  end
  cache,PArray(array)
end

function Arrays.evaluate!(
  cache,
  b::LagrangianDofBasis,
  field::OperationPField)

  @error "deprecate"
  cf,array = cache
  @inbounds for (i,opi) = enumerate(field.op)
    fi = OperationField(opi,field.fields)
    array[i] = evaluate!(cf[i],b,fi)
  end
  array
end
