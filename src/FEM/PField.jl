abstract type PField <: Field end

struct PGenericField <: PField
  fields::AbstractVector{GenericField}
end

PGenericField(f::AbstractPFunction) = PGenericField(get_fields(f))

Base.size(f::PGenericField) = size(f.fields)
Base.length(f::PGenericField) = length(f.fields)
Base.eachindex(f::PGenericField) = eachindex(f.fields)
Base.IndexStyle(::Type{<:PGenericField}) = IndexLinear()
Base.getindex(f::PGenericField,i::Integer) = GenericField(f.fields[i])

function Arrays.testitem(f::PGenericField)
  f[1]
end

for T in (:Point,:(AbstractArray{<:Point}))
  @eval begin
    function Arrays.return_cache(f::PGenericField,x::$T)
      fi = testitem(f)
      li = return_cache(fi,x)
      fix = evaluate!(li,fi,x)
      l = Vector{typeof(li)}(undef,size(f.fields))
      g = Vector{typeof(fix)}(undef,size(f.fields))
      for i in eachindex(f.fields)
        l[i] = return_cache(f.fields[i],x)
      end
      PArray(g),l
    end

    function Arrays.evaluate!(cache,f::PGenericField,x::$T)
      g,l = cache
      for i in eachindex(f.fields)
        g.array[i] = evaluate!(l[i],f.fields[i],x)
      end
      g
    end
  end
end

function Arrays.return_value(
  b::LagrangianDofBasis,
  field::OperationField{<:PGenericField})

  f1 = OperationField(testitem(field.op),field.fields)
  v1 = return_value(b,f1)
  allocate_parray(v1,length(field.op))
end

function Arrays.return_cache(
  b::LagrangianDofBasis,
  field::OperationField{<:PGenericField})

  f1 = OperationField(field.op[1],field.fields)
  c1 = return_cache(b,f1)
  a1 = evaluate!(c1,b,f1)
  cache = Vector{typeof(c1)}(undef,length(field.op))
  array = Vector{typeof(a1)}(undef,length(field.op))
  for i = eachindex(cache)
    fi = OperationField(field.op[i],field.fields)
    cache[i] = return_cache(b,fi)
  end
  cache,PArray(array)
end

function Arrays.evaluate!(
  cache,
  b::LagrangianDofBasis,
  field::OperationField{<:PGenericField})

  cf,array = cache
  @inbounds for i = eachindex(array)
    fi = OperationField(field.op[i],field.fields)
    array[i] = evaluate!(cf[i],b,fi)
  end
  array
end
