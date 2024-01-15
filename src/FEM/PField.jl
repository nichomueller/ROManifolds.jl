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
