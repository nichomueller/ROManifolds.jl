abstract type AbstractPTFunction{P,T} <: Function end

struct PFunction{P} <: AbstractPTFunction{P,Nothing}
  f::Function
  params::P
end

struct PTFunction{P,T} <: AbstractPTFunction{P,T}
  f::Function
  params::P
  times::T
end

function get_fields(pf::PFunction{<:AbstractVector{<:Number}})
  p = pf.params
  GenericField(pf.f(p))
end

function get_fields(pf::PFunction)
  p = pf.params
  np = length(p)
  fields = Vector{GenericField}(undef,np)
  @inbounds for k = eachindex(p)
    pk = p[k]
    fields[k] = GenericField(pf.f(pk))
  end
  fields
end

function get_fields(ptf::PTFunction{<:AbstractVector{<:Number},<:Real})
  p,t = ptf.params,ptf.times
  GenericField(ptf.f(p,t))
end

function get_fields(ptf::PTFunction{<:AbstractVector{<:Number},<:AbstractVector{<:Number}})
  p,t = ptf.params,ptf.times
  nt = length(t)
  fields = Vector{GenericField}(undef,nt)
  @inbounds for k = 1:nt
    tk = t[k]
    fields[k] = GenericField(ptf.f(p,tk))
  end
  fields
end

function get_fields(ptf::PTFunction)
  p,t = ptf.params,ptf.times
  np = length(p)
  nt = length(t)
  npt = np*nt
  fields = Vector{GenericField}(undef,npt)
  @inbounds for k = 1:npt
    pk = p[slow_idx(k,nt)]
    tk = t[fast_idx(k,nt)]
    fields[k] = GenericField(ptf.f(pk,tk))
  end
  fields
end

function Arrays.evaluate!(cache,f::AbstractPTFunction,x::Point)
  g = get_fields(f)
  map(g) do gi
    gi(x)
  end
end

abstract type PTField <: Field end

struct PTGenericField <: PTField
  fields::AbstractVector{GenericField}
  function PTGenericField(f::AbstractPTFunction)
    fields = get_fields(f)
    new(fields)
  end
end

Base.size(a::PTGenericField) = size(a.fields)
Base.length(a::PTGenericField) = length(a.fields)
Base.IndexStyle(::Type{<:PTGenericField}) = IndexLinear()
Base.getindex(a::PTGenericField,i::Integer) = GenericField(a.fields[i])
Arrays.testitem(f::PTGenericField) = f[1]

# function Base.broadcasted(f::PTGenericField,x)
#   array = map(f.fields) do f
#     f.(x)
#   end
#   PTArray(array)
# end

for T in (:Point,:(AbstractArray{<:Point}))
  @eval begin
    function Arrays.return_cache(f::PTGenericField,x::$T)
      fi = testitem(f)
      li = return_cache(fi,x)
      fix = evaluate!(li,fi,x)
      l = Vector{typeof(li)}(undef,size(f.fields))
      g = Vector{typeof(fix)}(undef,size(f.fields))
      for i in eachindex(f.fields)
        l[i] = return_cache(f.fields[i],x)
      end
      PTArray(g),l
    end

    function Arrays.evaluate!(cache,f::PTGenericField,x::$T)
      g,l = cache
      for i in eachindex(f.fields)
        g.array[i] = evaluate!(l[i],f.fields[i],x)
      end
      g
    end
  end
end
