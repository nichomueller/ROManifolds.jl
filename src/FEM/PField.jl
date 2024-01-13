abstract type AbstractPTFunction{P,T} <: Function end

struct PFunction{P} <: AbstractPTFunction{P,Nothing}
  f::Function
  params::P
end

Base.length(pf::PFunction) = length(pf.params)
Base.length(pf::PFunction{<:AbstractVector{<:Real}}) = 1
Base.eachindex(pf::PFunction) = Base.OneTo(length(pf))
_length(μ::Vector{<:Number}) = 1
_length(μ) = length(μ)

function Base.getindex(pf::PFunction,i::Int)
  pf.f(pf.params[i])
end

function Base.getindex(pf::PFunction{<:AbstractVector{<:Real}},i::Int)
  @assert i == 1
  pf.f(pf.params)
end

struct PTFunction{P,T} <: AbstractPTFunction{P,T}
  f::Function
  params::P
  times::T
end

Base.length(ptf::PTFunction) = length(ptf.params)*length(ptf.times)
Base.length(ptf::PTFunction{<:AbstractVector{<:Real},<:AbstractVector{<:Real}}) = length(ptf.times)
Base.length(ptf::PTFunction{<:AbstractVector{<:Real},<:Real}) = 1
Base.eachindex(ptf::PTFunction) = Base.OneTo(length(ptf))
_length(μ::Vector{<:Number},t) = length(t)
_length(μ,t) = length(μ)*length(t)

function Base.getindex(ptf::PTFunction,i::Int)
  nt = length(ptf.times)
  ptf.f(ptf.params[slow_idx(i,nt)],ptf.times[fast_idx(i,nt)])
end

function Base.getindex(ptf::PTFunction{<:AbstractVector{<:Real},<:AbstractVector{<:Real}},i::Int)
  ptf.f(ptf.params,ptf.times[i])
end

function Base.getindex(ptf::PTFunction{<:AbstractVector{<:Real},<:Real},i::Int)
  @assert i == 1
  ptf.f(ptf.params,ptf.times)
end

function get_fields(ptf::AbstractPTFunction)
  n = length(ptf)
  fields = Vector{GenericField}(undef,n)
  @inbounds for k = eachindex(ptf)
    fields[k] = GenericField(ptf[k])
  end
  fields
end

abstract type PField <: Field end

struct PGenericField <: PField
  fields::AbstractVector{GenericField}
  function PGenericField(f::AbstractPTFunction)
    fields = get_fields(f)
    new(fields)
  end
end

Base.size(a::PGenericField) = size(a.fields)
Base.length(a::PGenericField) = length(a.fields)
Base.eachindex(a::PGenericField) = eachindex(a.fields)
Base.IndexStyle(::Type{<:PGenericField}) = IndexLinear()
Base.getindex(a::PGenericField,i::Integer) = GenericField(a.fields[i])

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
