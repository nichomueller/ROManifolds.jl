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
