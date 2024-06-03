abstract type AbstractParamContainer{T,N} <: AbstractArray{T,N} end
abstract type AbstractParamBroadcast end

struct ParamContainer{T,L,A} <: AbstractParamContainer{T,1}
  array::A
  function ParamContainer(array::AbstractVector{T},::Val{L}) where {T,L}
    A = typeof(array)
    new{T,L,A}(array)
  end
end

ParamContainer(array::AbstractVector{T}) where T = ParamContainer(array,Val(length(array)))
ParamContainer(array::AbstractVector{T}) where T<:AbstractArray = ParamArray(array,Val(length(array)))

Arrays.get_array(a::ParamContainer) = a.array
Arrays.testitem(c::ParamContainer) = testitem(get_array(c))
Base.length(c::ParamContainer{T,L,A}) where {T,L,A} = L
Base.size(c::ParamContainer) = (length(c),)
Base.eachindex(c::ParamContainer) = Base.OneTo(length(c))
Base.getindex(c::ParamContainer,i...) = getindex(get_array(c),i...)
Base.setindex!(c::ParamContainer,v,i...) = setindex!(get_array(c),v,i...)

function Base.:+(a::T,b::T) where T<:ParamContainer
  c = similar(a.array)
  @inbounds for i = eachindex(a)
    c[i] = a[i] + b[i]
  end
  ParamContainer(c)
end

function Base.:-(a::T,b::T) where T<:ParamContainer
  c = similar(a.array)
  @inbounds for i = eachindex(a)
    c[i] = a[i] - b[i]
  end
  ParamContainer(c)
end

for T in (:(Point),:(AbstractVector{<:Point}))
  @eval begin
    function Arrays.return_cache(f::ParamContainer,x::$T)
      ci = return_cache(testitem(f),x)
      ai = evaluate!(ci,testitem(f),x)
      cache = Vector{typeof(ci)}(undef,length(f))
      array = Vector{typeof(ai)}(undef,length(f))
      for i = eachindex(f)
        cache[i] = return_cache(f[i],x)
      end
      cache,ParamContainer(array)
    end

    function Arrays.evaluate!(cache,f::ParamContainer,x::$T)
      cx,array = cache
      @inbounds for i = eachindex(array)
        array[i] = evaluate!(cx[i],f[i],x)
      end
      array
    end
  end
end

param_getindex(a,i) = a
param_getindex(a::ParamField,i) = a[i]
param_getindex(a::AbstractVector{<:ParamField},i) = a[i]
param_getindex(a::AbstractParamContainer{<:Field},i) = a[i]

function Arrays.return_value(k::Broadcasting{<:Operation},args::Union{Field,ParamContainer{<:Field}}...)
  pargs = filter(x->isa(x,AbstractParamContainer),args)
  L = length(first(pargs))
  ParamFieldContainer(Val(L),k.f.op,args...)
end

function Arrays.evaluate!(cache,k::Broadcasting{<:Operation},args::Union{Field,ParamContainer{<:Field}}...)
  pargs = filter(x->isa(x,AbstractParamContainer),args)
  L = length(first(pargs))
  ParamFieldContainer(Val(L),k.f.op,args...)
end

struct ParamFieldContainer{L,O,A} <: ParamField
  op::O
  args::A
  function ParamFieldContainer(::Val{L},op,args::Union{Field,AbstractVector{<:Field}}...) where L
    A = typeof(args)
    O = typeof(op)
    new{L,O,A}(op,args)
  end
end

Base.length(a::ParamFieldContainer{L}) where L = L

function Base.getindex(a::ParamFieldContainer,i::Integer)
  argi = map(x->param_getindex(x,i),a.args)
  Operation(a.op)(argi...)
end

function Arrays.testitem(a::ParamFieldContainer)
  fs = map(testitem,a.args)
  return_value(Operation(a.op),fs...)
end

for T in (:(Point),:(AbstractArray{<:Point}))
  @eval begin
    function Arrays.return_cache(f::ParamFieldContainer,x::$T)
      c = return_cache(testitem(f),x)
      cache = Vector{typeof(c)}(undef,length(f))
      for i = eachindex(cache)
        cache[i] = return_cache(f[i],x)
      end
      c,ParamContainer(cache)
    end

    function Arrays.evaluate!(cache,f::ParamFieldContainer,x::$T)
      c,pcache = cache
      @inbounds for i = eachindex(pcache)
        pcache[i] = evaluate!(c,f[i],x)
      end
      return pcache
    end
  end
end

struct BroadcastOpParamFieldArray{T,N,L,A} <: AbstractParamContainer{T,N}
  array::A
  function BroadcastOpParamFieldArray(::Val{L},op,args...) where L
    array = map(i->BroadcastOpFieldArray(op,param_getindex.(args,i)...),1:L)
    T = eltype(first(array))
    N = ndims(first(array))
    A = typeof(array)
    new{T,N,L,A}(array)
  end
end

# cannot overwrite constructor, implementing a few specific cases
function Fields.BroadcastOpFieldArray(
  op,
  a::Union{ParamField,AbstractArray{<:ParamField},AbstractParamContainer{<:Field}},
  args::Union{Field,AbstractArray{<:Field}}...)
  L = length(a)
  return BroadcastOpParamFieldArray(Val(L),op,a,args...)
end

function Fields.BroadcastOpFieldArray(
  op,a::Union{Field,AbstractArray{<:Field}},
  b::Union{ParamField,AbstractArray{<:ParamField},AbstractParamContainer{<:Field}},
  args::Union{Field,AbstractArray{<:Field}}...)
  L = length(b)
  return BroadcastOpParamFieldArray(Val(L),op,a,b,args...)
end

Arrays.get_array(a::BroadcastOpParamFieldArray) = a.array
Arrays.testitem(a::BroadcastOpParamFieldArray) = getindex(a,1)
Base.length(a::BroadcastOpParamFieldArray{T,N,L,A}) where {T,N,L,A} = L
Base.size(a::BroadcastOpParamFieldArray) = size(first(get_array(a)))
Base.axes(a::BroadcastOpParamFieldArray) = axes(first(get_array(a)))
Base.IndexStyle(::Type{<:BroadcastOpParamFieldArray}) = IndexLinear()
Base.eachindex(a::BroadcastOpParamFieldArray) = Base.OneTo(length(a))
Base.getindex(a::BroadcastOpParamFieldArray,i::Integer) = getindex(get_array(a),i)

for T in (:(Point),:(AbstractArray{<:Point}))
  @eval begin
    function Arrays.return_cache(f::BroadcastOpParamFieldArray,x::$T)
      ci = return_cache(testitem(f),x)
      bi = evaluate!(ci,testitem(f),x)
      cache = Vector{typeof(ci)}(undef,length(f))
      array = Vector{typeof(bi)}(undef,length(f))
      @inbounds for i = eachindex(f)
        cache[i] = return_cache(f[i],x)
      end
      cache,ParamContainer(array)
    end

    function Arrays.evaluate!(cache,f::BroadcastOpParamFieldArray,x::$T)
      cx,array = cache
      @inbounds for i = eachindex(array)
        array[i] = evaluate!(cx[i],f[i],x)
      end
      array
    end
  end
end
