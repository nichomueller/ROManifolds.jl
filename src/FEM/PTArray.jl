abstract type PTOrdering end
struct ParamOutTimeIn <: PTOrdering end
struct TimeOutParamIn <: PTOrdering end

struct PTArray{T,A}
  array::Vector{T}
  axis::A

  function PTArray(array::Vector{T},axis::A=TimeOutParamIn()) where {T,A}
    new{T,A}(array,axis)
  end

  function PTArray(a::T,length::Int,axis::A=TimeOutParamIn()) where {T,A}
    array = fill(a,length)
    new{T,A}(array,axis)
  end
end

Base.size(b::PTArray) = size(b.array)
Base.length(b::PTArray) = length(b.array)
Base.eltype(::Type{<:PTArray{T}}) where T = T
Base.eltype(::PTArray{T}) where T = T
Base.ndims(b::PTArray) = 1
Base.ndims(::Type{PTArray}) = 1
function Base.getindex(b::PTArray,i...)
  b.array[i...]
end
function Base.setindex!(b::PTArray,v,i...)
  b.array[i...] = v
end
function Base.show(io::IO,o::PTArray)
  print(io,"PTArray($(o.array), $(o.axis))")
end

function Arrays.testitem(f::PTArray{T}) where T
  @notimplementedif !isconcretetype(T)
  if length(f) != 0
    f.array[1]
  else
    testvalue(T)
  end
end

function Arrays.testvalue(::Type{PTArray{T,A}}) where {T,A}
  s = ntuple(i->0,Val(1))
  array = Vector{T}(undef,s)
  PTArray(array,A())
end

function Base.:≈(a::AbstractArray{<:PTArray},b::AbstractArray{<:PTArray})
  all(z->z[1]≈z[2],zip(a,b))
end

function Base.:≈(a::PTArray,b::PTArray)
  if size(a) != size(b) || a.axis != b.axis
    return false
  end
  for i in eachindex(a.array)
    if !(a.array[i] ≈ b.array[i])
      return false
    end
  end
  true
end

function Base.:(==)(a::PTArray,b::PTArray)
  if size(a) != size(b) || a.axis != b.axis
    return false
  end
  for i in eachindex(a.array)
    if !(a.array[i] == b.array[i])
      return false
    end
  end
  true
end

Base.copy(a::PTArray) = PTArray(copy(a.array),copy(a.axis))
Base.eachindex(a::PTArray) = eachindex(a.array)

struct PTMap{F,A}
  f::F
  length::Int
  axis::A

  function PTMap(f::F,length::Int,axis::A=TimeOutParamIn()) where {F,A}
    new{F,A}(f,length,axis)
  end
end

function Arrays.return_value(k::PTMap{F,A},args...) where {F,A}
  arg = map(testitem,args)
  value = return_value(F(),arg...)
  PTArray(value,k.length,A())
end

function Arrays.return_cache(k::PTMap{F},args...) where F
  arg = map(testitem,args)
  cache = return_cache(F(),arg...)
  argcache = map(array_cache,args)
  ptarray = return_value(k,args...)
  cache,argcache,ptarray
end

function Arrays.evaluate!(cache,k::PTMap{F},args...) where F
  cache,argcache,ptarray = cache
  @inbounds for q = 1:k.length
    argq = map((c,a) -> getindex!(c,a,q),argcache,args)
    ptarray[q] = evaluate!(cache,F(),argq...)
  end
  ptarray
end

function Arrays.return_value(
  f::Fields.BroadcastingFieldOpMap,
  a::PTArray,
  b::AbstractArray)

  a1 = testitem(a.array)
  value = return_value(f,a1,b)
  ptvalue = PTArray(value,length(a))
  ptvalue
end

function Arrays.return_cache(
  f::Fields.BroadcastingFieldOpMap,
  a::PTArray,
  b::AbstractArray)

  a1 = testitem(a.array)
  cache = return_cache(f,a1,b)
  ptarray = PTArray(cache.array,length(a))
  cache,ptarray
end

function Arrays.evaluate!(
  cache,
  f::Fields.BroadcastingFieldOpMap,
  a::PTArray,
  b::AbstractArray)

  cache,ptarray = cache
  @inbounds for i = eachindex(ptarray)
    ptarray[i] = evaluate!(cache,f,a[i],b)
  end
  ptarray
end

function Arrays.return_value(
  f::Fields.BroadcastingFieldOpMap,
  a::AbstractArray,
  b::PTArray)

  b1 = testitem(b.array)
  value = return_value(f,a,b1)
  ptvalue = PTArray(value,length(b))
  ptvalue
end

function Arrays.return_cache(
  f::Fields.BroadcastingFieldOpMap,
  a::AbstractArray,
  b::PTArray)

  b1 = testitem(b.array)
  cache = return_cache(f,a,b1)
  ptarray = PTArray(cache.array,length(b))
  cache,ptarray
end

function Arrays.evaluate!(
  cache,
  f::Fields.BroadcastingFieldOpMap,
  a::AbstractArray,
  b::PTArray)

  cache,ptarray = cache
  @inbounds for i = eachindex(ptarray)
    ptarray[i] = evaluate!(cache,f,a[i],b)
  end
  ptarray
end
