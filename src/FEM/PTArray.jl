struct PTArray{T} <: AbstractVector{T}
  array::Vector{T}

  function PTArray(array::Vector{T}) where T
    new{T}(array)
  end

  function PTArray(a::T,length::Int) where T
    array = fill(a,length)
    new{T}(array)
  end
end

Base.size(b::PTArray) = size(b.array)
Base.length(b::PTArray) = length(b.array)
Base.eltype(::Type{PTArray{T}}) where T = T
Base.eltype(::PTArray{T}) where T = T
Base.ndims(b::PTArray) = 1
Base.ndims(::Type{PTArray}) = 1
Base.eachindex(a::PTArray) = eachindex(a.array)

function Base.getindex(b::PTArray,i...)
  b.array[i...]
end

function Base.setindex!(b::PTArray,v,i...)
  b.array[i...] = v
end

function Base.show(io::IO,o::PTArray{T}) where T
  print(io,"PTArray of eltype $T and length $(length(o.array))")
end

Base.copy(a::PTArray) = PTArray(copy(a.array))

function Base.copyto!(a::PTArray,b::PTArray)
  PTArray(map(copyto!,a.array,b.array))
end

function Base.fill!(a::PTArray{T},v::T) where T
  PTArray(fill!(a.array,v))
end

function Base.fill!(a::PTArray{T},v::S) where {S,T<:AbstractVector{S}}
  PTArray(fill!(a.array,fill(v,length(testitem(a)))))
end

Base.similar(a::PTArray) = PTArray(map(similar,a.array))

# Base.broadcasted(f,a::PTArray,b) = PTArray(map(ai->broadcasted(f,ai,b),a.array))

# Base.broadcasted(f,a,b::PTArray) = PTArray(map(bi->broadcasted(f,a,bi),b.array))

# function Base.broadcasted(f,a::PTArray{T},b::PTArray{T}) where T
#   @assert length(a) == length(b)
#   c = copy(a)
#   @inbounds for i = eachindex(a)
#     c.array[i] = broadcasted(f,a[i],b[i])
#   end
#   c
# end

function LinearAlgebra.fillstored!(a::PTArray,z)
  a1 = testitem(a)
  fillstored!(a1,z)
  @inbounds for i = eachindex(ptarray)
    a.array[i] .= a1
  end
end

function Arrays.testitem(a::PTArray{T}) where T
  @notimplementedif !isconcretetype(T)
  if length(a) != 0
    a.array[1]
  else
    testvalue(T)
  end
end

function Arrays.testvalue(::Type{PTArray{T}}) where T
  s = ntuple(i->0,Val(1))
  array = Vector{T}(undef,s)
  PTArray(array)
end

function Base.:≈(a::AbstractArray{<:PTArray},b::AbstractArray{<:PTArray})
  all(z->z[1]≈z[2],zip(a,b))
end

function Base.:≈(a::PTArray,b::PTArray)
  if size(a) != size(b)
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
  if size(a) != size(b)
    return false
  end
  for i in eachindex(a.array)
    if !(a.array[i] == b.array[i])
      return false
    end
  end
  true
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
    ptarray.array[i] = evaluate!(cache,f,a[i],b)
  end
  ptarray
end

function Arrays.return_value(f::IntegrationMap,aq::PTArray,args...)
  aq1 = testitem(aq.array)
  value = return_value(f,aq1,args...)
  ptvalue = PTArray(value,length(aq.array))
  ptvalue
end

function Arrays.return_cache(f::IntegrationMap,aq::PTArray,args...)
  aq1 = testitem(aq.array)
  cache = return_cache(f,aq1,args...)
  ptarray = return_value(f,aq,args...)
  cache,ptarray
end

function Arrays.evaluate!(cache,f::IntegrationMap,aq::PTArray,args...)
  cache,ptarray = cache
  @inbounds for i = eachindex(ptarray)
    ptarray.array[i] = evaluate!(cache,f,aq[i],args...)
  end
  ptarray
end
