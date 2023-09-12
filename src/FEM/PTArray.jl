struct PTArray{T}
  array::AbstractVector{T}

  function PTArray(array::AbstractVector{T}) where T
    new{T}(array)
  end

  function PTArray(a::T,length::Int) where T
    array = fill(a,length)
    new{T}(array)
  end
end

Base.size(a::PTArray) = size(a.array)
Base.length(a::PTArray) = length(a.array)
Base.eltype(::Type{PTArray{T}}) where T = T
Base.eltype(::PTArray{T}) where T = T
Base.ndims(::PTArray) = 1
Base.ndims(::Type{<:PTArray}) = 1
Base.eachindex(a::PTArray) = eachindex(a.array)

function Base.getindex(a::PTArray,i...)
  a.array[i...]
end

function Base.setindex!(a::PTArray,v,i...)
  a.array[i...] = v
end

function Base.first(a::PTArray,i...)
  PTArray(first(a.array,i...))
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

function Base.materialize!(a::PTArray,b::Base.Broadcast.Broadcasted)
  map(x->Base.materialize!(x,b),a.array)
  a
end

function LinearAlgebra.fillstored!(a::PTArray,z)
  a1 = testitem(a)
  fillstored!(a1,z)
  @inbounds for i = eachindex(ptarray)
    a.array[i] .= a1
  end
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

function Arrays.lazy_map(::typeof(evaluate),a::PTArray,args...)
  lazy_arrays = map(x->lazy_map(evaluate,x,args...),a.array)
  PTArray(lazy_arrays)
end

function Arrays.lazy_map(k::Broadcasting{typeof(∘)},a::PTArray,args...)
  lazy_arrays = map(x->lazy_map(k,x,args...),a.array)
  PTArray(lazy_arrays)
end

function Arrays.lazy_map(k::Broadcasting{typeof(gradient)},a::PTArray)
  lazy_arrays = map(x->lazy_map(k,x),a.array)
  PTArray(lazy_arrays)
end

function Arrays.lazy_map(
  ::Broadcasting{typeof(push_∇)},
  cell_∇a::PTArray{<:AbstractArray},
  cell_map::AbstractArray)

  cell_Jt = lazy_map(∇,cell_map)
  cell_invJt = lazy_map(Operation(pinvJt),cell_Jt)
  lazy_arrays = map(x->lazy_map(Broadcasting(Operation(⋅)),cell_invJt,x),cell_∇a.array)
  PTArray(lazy_arrays)
end

function Arrays.lazy_map(
  k::Broadcasting{typeof(push_∇)},
  cell_∇a::PTArray{<:Fields.MemoArray},
  cell_map::AbstractArray)

  lazy_arrays = map(x->lazy_map(k,x,cell_map),cell_∇a.array)
  PTArray(lazy_arrays)
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
    ptarray.array[i] = evaluate!(cache,f,a,b[i])
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
