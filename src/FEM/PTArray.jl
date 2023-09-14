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
Base.eltype(::Type{PTArray{T}}) where T = eltype(T)
Base.eltype(::PTArray{T}) where T = eltype(T)
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

function Base.zero(a::PTArray)
  b = similar(a)
  b .= 0.
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

for op in (:+,:-)
  @eval begin
    function ($op)(a::PTArray,b::PTArray)
      PTArray(map($op,a.array,b.array))
    end

    function ($op)(a::PTArray{T},b::T) where T
      ptb = fill(b,length(a))
      PTArray(map($op,a.array,ptb))
    end

    function ($op)(a::T,b::PTArray{T}) where T
      pta = fill(a,length(b))
      PTArray(map($op,pta,b.array))
    end
  end
end

Algebra.create_from_nz(a::PTArray) = a

function Arrays.testitem(a::PTArray{T}) where T
  @notimplementedif !isconcretetype(T)
  if length(a) != 0
    a.array[1]
  else
    testvalue(T)
  end
end

function Arrays.testvalue(::Type{PTArray{T}}) where T
  array = Vector{T}(undef,(1,))
  PTArray(array)
end

function Arrays.lazy_map(k,a::PTArray)
  lazy_arrays = map(x->lazy_map(k,x),a.array)
  PTArray(lazy_arrays)
end

function Arrays.lazy_map(k,a::PTArray,b::AbstractArray...)
  lazy_arrays = map(x->lazy_map(k,x,b...),a.array)
  PTArray(lazy_arrays)
end

function Arrays.lazy_map(k,a::AbstractArray,b::PTArray)
  lazy_arrays = map(x->lazy_map(k,a,x),b.array)
  PTArray(lazy_arrays)
end

function Arrays.lazy_map(k,a::PTArray,b::PTArray)
  lazy_arrays = map((x,y)->lazy_map(k,x,y),a.array,b.array)
  PTArray(lazy_arrays)
end

function Arrays.lazy_map(
  k::Fields.BroadcastingFieldOpMap,
  a::PTArray,
  b::AbstractArray)

  a1 = testitem(a)
  ab1 = map(testitem,(a1,b))
  T = return_type(k,ab1...)
  PTArray(map(x->lazy_map(k,T,x,b),a.array))
end

function Arrays.lazy_map(
  k::Fields.BroadcastingFieldOpMap,
  a::AbstractArray,
  b::PTArray)

  b1 = testitem(b)
  ab1 = map(testitem,(a,b1))
  T = return_type(k,ab1...)
  PTArray(map(x->lazy_map(k,T,a,x),b.array))
end

function Arrays.lazy_map(
  k::Fields.BroadcastingFieldOpMap,
  a::PTArray,
  b::PTArray)

  a1 = testitem(a)
  b1 = testitem(b)
  ab1 = map(testitem,(a1,b1))
  T = return_type(k,ab1...)
  PTArray(map((x,y)->lazy_map(k,T,x,y),a.array,b.array))
end

function Arrays.return_value(
  f::Fields.BroadcastingFieldOpMap,
  a::PTArray,
  b::AbstractArray)

  a1 = testitem(a)
  value = return_value(f,a1,b)
  ptvalue = PTArray(value,length(a))
  ptvalue
end

function Arrays.return_value(
  f::Fields.BroadcastingFieldOpMap,
  a::AbstractArray,
  b::PTArray)

  b1 = testitem(b)
  value = return_value(f,a,b1)
  ptvalue = PTArray(value,length(b))
  ptvalue
end

function Arrays.return_value(
  f::Fields.BroadcastingFieldOpMap,
  a::PTArray,
  b::PTArray)

  @assert length(a) == length(b)
  a1 = testitem(a)
  b1 = testitem(b)
  value = return_value(f,a1,b1)
  ptvalue = PTArray(value,length(a))
  ptvalue
end

function Arrays.return_cache(
  f::Fields.BroadcastingFieldOpMap,
  a::PTArray,
  b::AbstractArray)

  a1 = testitem(a)
  cache = return_cache(f,a1,b)
  ptarray = return_value(f,a,b)
  cache,ptarray
end

function Arrays.return_cache(
  f::Fields.BroadcastingFieldOpMap,
  a::AbstractArray,
  b::PTArray)

  b1 = testitem(b)
  cache = return_cache(f,a,b1)
  ptarray = return_value(f,a,b)
  cache,ptarray
end

function Arrays.return_cache(
  f::Fields.BroadcastingFieldOpMap,
  a::PTArray,
  b::PTArray)

  a1 = testitem(a)
  b1 = testitem(b)
  cache = return_cache(f,a1,b1)
  ptarray = return_value(f,a,b)
  cache,ptarray
end

function Arrays.evaluate!(
  cache,
  f::Fields.BroadcastingFieldOpMap,
  a::PTArray,
  b::AbstractArray)

  cache,ptarray = cache
  @inbounds for i = eachindex(ptarray)
    ptarray.array[i] = evaluate!(cache,f,a[i],b)
  end
  ptarray
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

function Arrays.evaluate!(
  cache,
  f::Fields.BroadcastingFieldOpMap,
  a::PTArray,
  b::PTArray)

  cache,ptarray = cache
  @inbounds for i = eachindex(ptarray)
    ptarray.array[i] = evaluate!(cache,f,a[i],b[i])
  end
  ptarray
end

function get_arrays(a::PTArray...)
  a1 = first(a)
  n = length(a1)
  @assert all(map(length,a) .== n)
  map(i->map(x->x.array[i],a),1:n)
end

isaffine(a) = false
function isaffine(a::PTArray)
  a1 = testitem(a)
  n = length(a)
  all([a.array[i] == a1 for i = 2:n])
end

function test_ptarray(a::PTArray,b::AbstractArray)
  a1 = testitem(a)
  @assert typeof(a1) == typeof(b)
  @assert all(a1 .≈ b)
  return
end

function test_ptarray(a::AbstractArray,b::PTArray)
  test_ptarray(b,a)
end

function test_ptarray(a::PTArray,b::PTArray)
  (≈)(b,a)
end
