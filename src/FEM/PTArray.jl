struct PTArray{T<:AbstractArray}
  array::AbstractVector{T}

  function PTArray(array::AbstractVector{T}) where {T<:AbstractArray}
    new{T}(array)
  end

  function PTArray(a::T,length::Int) where T
    array = Vector{T}(undef,length)
    fill!(array,a)
    new{T}(array)
  end
end

get_array(a::PTArray) = a.array

Base.size(a::PTArray) = size(a.array)
Base.length(a::PTArray) = length(a.array)
Base.eltype(::Type{PTArray{T}}) where T = eltype(T)
Base.eltype(::PTArray{T}) where T = eltype(T)
Base.eachindex(a::PTArray) = eachindex(a.array)
Base.ndims(::PTArray) = 1
Base.ndims(::Type{<:PTArray}) = 1
Base.iterate(a::PTArray,i...) = iterate(a.array,i...)

function Base.getindex(a::PTArray,i...)
  a.array[i...]
end

function Base.setindex!(a::PTArray,v,i...)
  a.array[i...] = v
end

function Base.first(a::PTArray)
  PTArray([first(testitem(a))])
end

function Base.show(io::IO,o::PTArray{T}) where T
  print(io,"PTArray of type $T and length $(length(o.array))")
end

Base.copy(a::PTArray) = PTArray(copy(a.array))

Base.similar(a::PTArray) = PTArray(map(similar,a.array))

function Base.fill!(a::PTArray{T},v::S) where {S,T}
  array = Vector{S}(undef,length(a))
  fill!(array,v)
  PTArray(fill!(a.array,array))
end

struct PTArrayStyle <: Broadcast.BroadcastStyle end
Base.broadcastable(x::PTArray) = x
Broadcast.BroadcastStyle(::Type{<:PTArray}) = PTArrayStyle()
function Base.materialize!(a::PTArray,b::Base.Broadcast.Broadcasted)
  map(x->Base.materialize!(x,b),a.array)
  a
end

function Base.materialize!(
  a::PTArray,
  b::Broadcast.Broadcasted{<:PTArrayStyle})

  map((x,y)->Base.materialize!(x,y),a.array,map(z->z.array,b.args)...)
  a
end

function LinearAlgebra.fillstored!(a::PTArray,z)
  a1 = testitem(a)
  fillstored!(a1,z)
  @inbounds for i = eachindex(a)
    a.array[i] .= a1
  end
end

function Base.zero(a::PTArray)
  T = eltype(a)
  b = similar(a)
  b .= zero(T)
end

function Base.zeros(a::PTArray)
  zero(a).array
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

    # function ($op)(a::PTArray{T},b::T) where T
    #   ptb = fill(b,length(a))
    #   PTArray(map($op,a.array,ptb))
    # end

    # function ($op)(a::T,b::PTArray{T}) where T
    #   pta = fill(a,length(b))
    #   PTArray(map($op,pta,b.array))
    # end
  end
end

function Base.transpose(a::PTArray)
  PTArray(map(transpose,a.array))
end

Algebra.create_from_nz(a::PTArray) = a

function Arrays.CachedArray(a::PTArray)
  ai = testitem(a)
  ci = CachedArray(ai)
  array = Vector{typeof(ci)}(undef,length(a))
  for i in eachindex(a.array)
    array[i] = CachedArray(a.array[i])
  end
  PTArray(array)
end

function Arrays.testitem(a::PTArray{T}) where T
  @notimplementedif !isconcretetype(T)
  if length(a) != 0
    a.array[1]
  else
    fill(eltype(a),1)
  end
end

function Arrays.testvalue(::Type{PTArray{T}}) where T
  array = Vector{T}(undef,(1,))
  PTArray(array)
end

function Arrays.return_value(
  f::Gridap.Fields.BroadcastingFieldOpMap,
  a::PTArray,
  x::Vararg{Union{AbstractArray,PTArray}})

  a1 = get_at_index(1,a,x...)
  return_value(f,a1...)
end

function Arrays.return_cache(
  f::Gridap.Fields.BroadcastingFieldOpMap,
  a::PTArray,
  x::Vararg{Union{AbstractArray,PTArray}})

  sq1 = _get_1st_pta(a,x...)
  n = length(sq1)
  val = return_value(f,a,x...)
  ptval = PTArray(val,n)
  a1 = get_at_index(1,a,x...)
  cx = return_cache(f,a1...)
  cx,ptval
end

function Arrays.evaluate!(
  cache,
  f::Gridap.Fields.BroadcastingFieldOpMap,
  a::PTArray,
  x::Vararg{Union{AbstractArray,PTArray}})

  cx,ptval = cache
  @inbounds for i = eachindex(ptval)
    ai = get_at_index(i,a,x...)
    ptval.array[i] = evaluate!(cx,f,ai...)
  end
  ptval
end

function Arrays.return_value(
  f::Gridap.Fields.BroadcastingFieldOpMap,
  a::AbstractArray,
  x::PTArray)

  a1 = get_at_index(1,a,x)
  return_value(f,a1...)
end

function Arrays.return_cache(
  f::Gridap.Fields.BroadcastingFieldOpMap,
  a::AbstractArray,
  x::PTArray)

  sq1 = _get_1st_pta(a,x)
  n = length(sq1)
  val = return_value(f,a,x)
  ptval = PTArray(val,n)
  a1 = get_at_index(1,a,x)
  cx = return_cache(f,a1...)
  cx,ptval
end

function Arrays.evaluate!(
  cache,
  f::Gridap.Fields.BroadcastingFieldOpMap,
  a::AbstractArray,
  x::PTArray)

  cx,ptval = cache
  @inbounds for i = eachindex(ptval)
    ai = get_at_index(i,a,x)
    ptval.array[i] = evaluate!(cx,f,ai...)
  end
  ptval
end

function Arrays.return_value(f,a::PTArray,x::Union{AbstractArray,PTArray}...)
  a1 = get_at_index(1,a,x...)
  return_value(f,a1)
end

function Arrays.return_cache(f,a::PTArray,x::Union{AbstractArray,PTArray}...)
  sq1 = _get_1st_pta(a,x...)
  n = length(sq1)
  val = return_value(f,a,x...)
  ptval = PTArray(val,n)
  a1 = get_at_index(1,a,x...)
  cx = return_cache(f,a1)
  cx,ptval
end

function Arrays.evaluate!(cache,f,a::PTArray,x::Union{AbstractArray,PTArray}...)
  cx,ptval = cache
  @inbounds for i = eachindex(ptval)
    ai = get_at_index(i,a,x...)
    ptval.array[i] = evaluate!(cx,f,ai...)
  end
  ptval
end

function Arrays.return_value(f,a::AbstractArray,x::PTArray)
  a1 = get_at_index(1,a,x)
  return_value(f,a1)
end

function Arrays.return_cache(f,a::AbstractArray,x::PTArray)
  n = length(x)
  val = return_value(f,a,x)
  ptval = PTArray(val,n)
  a1 = get_at_index(1,a,x)
  cx = return_cache(f,a1...)
  cx,ptval
end

function Arrays.evaluate!(cache,f,a::AbstractArray,x::PTArray)
  cx,ptval = cache
  @inbounds for i = eachindex(ptval)
    ai = get_at_index(i,a,x)
    ptval.array[i] = evaluate!(cx,f,ai...)
  end
  ptval
end

function Arrays.lazy_map(
  f,
  a::PTArray,
  x::Union{AbstractArray,PTArray}...)

  lazy_arrays = map(eachindex(a)) do i
    ai = get_at_index(i,a,x...)
    lazy_map(f,ai...)
  end
  PTArray(lazy_arrays)
end

function Arrays.lazy_map(
  f,
  a::PTArray,
  b::Vararg{Union{AbstractArray,PTArray}})

  lazy_arrays = map(eachindex(a)) do i
    ai = get_at_index(i,a,b...)
    lazy_map(f,ai...)
  end
  PTArray(lazy_arrays)
end

function Arrays.lazy_map(f,a::AbstractArray,x::PTArray)
  lazy_arrays = map(y->lazy_map(f,a,y),x.array)
  PTArray(lazy_arrays)
end

function get_at_index(i::Int,x::Union{AbstractArray,PTArray}...)
  ret = ()
  @inbounds for xj in x
    ret = isa(xj,PTArray) ? (ret...,xj[i]) : (ret...,xj)
  end
  return ret
end

function _get_1st_pta(x::Union{AbstractArray,PTArray}...)
  @inbounds for xi in x
    if isa(xi,PTArray)
      return xi
    end
  end
end

function get_arrays(a::PTArray...)
  a1 = first(a)
  n = length(a1)
  @assert all(map(length,a) .== n)
  map(x->x.array,a)
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
