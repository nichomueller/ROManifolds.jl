const AbstractArrayBlock{T,N} = Union{AbstractArray{T,N},ArrayBlock{T,N}}

struct Nonaffine <: OperatorType end

function get_affinity(array::AbstractVector{<:AbstractArrayBlock})
  if all([a == first(array) for a in array])
    Affine()
  else
    Nonaffine()
  end
end

combine_affinity(A::OperatorType...) = Nonaffine()
combine_affinity(A::Affine...) = Affine()

struct PTArray{A,T}
  array::AbstractVector{T}

  function PTArray{A}(array::AbstractVector{T}) where {A,T}
    new{A,T}(array)
  end

  function PTArray{A}(a::T,length::Int) where {A,T}
    array = Vector{T}(undef,length)
    fill!(array,a)
    new{A,T}(array)
  end

  PTArray{A}(a::PTArray{B,T} where B) where {A,T}  = new{A,T}(a.array)
end

Base.size(a::PTArray) = size(a.array)
Base.length(a::PTArray) = length(a.array)
Base.eltype(::Type{PTArray{A,T}}) where {A,T} = eltype(T)
Base.eltype(::PTArray{A,T}) where {A,T} = eltype(T)
Base.eachindex(a::PTArray) = eachindex(a.array)
Base.ndims(::PTArray) = 1
Base.ndims(::Type{<:PTArray}) = 1

function Base.map(f,a::PTArray)
  n = length(a)
  fa1 = f(testitem(a))
  b = PTArray{Nonaffine}(fa1,n)
  @inbounds for i = 2:n
    b[i] = f(a[i])
  end
  b
end

function Base.map(f,a::PTArray,x::Union{AbstractArrayBlock,PTArray}...)
  n = _get_length(a,x...)
  ax1 = get_at_index(1,(a,x...))
  fax1 = f(ax1...)
  b = PTArray{Nonaffine}(fax1,n)
  @inbounds for i = 2:n
    axi = get_at_index(i,(a,x...))
    b[i] = f(axi...)
  end
  b
end

function Base.map(f,a::AbstractArrayBlock,x::PTArray)
  n = length(x)
  fax1 = f(a,testitem(x))
  b = PTArray{Nonaffine}(fax1,n)
  @inbounds for i = 2:n
    b[i] = f(a,x[i])
  end
  b
end

function Base.getindex(a::PTArray,i...)
  a.array[i...]
end

function Base.setindex!(a::PTArray,v,i...)
  a.array[i...] = v
end

function Base.first(a::PTArray{<:A}) where A
  PTArray{A}([first(testitem(a))])
end

function Base.show(io::IO,o::PTArray{A,T}) where {A,T}
  print(io,"$A PTArray of type $T and length $(length(o.array))")
end

Base.copy(a::PTArray{<:A}) where A = PTArray{A}(copy(a.array))

Base.similar(a::PTArray) = map(similar,a)

function Base.fill!(a::PTArray{A,T},v::T) where {A,T}
  fill!(a.array,v)
end

function Base.materialize!(a::PTArray,b::Base.Broadcast.Broadcasted)
  map(x->Base.materialize!(x,b),a)
  a
end

function LinearAlgebra.fillstored!(a::PTArray,z)
  a1 = testitem(a)
  fillstored!(a1,z)
  @inbounds for i = eachindex(a)
    a[i] .= a1
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

function Base.:≈(a::PTArray,b::PTArray)
  if size(a) != size(b)
    return false
  end
  for i in eachindex(a)
    if !(a[i] ≈ b[i])
      return false
    end
  end
  true
end

function Base.:(==)(a::PTArray,b::PTArray)
  if size(a) != size(b)
    return false
  end
  for i in eachindex(a)
    if !(a[i] == b[i])
      return false
    end
  end
  true
end

for op in (:+,:-)
  @eval begin
    function ($op)(a::PTArray,b::PTArray)
      map($op,a,b)
    end
  end
end

function Base.transpose(a::PTArray)
  map(transpose,a)
end

Algebra.create_from_nz(a::PTArray) = a

function Arrays.CachedArray(a::PTArray{<:A}) where A
  ai = testitem(a)
  ci = CachedArray(ai)
  array = Vector{typeof(ci)}(undef,length(a))
  @inbounds for i in eachindex(a)
    array[i] = CachedArray(a[i])
  end
  PTArray{A}(array)
end

function Arrays.testitem(a::PTArray{A,T}) where {A,T}
  @notimplementedif !isconcretetype(T)
  if length(a) != 0
    a[1]
  else
    fill(eltype(a),1)
  end
end

function Arrays.return_value(
  f::Gridap.Fields.BroadcastingFieldOpMap,
  a::PTArray,
  x::Vararg{Union{AbstractArrayBlock,PTArray}})

  ax1 = get_at_index(1,(a,x...))
  return_value(f,ax1...)
end

function Arrays.return_cache(
  f::Gridap.Fields.BroadcastingFieldOpMap,
  a::PTArray,
  x::Vararg{Union{AbstractArrayBlock,PTArray}})

  n = _get_length(a,x...)
  val = return_value(f,a,x...)
  ptval = PTArray{Nonaffine}(val,n)
  ax1 = get_at_index(1,(a,x...))
  cx = return_cache(f,ax1...)
  cx,ptval
end

function Arrays.evaluate!(
  cache,
  f::Gridap.Fields.BroadcastingFieldOpMap,
  a::PTArray,
  x::Vararg{Union{AbstractArrayBlock,PTArray}})

  cx,ptval = cache
  @inbounds for i = eachindex(ptval)
    axi = get_at_index(i,(a,x...))
    ptval[i] = evaluate!(cx,f,axi...)
  end
  ptval
end

function Arrays.return_value(
  f::Gridap.Fields.BroadcastingFieldOpMap,
  a::AbstractArrayBlock,
  x::PTArray)

  x1 = get_at_index(1,x)
  return_value(f,a,x1)
end

function Arrays.return_cache(
  f::Gridap.Fields.BroadcastingFieldOpMap,
  a::AbstractArrayBlock,
  x::PTArray)

  n = length(x)
  val = return_value(f,a,x)
  ptval = PTArray{Nonaffine}(val,n)
  ax1 = get_at_index(1,x)
  cx = return_cache(f,a,ax1)
  cx,ptval
end

function Arrays.evaluate!(
  cache,
  f::Gridap.Fields.BroadcastingFieldOpMap,
  a::AbstractArrayBlock,
  x::PTArray)

  cx,ptval = cache
  @inbounds for i = eachindex(ptval)
    xi = get_at_index(i,x)
    ptval[i] = evaluate!(cx,f,a,xi)
  end
  ptval
end

function Arrays.return_value(f,a::PTArray,x::Union{AbstractArrayBlock,PTArray}...)
  ax1 = get_at_index(1,(a,x...))
  return_value(f,ax1...)
end

function Arrays.return_cache(f,a::PTArray,x::Union{AbstractArrayBlock,PTArray}...)
  n = _get_length(a,x...)
  val = return_value(f,a,x...)
  ptval = PTArray{Nonaffine}(val,n)
  ax1 = get_at_index(1,(a,x...))
  cx = return_cache(f,ax1...)
  cx,ptval
end

function Arrays.evaluate!(cache,f,a::PTArray,x::Union{AbstractArrayBlock,PTArray}...)
  cx,ptval = cache
  @inbounds for i = eachindex(ptval)
    axi = get_at_index(i,(a,x...))
    ptval[i] = evaluate!(cx,f,axi...)
  end
  ptval
end

function Arrays.return_value(f,a::AbstractArrayBlock,x::PTArray)
  x1 = get_at_index(1,x)
  return_value(f,a,x1)
end

function Arrays.return_cache(f,a::AbstractArrayBlock,x::PTArray)
  n = length(x)
  val = return_value(f,a,x)
  ptval = PTArray{Nonaffine}(val,n)
  x1 = get_at_index(1,x)
  cx = return_cache(f,a,x1)
  cx,ptval
end

function Arrays.evaluate!(cache,f,a::AbstractArrayBlock,x::PTArray)
  cx,ptval = cache
  @inbounds for i = eachindex(ptval)
    xi = get_at_index(i,x)
    ptval[i] = evaluate!(cx,f,a,xi)
  end
  ptval
end

function Arrays.lazy_map(
  f,
  a::PTArray{<:A},
  x::Vararg{Union{AbstractArrayBlock,PTArray}}) where A

  lazy_arrays = map(eachindex(a)) do i
    axi = get_at_index(i,(a,x...))
    lazy_map(f,axi...)
  end
  C = if any(map(y->isa(y,PTArray),x))
    pta = filter(y->isa(y,PTArray),x)
    combine_affinity(A,get_affinity(pta)...)
  else
    A
  end
  PTArray{C}(lazy_arrays)
end

function Arrays.lazy_map(f,a::AbstractArrayBlock,x::PTArray)
  map(y->lazy_map(f,a,y),x)
end

# AFFINE SHORTCUTS
function Base.map(f,a::PTArray{<:Affine})
  n = length(a)
  fa1 = f(testitem(a))
  PTArray{Affine}(fa1,n)
end

function Base.map(f,a::PTArray{<:Affine},x::Union{AbstractArrayBlock,PTArray{<:Affine}}...)
  n = _get_length(a,x...)
  ax1 = get_at_index(1,(a,x...))
  fax1 = f(ax1...)
  PTArray{Affine}(fax1,n)
end

function Base.map(f,a::AbstractArrayBlock,x::PTArray{<:Affine})
  n = length(x)
  fax1 = f(a,testitem(x))
  PTArray{Affine}(fax1,n)
end

function Arrays.evaluate!(
  cache,
  f::Gridap.Fields.BroadcastingFieldOpMap,
  a::PTArray{<:Affine},
  x::Vararg{Union{AbstractArrayBlock,PTArray{<:Affine}}})

  cx,ptval = cache
  ax1 = get_at_index(1,(a,x...))
  evaluate!(cx,f,ax1...)
  fill!(ptval,cx.array)
  ptval
end

function Arrays.evaluate!(
  cache,
  f::Gridap.Fields.BroadcastingFieldOpMap,
  a::AbstractArrayBlock,
  x::PTArray{<:Affine})

  cx,ptval = cache
  x1 = get_at_index(1,x)
  evaluate!(cx,f,a,x1)
  fill!(ptval,cx.array)
  ptval
end

function Arrays.evaluate!(
  cache,f,
  a::PTArray{<:Affine},
  x::Union{AbstractArrayBlock,PTArray{<:Affine}}...)

  cx,ptval = cache
  ax1 = get_at_index(1,(a,x...))
  evaluate!(cx,f,ax1...)
  fill!(ptval,cx.array)
  ptval
end

function Arrays.evaluate!(cache,f,a::AbstractArrayBlock,x::PTArray{<:Affine})
  cx,ptval = cache
  x1 = get_at_index(1,x)
  evaluate!(cx,f,a,x1)
  fill!(ptval,cx.array)
  ptval
end

get_affinity(::PTArray{<:A}) where A = A

get_at_index(::Int,x) = x
get_at_index(i::Int,x::PTArray) = x[i]
function get_at_index(i::Int,x::NTuple{N,Union{AbstractArrayBlock,PTArray}}) where N
  ret = ()
  @inbounds for xj in x
    ret = (ret...,get_at_index(i,xj))
  end
  return ret
end

function get_at_index(::Colon,x::NTuple{N,PTArray}) where N
  ret = ()
  @inbounds for j in eachindex(first(x))
    ret = (ret...,get_at_index(j,x))
  end
  return ret
end

function _get_length(x::Union{AbstractArrayBlock,PTArray}...)
  pta = filter(y->isa(y,PTArray),x)
  n = length(first(pta))
  @check all([length(y) == n for y in pta])
  n
end

function test_ptarray(a::PTArray,b::AbstractArrayBlock)
  a1 = testitem(a)
  @assert typeof(a1) == typeof(b)
  @assert all(a1 .≈ b)
  return
end

function test_ptarray(a::AbstractArrayBlock,b::PTArray)
  test_ptarray(b,a)
end

function test_ptarray(a::PTArray,b::PTArray)
  (≈)(b,a)
end
