struct PTArray{T,N,A} <: AbstractArray{T,N}
  array::A
  function PTArray(array::AbstractArray)
    A = typeof(array)
    T = eltype(array)
    N = ndims(T)
    new{T,N,A}(array)
  end
end

Arrays.get_array(a::PTArray) = a.array
Base.size(a::PTArray,i...) = size(testitem(a),i...)
Base.eltype(::Type{PTArray{T}}) where T = eltype(T)
Base.eltype(::PTArray{T}) where T = eltype(T)
Base.ndims(::PTArray{T,N} where T) where N = N
Base.ndims(::Type{PTArray{T,N}} where T) where N = N
Base.first(a::PTArray) = first(testitem(a))
Base.length(a::PTArray) = length(a.array)
Base.eachindex(a::PTArray) = eachindex(a.array)
Base.lastindex(a::PTArray) = lastindex(a.array)
Base.getindex(a::PTArray,i...) = a.array[i...]
Base.setindex!(a::PTArray,v,i...) = a.array[i...] = v
Base.iterate(a::PTArray,i...) = iterate(a.array,i...)

function Base.show(io::IO,::MIME"text/plain",a::PTArray{T}) where T
  println(io, "PTArray with eltype $T and elements")
  for i in eachindex(a)
    println(io,"  ",a.array[i])
  end
end

function Base.copy(a::PTArray{T}) where T
  b = Vector{T}(undef,length(a))
  @inbounds for i = eachindex(a)
    b[i] = copy(a[i])
  end
  PTArray(b)
end

function Base.similar(a::PTArray{T}) where T
  b = Vector{T}(undef,length(a))
  @inbounds for i = eachindex(a)
    b[i] = similar(a[i])
  end
  PTArray(b)
end

function Base.zero(a::PTArray)
  T = eltype(a)
  b = similar(a)
  b .= zero(T)
end

function Base.zeros(a::PTArray)
  get_array(zero(a))
end

function Base.sum(a::PTArray)
  sum(a.array)
end

for op in (:+,:-,:*)
  @eval begin
    function ($op)(a::PTArray,b::PTArray)
      map($op,a,b)
    end
  end
end

function Base.:*(a::PTArray,b::Number)
  map(ai->(*)(ai,b),a)
end

function Base.:*(a::Number,b::PTArray)
  b*a
end

function Base.:≈(a::PTArray,b::PTArray)
  @assert size(a) == size(b)
  for i in eachindex(a)
    if !(a[i] ≈ b[i])
      return false
    end
  end
  true
end

function Base.:(==)(a::PTArray,b::PTArray)
  @assert size(a) == size(b)
  for i in eachindex(a)
    if !(a[i] == b[i])
      return false
    end
  end
  true
end

function Base.transpose(a::PTArray)
  map(transpose,a)
end

function Base.hcat(a::PTArray...)
  n = length(first(a))
  harray = map(1:n) do j
    arrays = ()
    @inbounds for i = eachindex(a)
      arrays = (arrays...,a[i][j])
    end
    hcat(arrays...)
  end
  PTArray(harray)
end

function Base.vcat(a::PTArray...)
  n = length(first(a))
  varray = map(1:n) do j
    arrays = ()
    @inbounds for i = eachindex(a)
      arrays = (arrays...,a[i][j])
    end
    vcat(arrays...)
  end
  PTArray(varray)
end

function Base.stack(a::PTArray...)
  n = length(first(a))
  harray = map(1:n) do j
    arrays = ()
    @inbounds for i = eachindex(a)
      arrays = (arrays...,a[i][j])
    end
    stack(arrays)
  end
  PTArray(harray)
end

function Base.hvcat(nblocks::Int,a::PTArray...)
  nrows = Int(length(a)/nblocks)
  varray = map(1:nrows) do row
    vcat(a[(row-1)*nblocks+1:row*nblocks]...)
  end
  hvarray = hcat(varray...)
  hvarray
end

function Base.fill!(a::PTArray,z)
  @inbounds for i = eachindex(a)
    ai = a[i]
    fill!(ai,z)
  end
end

function LinearAlgebra.fillstored!(a::PTArray,z)
  @inbounds for i = eachindex(a)
    ai = a[i]
    fillstored!(ai,z)
  end
end

function Arrays.CachedArray(a::PTArray)
  ai = testitem(a)
  ci = CachedArray(ai)
  array = Vector{typeof(ci)}(undef,length(a))
  @inbounds for i in eachindex(a)
    array[i] = CachedArray(a.array[i])
  end
  PTArray(array)
end

function Base.map(f,a::PTArray)
  fa1 = f(testitem(a))
  array = Vector{typeof(fa1)}(undef,length(a))
  @inbounds for i = eachindex(a)
    array[i] = f(a[i])
  end
  PTArray(array)
end

function Base.map(f,a::PTArray,b::AbstractArray...)
  f1 = f(a[1],b...)
  array = Vector{typeof(f1)}(undef,length(a))
  @inbounds for i = eachindex(a)
    array[i] = f(b[1],b...)
  end
  PTArray(array)
end

struct PTBroadcasted{T}
  array::PTArray{T}
end
_get_pta(a::PTArray) = a
_get_pta(a::PTBroadcasted) = a.array

function Base.broadcasted(f,a::Union{PTArray,PTBroadcasted}...)
  pta = map(_get_pta,a)
  PTBroadcasted(map(f,pta...))
end

function Base.broadcasted(f,a::Number,b::Union{PTArray,PTBroadcasted})
  PTBroadcasted(map(x->f(a,x),_get_pta(b)))
end

function Base.broadcasted(f,a::Union{PTArray,PTBroadcasted},b::Number)
  PTBroadcasted(map(x->f(x,b),_get_pta(a)))
end

function Base.broadcasted(
  f,a::Union{PTArray,PTBroadcasted},
  b::Broadcast.Broadcasted{Broadcast.DefaultArrayStyle{0}})
  Base.broadcasted(f,a,Base.materialize(b))
end

function Base.broadcasted(
  f,a::Broadcast.Broadcasted{Broadcast.DefaultArrayStyle{0}},
  b::Union{PTArray,PTBroadcasted})
  Base.broadcasted(f,Base.materialize(a),b)
end

function Base.materialize(b::PTBroadcasted{T}) where T
  a = similar(b)
  Base.materialize!(a,b)
  a
end

function Base.materialize!(a::PTArray,b::Broadcast.Broadcasted)
  map(x->Base.materialize!(x,b),a)
  a
end

function Base.materialize!(a::PTArray,b::PTBroadcasted)
  map(Base.materialize!,a,b.array)
  a
end

function Arrays.testitem(a::PTArray{T}) where T
  if length(a) != 0
    a[1]
  else
    fill(eltype(a),1)
  end
end

function Arrays.setsize!(a::PTArray{<:CachedArray},size::Tuple{Vararg{Int}})
  @inbounds for i in eachindex(a)
    setsize!(a[i],size)
  end
end

function LinearAlgebra.ldiv!(a::PTArray,m::LU,b::PTArray)
  @inbounds for i = eachindex(a)
    ai,bi = a[i],b[i]
    ldiv!(ai,m,bi)
  end
end

function Arrays.get_array(a::PTArray{<:CachedArray})
  map(x->getproperty(x,:array),a)
end

function get_at_offsets(x::PTArray{<:AbstractVector},offsets::Vector{Int},row::Int)
  map(y->y[offsets[row]+1:offsets[row+1]],x)
end

function get_at_offsets(x::PTArray{<:AbstractMatrix},offsets::Vector{Int},row::Int,col::Int)
  map(y->y[offsets[row]+1:offsets[row+1],offsets[col]+1:offsets[col+1]],x)
end

function Arrays.return_value(
  f::BroadcastingFieldOpMap,
  a::PTArray,
  b::AbstractArray...)

  v1 = return_value(f,a[1],b)
  array = Vector{typeof(v1)}(undef,length(a))
  for i = eachindex(a)
    array[i] = return_value(f,a[i],b...)
  end
  PTArray(array)
end

function Arrays.return_cache(
  f::BroadcastingFieldOpMap,
  a::PTArray,
  b::AbstractArray...)

  c1 = return_cache(f,a[1],b...)
  b1 = evaluate!(c1,f,a[1],b...)
  cache = Vector{typeof(c1)}(undef,length(a))
  array = Vector{typeof(b1)}(undef,length(a))
  for i = eachindex(a)
    cache[i] = return_cache(f,a[i],b...)
  end
  cache,array
end

function Arrays.evaluate!(
  cache,
  f::BroadcastingFieldOpMap,
  a::PTArray{<:AbstractArray{T,N}},
  b::AbstractArray{S,N}) where {T,S,N}

  cx,array = cache
  @inbounds for i = eachindex(array)
    array[i] = evaluate!(cx[i],f,a[i],b)
  end
  PTArray(array)
end

function Arrays.evaluate!(
  cache,
  f::BroadcastingFieldOpMap,
  a::PTArray{<:AbstractMatrix},
  b::AbstractArray{S,3} where S)

  cx,array = cache
  @inbounds for i = eachindex(array)
    array[i] = evaluate!(cx[i],f,a[i],b)
  end
  PTArray(array)
end

function Arrays.evaluate!(
  cache,
  f::BroadcastingFieldOpMap,
  b::PTArray{<:AbstractArray{S,3} where S},
  a::AbstractMatrix)

  cx,array = cache
  @inbounds for i = eachindex(array)
    array[i] = evaluate!(cx[i],f,b[i],a)
  end
  PTArray(array)
end

function Arrays.evaluate!(
  cache,
  f::BroadcastingFieldOpMap,
  a::PTArray{<:AbstractVector},
  b::AbstractMatrix)

  cx,array = cache
  @inbounds for i = eachindex(array)
    array[i] = evaluate!(cx[i],f,a[i],b)
  end
  PTArray(array)
end

function Arrays.evaluate!(
  cache,
  f::BroadcastingFieldOpMap,
  b::PTArray{<:AbstractMatrix},
  a::AbstractVector)

  cx,array = cache
  @inbounds for i = eachindex(array)
    array[i] = evaluate!(cx[i],f,b[i],a)
  end
  PTArray(array)
end

function Arrays.evaluate!(
  cache,
  f::BroadcastingFieldOpMap,
  a::PTArray{<:AbstractVector},
  b::AbstractArray{S,3} where S)

  cx,array = cache
  @inbounds for i = eachindex(array)
    array[i] = evaluate!(cx[i],f,a[i],b)
  end
  PTArray(array)
end

function Arrays.evaluate!(
  cache,
  f::BroadcastingFieldOpMap,
  b::PTArray{<:AbstractArray{S,3} where S},
  a::AbstractVector)

  cx,array = cache
  @inbounds for i = eachindex(array)
    array[i] = evaluate!(cx[i],f,b[i],a)
  end
  PTArray(array)
end

function Arrays.evaluate!(
  cache,
  f::BroadcastingFieldOpMap,
  a::PTArray,
  b::AbstractArray...)

  cx,array = cache
  @inbounds for i = eachindex(array)
    array[i] = evaluate!(cx[i],f,a[i],b...)
  end
  PTArray(array)
end

function Arrays.return_value(
  f::IntegrationMap,
  a::PTArray,
  w,
  j::AbstractVector)

  v1 = return_value(f,a[1],w,j)
  array = Vector{typeof(v1)}(undef,length(a))
  for i = eachindex(a)
    array[i] = return_value(f,a[i],w,j)
  end
  PTArray(array)
end

function Arrays.return_cache(
  f::IntegrationMap,
  a::PTArray,
  w,
  j::AbstractVector)

  c1 = return_cache(f,a[1],w,j)
  b1 = evaluate!(c1,f,a[1],w,j)
  cache = Vector{typeof(c1)}(undef,length(a))
  array = Vector{typeof(b1)}(undef,length(a))
  for i = eachindex(a)
    cache[i] = return_cache(f,a[i],w,j)
  end
  cache,array
end

function Arrays.evaluate!(
  cache,
  f::IntegrationMap,
  a::PTArray,
  w,
  j::AbstractVector)

  cx,array = cache
  @inbounds for i = eachindex(array)
    array[i] = evaluate!(cx[i],f,a[i],w,j)
  end
  PTArray(array)
end

function Utils.recenter(a::PTArray{T},a0::PTArray{T};kwargs...) where T
  n = length(a)
  n0 = length(a0)
  ndiff = Int(n/n0)
  array = Vector{T}(undef,n)
  @inbounds for i = 1:n0
    array[(i-1)*ndiff+1:i*ndiff] = recenter(a[(i-1)*ndiff+1:i*ndiff],a0[i];kwargs...)
  end
  PTArray(array)
end
