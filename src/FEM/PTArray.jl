const AbstractArrayBlock{T,N} = Union{AbstractArray{T,N},ArrayBlock{T,N}}

struct Nonaffine <: OperatorType end

function get_affinity(array::AbstractVector{<:AbstractArrayBlock})
  a1 = testitem(array)
  if all([a == a1 for a in array])
    Affine()
  else
    Nonaffine()
  end
end

isaffine(::AbstractArrayBlock) = true

# Abstract implementation
abstract type PTArray{T} end

function PTArray(array::Vector{T}) where {T<:AbstractArrayBlock}
  affinity = get_affinity(array)
  PTArray(affinity,array)
end

Arrays.get_array(a::PTArray) = a.array
Base.size(a::PTArray) = size(a.array)
Base.eltype(::Type{PTArray{T}}) where T = eltype(T)
Base.eltype(::PTArray{T}) where T = eltype(T)
Base.ndims(::PTArray) = 1
Base.ndims(::Type{<:PTArray}) = 1
Base.first(a::PTArray) = first(testitem(a))

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

function Base.vcat(a::PTArray...)
  n = _get_length(a...)
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
  n = _get_length(a...)
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
  hvarray = stack(varray)
  hvarray
end

function Arrays.lazy_map(f,a::Union{AbstractArrayBlock,PTArray}...)
  if any(map(x->isa(x,PTArray),a))
    pt_lazy_map(f,a...)
  else
    @assert all(map(x->isa(x,AbstractArray),a))
    ai = map(testitem,a)
    T = return_type(f,ai...)
    lazy_map(f,T,ai...)
  end
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
  @notimplementedif !isconcretetype(T)
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
  @check length(a) == length(b)
  @inbounds for i = eachindex(a)
    ai,bi = a[i],b[i]
    ldiv!(ai,m,bi)
  end
end

function Arrays.get_array(a::PTArray{<:CachedArray})
  map(x->getproperty(x,:array),a)
end

function recenter(a::PTArray{T},a0::PTArray{T};kwargs...) where T
  n = length(a)
  n0 = length(a0)
  ndiff = Int(n/n0)
  array = Vector{T}(undef,n)
  @inbounds for i = 1:n0
    array[(i-1)*ndiff+1:i*ndiff] = recenter(a[(i-1)*ndiff+1:i*ndiff],a0[i];kwargs...)
  end
  PTArray(array)
end

get_at_index(::Int,x) = x

get_at_index(i::Int,x::PTArray) = x[i]

function get_at_index(i::Int,x::NTuple{N,Union{AbstractArrayBlock,PTArray}}) where N
  ret = ()
  @inbounds for xj in x
    ret = (ret...,get_at_index(i,xj))
  end
  return ret
end

function get_at_index(range::UnitRange{Int},x::NTuple{N,PTArray}) where N
  ret = ()
  @inbounds for j in range
    ret = (ret...,get_at_index(j,x))
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

function test_ptarray(a::PTArray,b::AbstractArrayBlock;n=1)
  _a = a[n]
  @assert _a ≈ b "Incorrect approximation detected for index $n"
  if typeof(_a) != typeof(b)
    @warn "Detected difference in type"
  end
  return
end

function test_ptarray(a::AbstractArrayBlock,b::PTArray;kwargs...)
  test_ptarray(b,a;kwargs...)
end

function test_ptarray(a::PTArray,b::PTArray)
  @assert (≈)(b,a)
end

# Default implementation
struct NonaffinePTArray{T} <: PTArray{T}
  array::Vector{T}

  function NonaffinePTArray(array::Vector{T}) where {T<:AbstractArrayBlock}
    new{T}(array)
  end
end

function PTArray(::Nonaffine,array::Vector{T}) where {T<:AbstractArrayBlock}
  NonaffinePTArray(array)
end

isaffine(::NonaffinePTArray) = false

Base.length(a::NonaffinePTArray) = length(a.array)
Base.eachindex(a::NonaffinePTArray) = eachindex(a.array)
Base.lastindex(a::NonaffinePTArray) = lastindex(a.array)
Base.getindex(a::NonaffinePTArray,i...) = a.array[i...]
Base.setindex!(a::NonaffinePTArray,v,i...) = a.array[i...] = v

function Base.copy(a::NonaffinePTArray{T}) where T
  b = Vector{T}(undef,length(a))
  @inbounds for i = eachindex(a)
    b[i] = copy(a[i])
  end
  NonaffinePTArray(b)
end

function Base.similar(a::NonaffinePTArray{T}) where T
  b = Vector{T}(undef,length(a))
  @inbounds for i = eachindex(a)
    b[i] = similar(a[i])
  end
  NonaffinePTArray(b)
end

function Base.show(io::IO,a::NonaffinePTArray{T}) where T
  print(io,"Nonaffine PTArray of type $T and length $(length(a.array))")
end

function Base.transpose(a::NonaffinePTArray)
  map(transpose,a)
end

function Base.fill!(a::NonaffinePTArray,z)
  @inbounds for i = eachindex(a)
    ai = a[i]
    fill!(ai,z)
  end
end

function LinearAlgebra.fillstored!(a::NonaffinePTArray,z)
  @inbounds for i = eachindex(a)
    ai = a[i]
    fillstored!(ai,z)
  end
end

function Arrays.CachedArray(a::NonaffinePTArray)
  ai = testitem(a)
  ci = CachedArray(ai)
  array = Vector{typeof(ci)}(undef,length(a))
  @inbounds for i in eachindex(a)
    array[i] = CachedArray(a.array[i])
  end
  NonaffinePTArray(array)
end

function Base.map(f,a::PTArray)
  n = length(a)
  fa1 = f(testitem(a))
  array = Vector{typeof(fa1)}(undef,n)
  @inbounds for i = 1:n
    array[i] = f(a[i])
  end
  NonaffinePTArray(array)
end

function Base.map(f,a::PTArray,x::Union{AbstractArrayBlock,PTArray}...)
  n = _get_length(a,x...)
  ax1 = get_at_index(1,(a,x...))
  fax1 = f(ax1...)
  array = Vector{typeof(fax1)}(undef,n)
  @inbounds for i = 1:n
    axi = get_at_index(i,(a,x...))
    array[i] = f(axi...)
  end
  NonaffinePTArray(array)
end

function Base.map(f,a::AbstractArrayBlock,x::PTArray)
  n = length(x)
  fax1 = f(a,testitem(x))
  array = Vector{typeof(fax1)}(undef,n)
  @inbounds for i = 1:n
    array[i] = f(a,x[i])
  end
  NonaffinePTArray(array)
end

for F in (:Map,:Function,:(Gridap.Fields.BroadcastingFieldOpMap))
  @eval begin
    function Arrays.return_value(
      f::$F,
      a::PTArray,
      x::Vararg{Union{AbstractArrayBlock,PTArray}})

      ax1 = get_at_index(1,(a,x...))
      return_value(f,ax1...)
    end

    function Arrays.return_cache(
      f::$F,
      a::PTArray,
      x::Vararg{Union{AbstractArrayBlock,PTArray}})

      n = _get_length(a,x...)
      val = return_value(f,a,x...)
      array = Vector{typeof(val)}(undef,n)
      ax1 = get_at_index(1,(a,x...))
      cx = return_cache(f,ax1...)
      cx,array
    end

    function Arrays.evaluate!(
      cache,
      f::$F,
      a::PTArray,
      x::Vararg{Union{AbstractArrayBlock,PTArray}})

      cx,array = cache
      @inbounds for i = eachindex(array)
        axi = get_at_index(i,(a,x...))
        array[i] = evaluate!(cx,f,axi...)
      end
      NonaffinePTArray(array)
    end

    function Arrays.return_value(
      f::$F,
      a::AbstractArrayBlock,
      x::PTArray)

      x1 = get_at_index(1,x)
      return_value(f,a,x1)
    end

    function Arrays.return_cache(
      f::$F,
      a::AbstractArrayBlock,
      x::PTArray)

      n = length(x)
      val = return_value(f,a,x)
      array = Vector{typeof(val)}(undef,n)
      ax1 = get_at_index(1,x)
      cx = return_cache(f,a,ax1)
      cx,array
    end

    function Arrays.evaluate!(
      cache,
      f::$F,
      a::AbstractArrayBlock,
      x::PTArray)

      cx,array = cache
      @inbounds for i = eachindex(array)
        xi = get_at_index(i,x)
        array[i] = evaluate!(cx,f,a,xi)
      end
      NonaffinePTArray(array)
    end

    function pt_lazy_map(f,a::Union{AbstractArrayBlock,PTArray}...)
      n = _get_length(a...)
      lazy_arrays = map(1:n) do i
        ai = get_at_index(i,a)
        lazy_map(f,ai...)
      end
      NonaffinePTArray(lazy_arrays)
    end
  end
end

# Affine implementation: shortcut for parameter- or time-independent quantities
struct AffinePTArray{T} <: PTArray{T}
  array::T
  len::Int

  function AffinePTArray(array::T,len::Int=1) where {T<:AbstractArrayBlock}
    new{T}(array,len)
  end
end

function PTArray(::Affine,array::Vector{T}) where {T<:AbstractArrayBlock}
  n = length(array)
  a1 = first(array)
  AffinePTArray(a1,n)
end

isaffine(::AffinePTArray) = true

Base.length(a::AffinePTArray) = a.len
Base.eachindex(a::AffinePTArray) = Base.OneTo(a.len)
Base.lastindex(a::AffinePTArray) = a.len
Base.getindex(a::AffinePTArray,i...) = a.array
Base.setindex!(a::AffinePTArray,v,i...) = a.array = v
Base.copy(a::AffinePTArray) = AffinePTArray(copy(a.array),a.len)
Base.similar(a::AffinePTArray) = AffinePTArray(similar(a.array),a.len)

function Base.show(io::IO,a::AffinePTArray{T}) where T
  print(io,"Affine PTArray of type $T and length $(a.len)")
end

function Base.transpose(a::AffinePTArray)
  a.array = a.array'
end

function Base.vcat(a::AffinePTArray...)
  n = _get_length(a...)
  j = 1
  arrays = ()
  @inbounds for i = eachindex(a)
    arrays = (arrays...,a[i][j])
  end
  varray = vcat(arrays...)
  AffinePTArray(varray,n)
end

function Base.stack(a::PTArray...)
  n = _get_length(a...)
  j = 1
  arrays = ()
  @inbounds for i = eachindex(a)
    arrays = (arrays...,a[i][j])
  end
  harray = stack(arrays)
  PTArray(harray)
end

Base.fill!(a::AffinePTArray,z) = fill!(a.array,z)

LinearAlgebra.fillstored!(a::AffinePTArray,z) = fillstored!(a.array,z)

function LinearAlgebra.ldiv!(a::AffinePTArray,m::LU,b::AffinePTArray)
  @check length(a) == length(b)
  ldiv!(testitem(a),m,testitem(b))
end

function Arrays.CachedArray(a::AffinePTArray)
  n = length(a)
  ai = testitem(a)
  ci = CachedArray(ai)
  AffinePTArray(array,n)
end

function Base.map(f,a::AffinePTArray)
  n = length(a)
  fa1 = f(testitem(a))
  AffinePTArray(fa1,n)
end

function Base.map(f,a::AffinePTArray,x::Union{AbstractArrayBlock,AffinePTArray}...)
  n = _get_length(a,x...)
  ax1 = get_at_index(1,(a,x...))
  fax1 = f(ax1...)
  AffinePTArray(fax1,n)
end

function Base.map(f,a::AbstractArrayBlock,x::AffinePTArray)
  n = length(x)
  fax1 = f(a,testitem(x))
  AffinePTArray(fax1,n)
end

for F in (:Map,:Function,:(Gridap.Fields.BroadcastingFieldOpMap))
  @eval begin
    function Arrays.evaluate!(
      cache,
      f::$F,
      a::AffinePTArray,
      x::Vararg{Union{AbstractArrayBlock,AffinePTArray}})

      cx,array = cache
      ax1 = get_at_index(1,(a,x...))
      evaluate!(cx,f,ax1...)
      AffinePTArray(cx.array,length(array))
    end

    function Arrays.evaluate!(
      cache,
      f::$F,
      a::AbstractArrayBlock,
      x::AffinePTArray)

      cx,array = cache
      x1 = get_at_index(1,x)
      evaluate!(cx,f,a,x1)
      AffinePTArray(cx.array,length(array))
    end

    function pt_lazy_map(f::$F,a::Union{AbstractArrayBlock,AffinePTArray}...)
      n = _get_length(a...)
      a1 = get_at_index(1,a)
      AffinePTArray(lazy_map(f,a1...),n)
    end
  end
end
