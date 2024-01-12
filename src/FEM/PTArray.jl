struct PTArray{T,N,A} <: AbstractArray{T,N}
  array::A
  function PTArray(array::AbstractVector{T}) where T<:AbstractArray
    N = ndims(T)
    A = typeof(array)
    new{T,N,A}(array)
  end
  function PTArray(array::A) where A
    T = eltype(array)
    N = 1
    new{T,N,A}(array)
  end
end

Arrays.get_array(a::PTArray) = a.array
Arrays.testitem(a::PTArray) = testitem(get_array(a))
Base.size(a::PTArray,i...) = size(testitem(a),i...)
Base.eltype(::Type{<:PTArray{T}}) where T = eltype(T)
Base.eltype(::PTArray{T}) where T = eltype(T)
Base.ndims(::PTArray{T,N} where T) where N = N
Base.ndims(::Type{<:PTArray{T,N}} where T) where N = N
Base.first(a::PTArray) = testitem(a)
Base.length(a::PTArray) = length(get_array(a))
Base.eachindex(a::PTArray) = eachindex(get_array(a))
Base.lastindex(a::PTArray) = lastindex(get_array(a))
Base.getindex(a::PTArray,i...) = get_array(a)[i...]
Base.setindex!(a::PTArray,v,i...) = get_array(a)[i...] = v

function Base.show(io::IO,::MIME"text/plain",a::PTArray{T}) where T
  println(io, "PTArray with eltype $T and elements")
  for i in eachindex(a)
    println(io,"  ",a[i])
  end
end

function Base.copy(a::PTArray{T}) where T
  b = Vector{T}(undef,length(a))
  @inbounds for i = eachindex(a)
    b[i] = copy(a[i])
  end
  PTArray(b)
end

function Base.similar(
  a::PTArray,
  type::Type{T}=eltype(testitem(a)),
  ndim::Union{Integer,AbstractUnitRange}=length(testitem(a))) where T

  elb = similar(testitem(a),type,ndim)
  b = Vector{typeof(elb)}(undef,length(a))
  @inbounds for i = eachindex(a)
    b[i] = similar(a[i],type,ndim)
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

function ptarray(a::AbstractArray,N::Integer)
  PTArray([copy(a) for _ = 1:N])
end

function ptzeros(a::AbstractArray{T},N::Integer) where T
  b = similar(a)
  fill!(b,zero(T))
  PTArray([copy(b) for _ = 1:N])
end

function Base.sum(a::PTArray)
  sum(get_array(a))
end

for op in (:+,:-)
  @eval begin
    function ($op)(a::PTArray{T},b::PTArray{T}) where T
      array = ($op)(get_array(a),get_array(b))
      PTArray(array)
    end

    function ($op)(a::PTArray{T},b::AbstractArray{T}) where T
      array = ($op)(get_array(a),b)
      PTArray(array)
    end

    function ($op)(a::AbstractArray{T},b::PTArray{T}) where T
      array = ($op)(a,get_array(b))
      PTArray(array)
    end
  end
end

function Base.:*(a::PTArray,b::Number)
  array = get_array(a)*b
  PTArray(array)
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

function Base.maximum(f,a::PTArray)
  map(a) do ai
    maximum(f,ai)
  end
end

function Base.minimum(f,a::PTArray)
  map(a) do ai
    minimum(f,ai)
  end
end

function LinearAlgebra.fillstored!(a::PTArray,z)
  @inbounds for i = eachindex(a)
    ai = a[i]
    fillstored!(ai,z)
  end
end

function LinearAlgebra.mul!(
  c::PTArray,
  a::PTArray,
  b::PTArray,
  α::Number,β::Number)

  @inbounds for i = eachindex(a)
    ci,ai,bi = c[i],a[i],b[i]
    mul!(ci,ai,bi,α,β)
  end
end

function LinearAlgebra.ldiv!(a::PTArray,m::LU,b::PTArray)
  @inbounds for i = eachindex(a)
    ai,bi = a[i],b[i]
    ldiv!(ai,m,bi)
  end
end

function LinearAlgebra.ldiv!(a::PTArray,m::AbstractArray,b::PTArray)
  @inbounds for i = eachindex(a)
    ai,mi,bi = a[i],m[i],b[i]
    ldiv!(ai,mi,bi)
  end
end

function LinearAlgebra.rmul!(a::PTArray,b::Number)
  @inbounds for i = eachindex(a)
    ai = a[i]
    rmul!(ai,b)
  end
end

function LinearAlgebra.lu(a::PTArray)
  map(a) do ai
    lu(ai)
  end
end

function LinearAlgebra.lu!(a::PTArray,b::PTArray)
  @inbounds for i = eachindex(a)
    ai,bi = a[i],b[i]
    lu!(ai,bi)
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

function Arrays.setsize!(
  a::PTArray{<:CachedArray{T,N}},
  s::NTuple{N,Int}) where {T,N}

  @inbounds for i in eachindex(a)
    setsize!(a.array[i],s)
  end
end

function SparseArrays.resize!(a::PTArray,args...)
  map(a) do ai
    resize!(ai,args...)
  end
end

Arrays.get_array(a::PTArray{<:CachedArray}) = map(x->x.array,a.array)

function Base.map(f,a::PTArray...)
  fa1 = f(map(testitem,a)...)
  array = Vector{typeof(fa1)}(undef,length(first(a)))
  @inbounds for i = eachindex(first(a))
    ai = map(x->getindex(x,i),a)
    array[i] = f(ai...)
  end
  PTArray(array)
end

function Base.map(f,a::PTArray,b::AbstractArray...)
  f1 = f(a[1],b...)
  array = Vector{typeof(f1)}(undef,length(a))
  @inbounds for i = eachindex(a)
    array[i] = f(a[i],b...)
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
  a = similar(b.array)
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

function Base.length(
  a::BroadcastOpFieldArray{O,T,N,<:Tuple{Vararg{Union{Any,PTArray}}}}
  ) where {O,T,N}
  pta = filter(x->isa(x,PTArray),a.args)
  l = map(length,pta)
  @assert all(l .== first(l))
  return first(l)
end

function Base.size(
  a::BroadcastOpFieldArray{O,T,N,<:Tuple{Vararg{Union{Any,PTArray}}}}
  ) where {O,T,N}
  return (length(a),)
end

function Base.getindex(
  a::BroadcastOpFieldArray{O,T,N,<:Tuple{PTArray,Any}},
  i::Integer) where {O,T,N}

  ai = a.args[1][i]
  Operation(a.op)(ai,a.args[2])
end

function Base.getindex(
  a::BroadcastOpFieldArray{O,T,N,<:Tuple{Any,PTArray}},
  i::Integer) where {O,T,N}

  ai = a.args[2][i]
  Operation(a.op)(a.args[1],ai)
end

function Arrays.return_value(f::Broadcasting{typeof(∘)},a::PTArray{<:Field},b::Field)
  args = map(x->return_value(f,x,b),a)
  data = map(x->getproperty(x,:fields),args)
  OperationField(∘,data)
end

function Arrays.return_value(f::Broadcasting{typeof(∘)},a::Field,b::PTArray{<:Field})
  args = map(x->return_value(f,a,x),b)
  data = map(x->getproperty(x,:fields),args)
  OperationField(∘,data)
end

for op in (:+,:-,:*)
  @eval begin
    function Arrays.return_value(
      f::Broadcasting{typeof($op)},
      a::PTArray)

      v1 = return_value(f,a[1])
      array = Vector{typeof(v1)}(undef,length(a))
      for i = eachindex(a)
        array[i] = return_value(f,a[i])
      end
      PTArray(array)
    end

    function Arrays.return_cache(
      f::Broadcasting{typeof($op)},
      a::PTArray)

      c1 = return_cache(f,a[1])
      b1 = evaluate!(c1,f,a[1])
      cache = Vector{typeof(c1)}(undef,length(a))
      array = Vector{typeof(b1)}(undef,length(a))
      for i = eachindex(a)
        cache[i] = return_cache(f,a[i])
      end
      cache,PTArray(array)
    end

    function Arrays.evaluate!(
      cache,
      f::Broadcasting{typeof($op)},
      a::PTArray)

      cx,array = cache
      @inbounds for i = eachindex(array)
        array[i] = evaluate!(cx[i],f,a[i])
      end
      array
    end
  end
end

for op in (:+,:-,:*)
  @eval begin
    function Arrays.return_value(
      f::Broadcasting{typeof($op)},
      a::PTArray,
      b::AbstractArray)

      v1 = return_value(f,a[1],b)
      array = Vector{typeof(v1)}(undef,length(a))
      for i = eachindex(a)
        array[i] = return_value(f,a[i],b)
      end
      PTArray(array)
    end

    function Arrays.return_value(
      f::Broadcasting{typeof($op)},
      a::AbstractArray,
      b::PTArray)

      v1 = return_value(f,a,b[1])
      array = Vector{typeof(v1)}(undef,length(b))
      for i = eachindex(b)
        array[i] = return_value(f,a,b[i])
      end
      PTArray(array)
    end

    function Arrays.return_value(
      f::Broadcasting{typeof($op)},
      a::PTArray,
      b::PTArray)

      v1 = return_value(f,a[1],b[1])
      array = Vector{typeof(v1)}(undef,length(a))
      for i = eachindex(a)
        array[i] = return_value(f,a[i],b[i])
      end
      PTArray(array)
    end

    function Arrays.return_cache(
      f::Broadcasting{typeof($op)},
      a::PTArray,
      b::AbstractArray)

      c1 = return_cache(f,a[1],b)
      b1 = evaluate!(c1,f,a[1],b)
      cache = Vector{typeof(c1)}(undef,length(a))
      array = Vector{typeof(b1)}(undef,length(a))
      for i = eachindex(a)
        cache[i] = return_cache(f,a[i],b)
      end
      cache,PTArray(array)
    end

    function Arrays.return_cache(
      f::Broadcasting{typeof($op)},
      a::AbstractArray,
      b::PTArray)

      c1 = return_cache(f,a,b[1])
      b1 = evaluate!(c1,f,a,b[1])
      cache = Vector{typeof(c1)}(undef,length(b))
      array = Vector{typeof(b1)}(undef,length(b))
      for i = eachindex(b)
        cache[i] = return_cache(f,a,b[i])
      end
      cache,PTArray(array)
    end

    function Arrays.return_cache(
      f::Broadcasting{typeof($op)},
      a::PTArray,
      b::PTArray)

      c1 = return_cache(f,a[1],b[1])
      b1 = evaluate!(c1,f,a[1],b[1])
      cache = Vector{typeof(c1)}(undef,length(a))
      array = Vector{typeof(b1)}(undef,length(a))
      for i = eachindex(a)
        cache[i] = return_cache(f,a[i],b[i])
      end
      cache,PTArray(array)
    end

    function Arrays.evaluate!(
      cache,
      f::Broadcasting{typeof($op)},
      a::PTArray,
      b::AbstractArray)

      cx,array = cache
      @inbounds for i = eachindex(array)
        array[i] = evaluate!(cx[i],f,a[i],b)
      end
      array
    end

    function Arrays.evaluate!(
      cache,
      f::Broadcasting{typeof($op)},
      a::AbstractArray,
      b::PTArray)

      cx,array = cache
      @inbounds for i = eachindex(array)
        array[i] = evaluate!(cx[i],f,a,b[i])
      end
      array
    end

    function Arrays.evaluate!(
      cache,
      f::Broadcasting{typeof($op)},
      a::PTArray,
      b::PTArray)

      cx,array = cache
      @inbounds for i = eachindex(array)
        array[i] = evaluate!(cx[i],f,a[i],b[i])
      end
      array
    end
  end
end

function Arrays.return_value(
  f::Broadcasting{typeof(*)},
  a::PTArray,
  b::Number)

  v1 = return_value(f,a[1],b)
  array = Vector{typeof(v1)}(undef,length(a))
  for i = eachindex(a)
    array[i] = return_value(f,a[i],b)
  end
  PTArray(array)
end

function Arrays.return_cache(
  f::Broadcasting{typeof(*)},
  a::PTArray,
  b::Number)

  c1 = return_cache(f,a[1],b)
  b1 = evaluate!(c1,f,a[1],b)
  cache = Vector{typeof(c1)}(undef,length(a))
  array = Vector{typeof(b1)}(undef,length(a))
  for i = eachindex(a)
    cache[i] = return_cache(f,a[i],b)
  end
  cache,PTArray(array)
end

function Arrays.evaluate!(
  cache,
  f::Broadcasting{typeof(*)},
  a::PTArray,
  b::Number)

  cx,array = cache
  @inbounds for i = eachindex(array)
    array[i] = evaluate!(cx[i],f,a[i],b)
  end
  array
end

function Arrays.return_value(
  f::Broadcasting{typeof(*)},
  a::Number,
  b::PTArray)

  return_value(f,b,a)
end

function Arrays.return_cache(
  f::Broadcasting{typeof(*)},
  a::Number,
  b::PTArray)

  return_cache(f,b,a)
end

function Arrays.evaluate!(
  cache,
  f::Broadcasting{typeof(*)},
  a::Number,
  b::PTArray)

  evaluate!(cache,f,b,a)
end

function Arrays.return_value(
  f::BroadcastingFieldOpMap,
  a::PTArray,
  b::AbstractArray)

  v1 = return_value(f,a[1],b)
  array = Vector{typeof(v1)}(undef,length(a))
  for i = eachindex(a)
    array[i] = return_value(f,a[i],b)
  end
  PTArray(array)
end

function Arrays.return_value(
  f::BroadcastingFieldOpMap,
  a::AbstractArray,
  b::PTArray)

  v1 = return_value(f,a,b[1])
  array = Vector{typeof(v1)}(undef,length(b))
  for i = eachindex(b)
    array[i] = return_value(f,a,b[i])
  end
  PTArray(array)
end

function Arrays.return_value(
  f::BroadcastingFieldOpMap,
  a::PTArray,
  b::PTArray)

  v1 = return_value(f,a[1],b[1])
  array = Vector{typeof(v1)}(undef,length(a))
  for i = eachindex(a)
    array[i] = return_value(f,a[i],b[i])
  end
  PTArray(array)
end

function Arrays.return_cache(
  f::BroadcastingFieldOpMap,
  a::PTArray,
  b::AbstractArray)

  c1 = return_cache(f,a[1],b)
  b1 = evaluate!(c1,f,a[1],b)
  cache = Vector{typeof(c1)}(undef,length(a))
  array = Vector{typeof(b1)}(undef,length(a))
  for i = eachindex(a)
    cache[i] = return_cache(f,a[i],b)
  end
  cache,PTArray(array)
end

function Arrays.return_cache(
  f::BroadcastingFieldOpMap,
  a::AbstractArray,
  b::PTArray)

  c1 = return_cache(f,a,b[1])
  b1 = evaluate!(c1,f,a,b[1])
  cache = Vector{typeof(c1)}(undef,length(b))
  array = Vector{typeof(b1)}(undef,length(b))
  for i = eachindex(b)
    cache[i] = return_cache(f,a,b[i])
  end
  cache,PTArray(array)
end

function Arrays.return_cache(
  f::BroadcastingFieldOpMap,
  a::PTArray,
  b::PTArray)

  c1 = return_cache(f,a[1],b[1])
  b1 = evaluate!(c1,f,a[1],b[1])
  cache = Vector{typeof(c1)}(undef,length(a))
  array = Vector{typeof(b1)}(undef,length(a))
  for i = eachindex(a)
    cache[i] = return_cache(f,a[i],b[i])
  end
  cache,PTArray(array)
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
  array
end

function Arrays.evaluate!(
  cache,
  f::BroadcastingFieldOpMap,
  a::AbstractArray{T,N},
  b::PTArray{<:AbstractArray{S,N}}) where {T,S,N}

  cx,array = cache
  @inbounds for i = eachindex(array)
    array[i] = evaluate!(cx[i],f,a,b[i])
  end
  array
end

function Arrays.evaluate!(
  cache,
  f::BroadcastingFieldOpMap,
  a::PTArray{<:AbstractArray{T,N}},
  b::PTArray{<:AbstractArray{S,N}}) where {T,S,N}

  cx,array = cache
  @inbounds for i = eachindex(array)
    array[i] = evaluate!(cx[i],f,a[i],b[i])
  end
  array
end

for S in (:AbstractVector,:AbstractMatrix)
  for T in setdiff((:AbstractVector,:AbstractMatrix),(S,))
    @eval begin
      function Arrays.evaluate!(
        cache,
        f::BroadcastingFieldOpMap,
        a::PTArray{<:$S},
        b::$T)

        cx,array = cache
        @inbounds for i = eachindex(array)
          array[i] = evaluate!(cx[i],f,a[i],b)
        end
        array
      end

      function Arrays.evaluate!(
        cache,
        f::BroadcastingFieldOpMap,
        a::$S,
        b::PTArray{<:$T})

        cx,array = cache
        @inbounds for i = eachindex(array)
          array[i] = evaluate!(cx[i],f,a,b[i])
        end
        array
      end

      function Arrays.evaluate!(
        cache,
        f::BroadcastingFieldOpMap,
        a::PTArray{<:$S},
        b::PTArray{<:$T})

        cx,array = cache
        @inbounds for i = eachindex(array)
          array[i] = evaluate!(cx[i],f,a[i],b[i])
        end
        array
      end
    end
  end

  @eval begin
    function Arrays.evaluate!(
      cache,
      f::BroadcastingFieldOpMap,
      a::PTArray{<:$S},
      b::AbstractArray{U,3} where U)

      cx,array = cache
      @inbounds for i = eachindex(array)
        array[i] = evaluate!(cx[i],f,a[i],b)
      end
      array
    end

    function Arrays.evaluate!(
      cache,
      f::BroadcastingFieldOpMap,
      a::$S,
      b::PTArray{<:AbstractArray{U,3}} where U)

      cx,array = cache
      @inbounds for i = eachindex(array)
        array[i] = evaluate!(cx[i],f,a,b[i])
      end
      array
    end

    function Arrays.evaluate!(
      cache,
      f::BroadcastingFieldOpMap,
      a::PTArray{<:AbstractArray{U,3}} where U,
      b::$S)

      cx,array = cache
      @inbounds for i = eachindex(array)
        array[i] = evaluate!(cx[i],f,a[i],b)
      end
      array
    end

    function Arrays.evaluate!(
      cache,
      f::BroadcastingFieldOpMap,
      a::AbstractArray{U,3} where U,
      b::PTArray{<:$S})

      cx,array = cache
      @inbounds for i = eachindex(array)
        array[i] = evaluate!(cx[i],f,a,b[i])
      end
      array
    end
  end
end

function Base.getindex(k::LinearCombinationField{<:PTArray},i::Int)
  LinearCombinationField(k.values[i],k.fields,k.column)
end

for T in (:(Point),:(AbstractVector{<:Point}))
  @eval begin
    function Arrays.return_value(a::LinearCombinationField{<:PTArray},x::$T)
      v1 = return_value(a[1],x)
      array = Vector{typeof(v1)}(undef,length(a.values))
      for i = eachindex(a.values)
        array[i] = return_value(a[i],x)
      end
      PTArray(array)
    end

    function Arrays.return_cache(a::LinearCombinationField{<:PTArray},x::$T)
      c1 = return_cache(a[1],x)
      b1 = evaluate!(c1,a[1],x)
      cache = Vector{typeof(c1)}(undef,length(a.values))
      array = Vector{typeof(b1)}(undef,length(a.values))
      for i = eachindex(a.values)
        cache[i] = return_cache(a[i],x)
      end
      cache,PTArray(array)
    end

    function Arrays.evaluate!(cache,a::LinearCombinationField{<:PTArray},x::$T)
      cx,array = cache
      @inbounds for i = eachindex(array)
        array[i] = evaluate!(cx[i],a[i],x)
      end
      array
    end
  end
end

for S in (:AbstractVector,:AbstractMatrix,:AbstractArray)
  for T in (:AbstractVector,:AbstractMatrix,:AbstractArray)
    @eval begin
      function Arrays.return_value(
        k::LinearCombinationMap{<:Integer},
        v::PTArray{<:$S},
        fx::$T)

        v1 = return_value(k,v[1],fx)
        array = Vector{typeof(v1)}(undef,length(v))
        for i = eachindex(v)
          array[i] = return_value(k,v[i],fx)
        end
        PTArray(array)
      end

      function Arrays.return_cache(
        k::LinearCombinationMap{<:Integer},
        v::PTArray{<:$S},
        fx::$T)

        c1 = return_cache(k,v[1],fx)
        b1 = evaluate!(c1,k,v[1],fx)
        cache = Vector{typeof(c1)}(undef,length(v))
        array = Vector{typeof(b1)}(undef,length(v))
        for i = eachindex(v)
          cache[i] = return_cache(k,v[i],fx)
        end
        cache,PTArray(array)
      end

      function Arrays.evaluate!(
        cache,
        k::LinearCombinationMap{<:Integer},
        v::PTArray{<:$S},
        fx::$T)

        cx,array = cache
        @inbounds for i = eachindex(array)
          array[i] = evaluate!(cx[i],k,v[i],fx)
        end
        array
      end
    end
  end
end

function Fields.linear_combination(a::PTArray,b::AbstractArray)
  ab1 = linear_combination(a[1],b)
  c = Vector{typeof(ab1)}(undef,length(a))
  for i in eachindex(a)
    c[i] = linear_combination(a[i],b)
  end
  PTArray(c)
end

function Base.getindex(k::Broadcasting{<:PosNegReindex{<:PTArray,<:PTArray}},i::Int)
  fi = PosNegReindex(k.f.values_pos[i],k.f.values_neg[i])
  Broadcasting(fi)
end

function Arrays.return_value(
  k::Broadcasting{<:PosNegReindex{<:PTArray,<:PTArray}},
  x::Union{Number,AbstractArray{<:Number}}...)

  npos = length(k.f.values_pos)
  nneg = length(k.f.values_neg)
  @assert npos == nneg
  v1 = return_value(k[1],x...)
  array = Vector{typeof(v1)}(undef,npos)
  for i = 1:npos
    array[i] = return_value(k[i],x...)
  end
  PTArray(array)
end

function Arrays.return_cache(
  k::Broadcasting{<:PosNegReindex{<:PTArray,<:PTArray}},
  x::Union{Number,AbstractArray{<:Number}}...)

  npos = length(k.f.values_pos)
  nneg = length(k.f.values_neg)
  @assert npos == nneg
  c1 = return_cache(k[1],x...)
  b1 = evaluate!(c1,k[1],x...)
  cache = Vector{typeof(c1)}(undef,npos)
  array = Vector{typeof(b1)}(undef,npos)
  for i = 1:npos
    cache[i] = return_cache(k[i],x...)
  end
  cache,PTArray(array)
end

function Arrays.evaluate!(
  cache,
  k::Broadcasting{<:PosNegReindex{<:PTArray,<:PTArray}},
  x::Union{Number,AbstractArray{<:Number}}...)

  cx,array = cache
  @inbounds for i = eachindex(array)
    array[i] = evaluate!(cx[i],k[i],x...)
  end
  array
end

function Arrays.evaluate!(
  cache,
  k::Broadcasting{<:PosNegReindex{<:PTArray,<:PTArray}},
  x::AbstractArray{<:Number})

  cx,array = cache
  @inbounds for i = eachindex(array)
    array[i] = evaluate!(cx[i],k[i],x)
  end
  array
end

function Arrays.evaluate!(
  cache,
  k::Broadcasting{<:PosNegReindex{<:PTArray,<:PTArray}},
  x::Number...)

  cx,array = cache
  @inbounds for i = eachindex(array)
    array[i] = evaluate!(cx[i],k[i],x...)
  end
  array
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
  cache,PTArray(array)
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
  array
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

function get_at_offsets(x::PTArray{<:AbstractVector},offsets::Vector{Int},row::Int)
  map(y->y[offsets[row]+1:offsets[row+1]],x)
end

function get_at_offsets(x::PTArray{<:AbstractMatrix},offsets::Vector{Int},row::Int,col::Int)
  map(y->y[offsets[row]+1:offsets[row+1],offsets[col]+1:offsets[col+1]],x)
end
