struct PArray{T,N,A,L} <: AbstractArray{T,N}
  array::A
  indices::NTuple{L,Bool}
  function PArray(array::AbstractVector{T},indices::NTuple{L,Bool}) where {T<:AbstractArray,L}
    N = ndims(T)
    A = typeof(array)
    new{T,N,A,L}(array,indices)
  end
  function PArray(array::A,indices::NTuple{L,Bool}) where {A,L}
    T = eltype(array)
    N = 1
    new{T,N,A,L}(array,indices)
  end
end

const AffinePArray{T,N,A} = PArray{T,N,A,1}

function PArray(array)
  indices = ntuple(i->true,Val(length(array)))
  PArray(array,indices)
end

function PArray(array::AbstractVector{T}) where {T<:Number}
  array
end

function PArray{T}(::UndefInitializer,L::Integer) where T
  array = Vector{T}(undef,L)
  PArray(array)
end

Arrays.get_array(a::PArray) = a.array
Arrays.testitem(a::PArray) = testitem(get_array(a))
Base.size(a::PArray{T},i...) where {T<:AbstractArray} = size(testitem(a),i...)
Base.size(a::PArray) = (length(a),)
Base.eltype(::Type{<:PArray{T}}) where T = eltype(T)
Base.eltype(::PArray{T}) where T = eltype(T)
Base.ndims(::PArray{T,N} where T) where N = N
Base.ndims(::Type{<:PArray{T,N}} where T) where N = N
Base.first(a::PArray) = testitem(a)
Base.length(::PArray{T,N,A,L} where {T,N,A}) where L = L
Base.eachindex(::PArray{T,N,A,L} where {T,N,A}) where L = Base.OneTo(L)
Base.lastindex(::PArray{T,N,A,L} where {T,N,A}) where L = L
Base.getindex(a::PArray,i...) = get_array(a)[i...]
Base.setindex!(a::PArray,v,i...) = get_array(a)[i...] = v
Base.iterate(a::PArray,i...) = iterate(a.array,i...)

function Base.show(io::IO,::MIME"text/plain",a::PArray{T,N,A,L}) where {T,N,A,L}
  println(io, "PArray of length $L and elements of type $T:")
  println(io,a)
end

function Base.copy(a::PArray{T}) where T
  b = Vector{T}(undef,length(a))
  @inbounds for i = eachindex(a)
    b[i] = copy(a[i])
  end
  PArray(b)
end

function Base.similar(
  a::PArray{T,N,A,L},
  element_type::Type{S}=eltype(a),
  dims::Tuple{Int,Vararg{Int}}=size(a)) where {T,N,A,L,S}

  elb = similar(testitem(a),element_type,dims)
  b = Vector{typeof(elb)}(undef,length(a))
  @inbounds for i = eachindex(a)
    b[i] = similar(a[i],element_type,dims)
  end
  PArray(b)
end

function Base.zero(a::PArray)
  T = eltype(a)
  b = similar(a)
  b .= zero(T)
end

function Base.zeros(a::PArray)
  get_array(zero(a))
end

function allocate_parray(a::AbstractArray,N::Integer)
  PArray([copy(a) for _ = 1:N])
end

function zero_parray(a::AbstractArray{T},N::Integer) where T
  b = similar(a)
  fill!(b,zero(T))
  PArray([copy(b) for _ = 1:N])
end

function Base.sum(a::PArray)
  sum(get_array(a))
end

for op in (:+,:-)
  @eval begin
    function ($op)(a::PArray{T},b::PArray{T}) where T
      array = ($op)(get_array(a),get_array(b))
      PArray(array)
    end

    function ($op)(a::PArray{T},b::AbstractArray{T}) where T
      array = ($op)(get_array(a),b)
      PArray(array)
    end

    function ($op)(a::AbstractArray{T},b::PArray{T}) where T
      array = ($op)(a,get_array(b))
      PArray(array)
    end
  end
end

function Base.:*(a::PArray,b::Number)
  array = get_array(a)*b
  PArray(array)
end

function Base.:*(a::Number,b::PArray)
  b*a
end

function Base.:\(a::PArray{<:AbstractMatrix},b::PArray{<:AbstractVector})
  c = allocate_in_range(a)
  @inbounds for i = eachindex(a)
    c[i] = a[i] \ b[i]
  end
  c
end

function Base.:≈(a::PArray,b::PArray)
  @assert size(a) == size(b)
  for i in eachindex(a)
    if !(a[i] ≈ b[i])
      return false
    end
  end
  true
end

function Base.:(==)(a::PArray,b::PArray)
  @assert size(a) == size(b)
  for i in eachindex(a)
    if !(a[i] == b[i])
      return false
    end
  end
  true
end

function Base.transpose(a::PArray)
  map(transpose,a)
end

function Base.hcat(a::PArray...)
  n = length(first(a))
  harray = map(1:n) do j
    arrays = ()
    @inbounds for i = eachindex(a)
      arrays = (arrays...,a[i][j])
    end
    hcat(arrays...)
  end
  PArray(harray)
end

function Base.vcat(a::PArray...)
  n = length(first(a))
  varray = map(1:n) do j
    arrays = ()
    @inbounds for i = eachindex(a)
      arrays = (arrays...,a[i][j])
    end
    vcat(arrays...)
  end
  PArray(varray)
end

function Base.stack(a::PArray...)
  n = length(first(a))
  harray = map(1:n) do j
    arrays = ()
    @inbounds for i = eachindex(a)
      arrays = (arrays...,a[i][j])
    end
    stack(arrays)
  end
  PArray(harray)
end

function Base.hvcat(nblocks::Int,a::PArray...)
  nrows = Int(length(a)/nblocks)
  varray = map(1:nrows) do row
    vcat(a[(row-1)*nblocks+1:row*nblocks]...)
  end
  hvarray = hcat(varray...)
  hvarray
end

function Base.fill!(a::PArray,z)
  @inbounds for i = eachindex(a)
    ai = a[i]
    fill!(ai,z)
  end
end

function Base.maximum(f,a::PArray)
  map(a) do ai
    maximum(f,ai)
  end
end

function Base.minimum(f,a::PArray)
  map(a) do ai
    minimum(f,ai)
  end
end

function LinearAlgebra.fillstored!(a::PArray,z)
  @inbounds for i = eachindex(a)
    ai = a[i]
    fillstored!(ai,z)
  end
end

function LinearAlgebra.mul!(
  c::PArray,
  a::PArray,
  b::PArray,
  α::Number,β::Number)

  @inbounds for i = eachindex(a)
    ci,ai,bi = c[i],a[i],b[i]
    mul!(ci,ai,bi,α,β)
  end
end

function LinearAlgebra.ldiv!(a::PArray,m::LU,b::PArray)
  @inbounds for i = eachindex(a)
    ai,bi = a[i],b[i]
    ldiv!(ai,m,bi)
  end
end

function LinearAlgebra.ldiv!(a::PArray,m::AbstractArray,b::PArray)
  @inbounds for i = eachindex(a)
    ai,mi,bi = a[i],m[i],b[i]
    ldiv!(ai,mi,bi)
  end
end

function LinearAlgebra.rmul!(a::PArray,b::Number)
  @inbounds for i = eachindex(a)
    ai = a[i]
    rmul!(ai,b)
  end
end

function LinearAlgebra.lu(a::PArray)
  map(a) do ai
    lu(ai)
  end
end

function LinearAlgebra.lu!(a::PArray,b::PArray)
  @inbounds for i = eachindex(a)
    ai,bi = a[i],b[i]
    lu!(ai,bi)
  end
end

function SparseArrays.resize!(a::PArray,args...)
  map(a) do ai
    resize!(ai,args...)
  end
end

function Arrays.CachedArray(a::PArray)
  ai = testitem(a)
  ci = CachedArray(ai)
  array = Vector{typeof(ci)}(undef,length(a))
  @inbounds for i in eachindex(a)
    array[i] = CachedArray(a.array[i])
  end
  PArray(array)
end

function Arrays.setsize!(
  a::PArray{<:CachedArray{T,N}},
  s::NTuple{N,Int}) where {T,N}

  @inbounds for i in eachindex(a)
    setsize!(a.array[i],s)
  end
end

Arrays.get_array(a::PArray{<:CachedArray}) = map(x->x.array,a.array)

function Base.map(f,a::PArray...)
  PArray(map(f,map(get_array,a)...))
end

function Base.map(f,a::PArray,b::AbstractArray...)
  error("DEPRECATED")
  f1 = f(a[1],b...)
  array = Vector{typeof(f1)}(undef,length(a))
  @inbounds for i = eachindex(a)
    array[i] = f(a[i],b...)
  end
  PArray(array)
end

struct PTBroadcasted{T}
  array::PArray{T}
end
_get_pta(a::PArray) = a
_get_pta(a::PTBroadcasted) = a.array

function Base.broadcasted(f,a::Union{PArray,PTBroadcasted}...)
  pta = map(_get_pta,a)
  PTBroadcasted(map(f,pta...))
end

function Base.broadcasted(f,a::Number,b::Union{PArray,PTBroadcasted})
  PTBroadcasted(map(x->f(a,x),_get_pta(b)))
end

function Base.broadcasted(f,a::Union{PArray,PTBroadcasted},b::Number)
  PTBroadcasted(map(x->f(x,b),_get_pta(a)))
end

function Base.broadcasted(
  f,a::Union{PArray,PTBroadcasted},
  b::Broadcast.Broadcasted{Broadcast.DefaultArrayStyle{0}})
  Base.broadcasted(f,a,Base.materialize(b))
end

function Base.broadcasted(
  f,a::Broadcast.Broadcasted{Broadcast.DefaultArrayStyle{0}},
  b::Union{PArray,PTBroadcasted})
  Base.broadcasted(f,Base.materialize(a),b)
end

function Base.materialize(b::PTBroadcasted{T}) where T
  a = similar(b.array)
  Base.materialize!(a,b)
  a
end

function Base.materialize!(a::PArray,b::Broadcast.Broadcasted)
  map(x->Base.materialize!(x,b),a)
  a
end

function Base.materialize!(a::PArray,b::PTBroadcasted)
  map(Base.materialize!,a,b.array)
  a
end

function Base.length(
  a::BroadcastOpFieldArray{O,T,N,<:Tuple{Vararg{Union{Any,PArray}}}}
  ) where {O,T,N}
  pta = filter(x->isa(x,PArray),a.args)
  l = map(length,pta)
  @assert all(l .== first(l))
  return first(l)
end

function Base.size(
  a::BroadcastOpFieldArray{O,T,N,<:Tuple{Vararg{Union{Any,PArray}}}}
  ) where {O,T,N}
  return (length(a),)
end

function Base.getindex(
  a::BroadcastOpFieldArray{O,T,N,<:Tuple{PArray,Any}},
  i::Integer) where {O,T,N}

  ai = a.args[1][i]
  Operation(a.op)(ai,a.args[2])
end

function Base.getindex(
  a::BroadcastOpFieldArray{O,T,N,<:Tuple{Any,PArray}},
  i::Integer) where {O,T,N}

  ai = a.args[2][i]
  Operation(a.op)(a.args[1],ai)
end

function Arrays.return_value(f::Broadcasting{typeof(∘)},a::PArray{<:Field},b::Field)
  args = map(x->return_value(f,x,b),a)
  data = map(x->getproperty(x,:fields),args)
  OperationField(∘,data)
end

function Arrays.return_value(f::Broadcasting{typeof(∘)},a::Field,b::PArray{<:Field})
  args = map(x->return_value(f,a,x),b)
  data = map(x->getproperty(x,:fields),args)
  OperationField(∘,data)
end

for op in (:+,:-,:*)
  @eval begin
    function Arrays.return_value(
      f::Broadcasting{typeof($op)},
      a::PArray)

      v1 = return_value(f,a[1])
      array = Vector{typeof(v1)}(undef,length(a))
      for i = eachindex(a)
        array[i] = return_value(f,a[i])
      end
      PArray(array)
    end

    function Arrays.return_cache(
      f::Broadcasting{typeof($op)},
      a::PArray)

      c1 = return_cache(f,a[1])
      b1 = evaluate!(c1,f,a[1])
      cache = Vector{typeof(c1)}(undef,length(a))
      array = Vector{typeof(b1)}(undef,length(a))
      for i = eachindex(a)
        cache[i] = return_cache(f,a[i])
      end
      cache,PArray(array)
    end

    function Arrays.evaluate!(
      cache,
      f::Broadcasting{typeof($op)},
      a::PArray)

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
      a::PArray,
      b::AbstractArray)

      v1 = return_value(f,a[1],b)
      array = Vector{typeof(v1)}(undef,length(a))
      for i = eachindex(a)
        array[i] = return_value(f,a[i],b)
      end
      PArray(array)
    end

    function Arrays.return_value(
      f::Broadcasting{typeof($op)},
      a::AbstractArray,
      b::PArray)

      v1 = return_value(f,a,b[1])
      array = Vector{typeof(v1)}(undef,length(b))
      for i = eachindex(b)
        array[i] = return_value(f,a,b[i])
      end
      PArray(array)
    end

    function Arrays.return_value(
      f::Broadcasting{typeof($op)},
      a::PArray,
      b::PArray)

      v1 = return_value(f,a[1],b[1])
      array = Vector{typeof(v1)}(undef,length(a))
      for i = eachindex(a)
        array[i] = return_value(f,a[i],b[i])
      end
      PArray(array)
    end

    function Arrays.return_cache(
      f::Broadcasting{typeof($op)},
      a::PArray,
      b::AbstractArray)

      c1 = return_cache(f,a[1],b)
      b1 = evaluate!(c1,f,a[1],b)
      cache = Vector{typeof(c1)}(undef,length(a))
      array = Vector{typeof(b1)}(undef,length(a))
      for i = eachindex(a)
        cache[i] = return_cache(f,a[i],b)
      end
      cache,PArray(array)
    end

    function Arrays.return_cache(
      f::Broadcasting{typeof($op)},
      a::AbstractArray,
      b::PArray)

      c1 = return_cache(f,a,b[1])
      b1 = evaluate!(c1,f,a,b[1])
      cache = Vector{typeof(c1)}(undef,length(b))
      array = Vector{typeof(b1)}(undef,length(b))
      for i = eachindex(b)
        cache[i] = return_cache(f,a,b[i])
      end
      cache,PArray(array)
    end

    function Arrays.return_cache(
      f::Broadcasting{typeof($op)},
      a::PArray,
      b::PArray)

      c1 = return_cache(f,a[1],b[1])
      b1 = evaluate!(c1,f,a[1],b[1])
      cache = Vector{typeof(c1)}(undef,length(a))
      array = Vector{typeof(b1)}(undef,length(a))
      for i = eachindex(a)
        cache[i] = return_cache(f,a[i],b[i])
      end
      cache,PArray(array)
    end

    function Arrays.evaluate!(
      cache,
      f::Broadcasting{typeof($op)},
      a::PArray,
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
      b::PArray)

      cx,array = cache
      @inbounds for i = eachindex(array)
        array[i] = evaluate!(cx[i],f,a,b[i])
      end
      array
    end

    function Arrays.evaluate!(
      cache,
      f::Broadcasting{typeof($op)},
      a::PArray,
      b::PArray)

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
  a::PArray,
  b::Number)

  v1 = return_value(f,a[1],b)
  array = Vector{typeof(v1)}(undef,length(a))
  for i = eachindex(a)
    array[i] = return_value(f,a[i],b)
  end
  PArray(array)
end

function Arrays.return_cache(
  f::Broadcasting{typeof(*)},
  a::PArray,
  b::Number)

  c1 = return_cache(f,a[1],b)
  b1 = evaluate!(c1,f,a[1],b)
  cache = Vector{typeof(c1)}(undef,length(a))
  array = Vector{typeof(b1)}(undef,length(a))
  for i = eachindex(a)
    cache[i] = return_cache(f,a[i],b)
  end
  cache,PArray(array)
end

function Arrays.evaluate!(
  cache,
  f::Broadcasting{typeof(*)},
  a::PArray,
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
  b::PArray)

  return_value(f,b,a)
end

function Arrays.return_cache(
  f::Broadcasting{typeof(*)},
  a::Number,
  b::PArray)

  return_cache(f,b,a)
end

function Arrays.evaluate!(
  cache,
  f::Broadcasting{typeof(*)},
  a::Number,
  b::PArray)

  evaluate!(cache,f,b,a)
end

function Arrays.return_value(
  f::BroadcastingFieldOpMap,
  a::PArray,
  b::AbstractArray)

  v1 = return_value(f,a[1],b)
  array = Vector{typeof(v1)}(undef,length(a))
  for i = eachindex(a)
    array[i] = return_value(f,a[i],b)
  end
  PArray(array)
end

function Arrays.return_value(
  f::BroadcastingFieldOpMap,
  a::AbstractArray,
  b::PArray)

  v1 = return_value(f,a,b[1])
  array = Vector{typeof(v1)}(undef,length(b))
  for i = eachindex(b)
    array[i] = return_value(f,a,b[i])
  end
  PArray(array)
end

function Arrays.return_value(
  f::BroadcastingFieldOpMap,
  a::PArray,
  b::PArray)

  v1 = return_value(f,a[1],b[1])
  array = Vector{typeof(v1)}(undef,length(a))
  for i = eachindex(a)
    array[i] = return_value(f,a[i],b[i])
  end
  PArray(array)
end

function Arrays.return_cache(
  f::BroadcastingFieldOpMap,
  a::PArray,
  b::AbstractArray)

  c1 = return_cache(f,a[1],b)
  b1 = evaluate!(c1,f,a[1],b)
  cache = Vector{typeof(c1)}(undef,length(a))
  array = Vector{typeof(b1)}(undef,length(a))
  for i = eachindex(a)
    cache[i] = return_cache(f,a[i],b)
  end
  cache,PArray(array)
end

function Arrays.return_cache(
  f::BroadcastingFieldOpMap,
  a::AbstractArray,
  b::PArray)

  c1 = return_cache(f,a,b[1])
  b1 = evaluate!(c1,f,a,b[1])
  cache = Vector{typeof(c1)}(undef,length(b))
  array = Vector{typeof(b1)}(undef,length(b))
  for i = eachindex(b)
    cache[i] = return_cache(f,a,b[i])
  end
  cache,PArray(array)
end

function Arrays.return_cache(
  f::BroadcastingFieldOpMap,
  a::PArray,
  b::PArray)

  c1 = return_cache(f,a[1],b[1])
  b1 = evaluate!(c1,f,a[1],b[1])
  cache = Vector{typeof(c1)}(undef,length(a))
  array = Vector{typeof(b1)}(undef,length(a))
  for i = eachindex(a)
    cache[i] = return_cache(f,a[i],b[i])
  end
  cache,PArray(array)
end

function Arrays.evaluate!(
  cache,
  f::BroadcastingFieldOpMap,
  a::PArray{<:AbstractArray{T,N}},
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
  b::PArray{<:AbstractArray{S,N}}) where {T,S,N}

  cx,array = cache
  @inbounds for i = eachindex(array)
    array[i] = evaluate!(cx[i],f,a,b[i])
  end
  array
end

function Arrays.evaluate!(
  cache,
  f::BroadcastingFieldOpMap,
  a::PArray{<:AbstractArray{T,N}},
  b::PArray{<:AbstractArray{S,N}}) where {T,S,N}

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
        a::PArray{<:$S},
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
        b::PArray{<:$T})

        cx,array = cache
        @inbounds for i = eachindex(array)
          array[i] = evaluate!(cx[i],f,a,b[i])
        end
        array
      end

      function Arrays.evaluate!(
        cache,
        f::BroadcastingFieldOpMap,
        a::PArray{<:$S},
        b::PArray{<:$T})

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
      a::PArray{<:$S},
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
      b::PArray{<:AbstractArray{U,3}} where U)

      cx,array = cache
      @inbounds for i = eachindex(array)
        array[i] = evaluate!(cx[i],f,a,b[i])
      end
      array
    end

    function Arrays.evaluate!(
      cache,
      f::BroadcastingFieldOpMap,
      a::PArray{<:AbstractArray{U,3}} where U,
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
      b::PArray{<:$S})

      cx,array = cache
      @inbounds for i = eachindex(array)
        array[i] = evaluate!(cx[i],f,a,b[i])
      end
      array
    end
  end
end

function Base.getindex(k::LinearCombinationField{<:PArray},i::Int)
  LinearCombinationField(k.values[i],k.fields,k.column)
end

for T in (:(Point),:(AbstractVector{<:Point}))
  @eval begin
    function Arrays.return_value(a::LinearCombinationField{<:PArray},x::$T)
      v1 = return_value(a[1],x)
      array = Vector{typeof(v1)}(undef,length(a.values))
      for i = eachindex(a.values)
        array[i] = return_value(a[i],x)
      end
      PArray(array)
    end

    function Arrays.return_cache(a::LinearCombinationField{<:PArray},x::$T)
      c1 = return_cache(a[1],x)
      b1 = evaluate!(c1,a[1],x)
      cache = Vector{typeof(c1)}(undef,length(a.values))
      array = Vector{typeof(b1)}(undef,length(a.values))
      for i = eachindex(a.values)
        cache[i] = return_cache(a[i],x)
      end
      cache,PArray(array)
    end

    function Arrays.evaluate!(cache,a::LinearCombinationField{<:PArray},x::$T)
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
        v::PArray{<:$S},
        fx::$T)

        v1 = return_value(k,v[1],fx)
        array = Vector{typeof(v1)}(undef,length(v))
        for i = eachindex(v)
          array[i] = return_value(k,v[i],fx)
        end
        PArray(array)
      end

      function Arrays.return_cache(
        k::LinearCombinationMap{<:Integer},
        v::PArray{<:$S},
        fx::$T)

        c1 = return_cache(k,v[1],fx)
        b1 = evaluate!(c1,k,v[1],fx)
        cache = Vector{typeof(c1)}(undef,length(v))
        array = Vector{typeof(b1)}(undef,length(v))
        for i = eachindex(v)
          cache[i] = return_cache(k,v[i],fx)
        end
        cache,PArray(array)
      end

      function Arrays.evaluate!(
        cache,
        k::LinearCombinationMap{<:Integer},
        v::PArray{<:$S},
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

function Fields.linear_combination(a::PArray,b::AbstractArray)
  ab1 = linear_combination(a[1],b)
  c = Vector{typeof(ab1)}(undef,length(a))
  for i in eachindex(a)
    c[i] = linear_combination(a[i],b)
  end
  PArray(c)
end

function Base.getindex(k::Broadcasting{<:PosNegReindex{<:PArray,<:PArray}},i::Int)
  fi = PosNegReindex(k.f.values_pos[i],k.f.values_neg[i])
  Broadcasting(fi)
end

function Arrays.return_value(
  k::Broadcasting{<:PosNegReindex{<:PArray,<:PArray}},
  x::Union{Number,AbstractArray{<:Number}}...)

  npos = length(k.f.values_pos)
  nneg = length(k.f.values_neg)
  @assert npos == nneg
  v1 = return_value(k[1],x...)
  array = Vector{typeof(v1)}(undef,npos)
  for i = 1:npos
    array[i] = return_value(k[i],x...)
  end
  PArray(array)
end

function Arrays.return_cache(
  k::Broadcasting{<:PosNegReindex{<:PArray,<:PArray}},
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
  cache,PArray(array)
end

function Arrays.evaluate!(
  cache,
  k::Broadcasting{<:PosNegReindex{<:PArray,<:PArray}},
  x::Union{Number,AbstractArray{<:Number}}...)

  cx,array = cache
  @inbounds for i = eachindex(array)
    array[i] = evaluate!(cx[i],k[i],x...)
  end
  array
end

function Arrays.evaluate!(
  cache,
  k::Broadcasting{<:PosNegReindex{<:PArray,<:PArray}},
  x::AbstractArray{<:Number})

  cx,array = cache
  @inbounds for i = eachindex(array)
    array[i] = evaluate!(cx[i],k[i],x)
  end
  array
end

function Arrays.evaluate!(
  cache,
  k::Broadcasting{<:PosNegReindex{<:PArray,<:PArray}},
  x::Number...)

  cx,array = cache
  @inbounds for i = eachindex(array)
    array[i] = evaluate!(cx[i],k[i],x...)
  end
  array
end

function Arrays.return_value(
  f::IntegrationMap,
  a::PArray,
  w,
  j::AbstractVector)

  v1 = return_value(f,a[1],w,j)
  array = Vector{typeof(v1)}(undef,length(a))
  for i = eachindex(a)
    array[i] = return_value(f,a[i],w,j)
  end
  PArray(array)
end

function Arrays.return_cache(
  f::IntegrationMap,
  a::PArray,
  w,
  j::AbstractVector)

  c1 = return_cache(f,a[1],w,j)
  b1 = evaluate!(c1,f,a[1],w,j)
  cache = Vector{typeof(c1)}(undef,length(a))
  array = Vector{typeof(b1)}(undef,length(a))
  for i = eachindex(a)
    cache[i] = return_cache(f,a[i],w,j)
  end
  cache,PArray(array)
end

function Arrays.evaluate!(
  cache,
  f::IntegrationMap,
  a::PArray,
  w,
  j::AbstractVector)

  cx,array = cache
  @inbounds for i = eachindex(array)
    array[i] = evaluate!(cx[i],f,a[i],w,j)
  end
  array
end

function Arrays.return_value(
  f::Broadcasting{typeof(MultiField._sum_if_first_positive)},
  dofs::PArray{<:VectorBlock},
  o)

  v1 = return_value(f,dofs[1],o)
  allocate_parray(v1,length(dofs))
end

function Arrays.return_cache(
  f::Broadcasting{typeof(MultiField._sum_if_first_positive)},
  dofs::PArray{<:VectorBlock},
  o)

  c1 = return_cache(f,dofs[1],o)
  b1 = evaluate!(c1,f,dofs[1],o)
  cache = Vector{typeof(c1)}(undef,length(dofs))
  array = Vector{typeof(b1)}(undef,length(dofs))
  for i = eachindex(dofs)
    cache[i] = return_cache(f,dofs[i],o)
  end
  cache,PArray(array)
end

function Arrays.evaluate!(
  cache,
  f::Broadcasting{typeof(MultiField._sum_if_first_positive)},
  dofs::PArray{<:VectorBlock},
  o)

  cx,array = cache
  @inbounds for i = eachindex(array)
    array[i] = evaluate!(cx[i],f,dofs[i],o)
  end
  array
end

function Utils.recenter(a::PArray{T},a0::PArray{T};kwargs...) where T
  n = length(a)
  n0 = length(a0)
  ndiff = Int(n/n0)
  array = Vector{T}(undef,n)
  @inbounds for i = 1:n0
    array[(i-1)*ndiff+1:i*ndiff] = recenter(a[(i-1)*ndiff+1:i*ndiff],a0[i];kwargs...)
  end
  PArray(array)
end

function get_at_offsets(x::PArray{<:AbstractVector},offsets::Vector{Int},row::Int)
  map(y->y[offsets[row]+1:offsets[row+1]],x)
end

function get_at_offsets(x::PArray{<:AbstractMatrix},offsets::Vector{Int},row::Int,col::Int)
  map(y->y[offsets[row]+1:offsets[row+1],offsets[col]+1:offsets[col+1]],x)
end
