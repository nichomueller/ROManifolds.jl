struct PArray{T,N,A,L} <: AbstractArray{T,N}
  array::A
  function PArray(array::AbstractVector{T},::Val{L}) where {T<:AbstractArray,L}
    N = ndims(T)
    A = typeof(array)
    new{T,N,A,L}(array)
  end
  function PArray(array::A,::Val{L}) where {A,L}
    T = eltype(array)
    N = 1
    new{T,N,A,L}(array)
  end
end

const AffinePArray{T,N,A} = PArray{T,N,A,1}

function PArray(array)
  PArray(array,Val(length(array)))
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
Base.iterate(a::PArray,i...) = iterate(get_array(a),i...)

function Base.show(io::IO,::MIME"text/plain",a::PArray{T,N,A,L}) where {T,N,A,L}
  println(io, "PArray{T = $T, L = $L} with entries:")
  show(io,a.array)
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
  b
end

function Base.zeros(a::PArray)
  get_array(zero(a))
end

function Arrays.testvalue(::Type{PArray{T,N,A,L}}) where {T,N,A,L}
  array = map(1:L) do i
    testvalue(T)
  end
  PArray(array,Val(L))
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

function Base.:+(a::T,b::T) where T<:PArray
  map(a,b) do a,b
    a+b
  end
end

function Base.:-(a::T,b::T) where T<:PArray
  map(a,b) do a,b
    a-b
  end
end

function Base.:+(a::PArray{T},b::T) where T<:AbstractArray
  map(a) do a
    a+b
  end
end

function Base.:+(a::T,b::PArray{T}) where T<:AbstractArray
  map(b) do b
    a+b
  end
end

function Base.:-(a::PArray{T},b::T) where T<:AbstractArray
  map(a) do a
    a-b
  end
end

function Base.:-(a::T,b::PArray{T}) where T<:AbstractArray
  map(b) do b
    a-b
  end
end

(Base.:-)(a::PArray) = a .* -1

function Base.:*(a::PArray,b::Number)
  PArray(get_array(a)*b)
end

function Base.:*(a::Number,b::PArray)
  b*a
end

function Base.:*(a::PArray{<:AbstractMatrix},b::PArray{<:AbstractVector})
  map(a,b) do a,b
    a*b
  end
end

function Base.:\(a::PArray{<:AbstractMatrix},b::PArray{<:AbstractVector})
  map(a,b) do a,b
    a\b
  end
end

function Base.:≈(a::PArray,b::PArray)
  bools = map(a,b) do a,b
    a ≈ b
  end
  all(bools)
end

function Base.:(==)(a::PArray,b::PArray)
  bools = map(a,b) do a,b
    a == b
  end
  all(bools)
end

function Base.transpose(a::PArray)
  map(transpose,a)
end

function Base.stack(a::Tuple{Vararg{PArray{T,N,A,L}}}) where {T,N,A,L}
  arrays = map(get_array,a)
  array = map(1:L) do i
    stack(map(b->getindex(b,i),arrays))
  end
  PArray(array)
end

function Base.hcat(a::PArray{T,N,A,L}...) where {T,N,A,L}
  arrays = map(get_array,a)
  array = map(1:L) do i
    hcat(map(b->getindex(b,i),arrays)...)
  end
  PArray(array)
end

function Base.vcat(a::PArray{T,N,A,L}...) where {T,N,A,L}
  arrays = map(get_array,a)
  array = map(1:L) do i
    vcat(map(b->getindex(b,i),arrays)...)
  end
  PArray(array)
end

function Base.hvcat(nblocks::Int,a::PArray...)
  nrows = Int(length(a)/nblocks)
  varray = map(1:nrows) do row
    vcat(a[(row-1)*nblocks+1:row*nblocks]...)
  end
  stack(varray)
end

function Base.fill!(a::PArray,z)
  map(a) do a
    fill!(a,z)
  end
end

function Base.maximum(f,a::PArray)
  map(a) do a
    maximum(f,a)
  end
end

function Base.minimum(f,a::PArray)
  map(a) do a
    minimum(f,a)
  end
end

function LinearAlgebra.fillstored!(a::PArray,z)
  map(a) do a
    fillstored!(a,z)
  end
end

function LinearAlgebra.mul!(
  c::PArray,
  a::PArray,
  b::PArray,
  α::Number,β::Number)

  map(c,a,b) do c,a,b
    mul!(c,a,b,α,β)
  end
end

function LinearAlgebra.ldiv!(a::PArray,m::LU,b::PArray)
  map(a,b) do a,b
    ldiv!(a,m,b)
  end
end

function LinearAlgebra.ldiv!(a::PArray,m::AbstractArray,b::PArray)
  @assert length(a) == length(m) == length(b)
  map(a,m,b) do a,m,b
    ldiv!(a,m,b)
  end
end

function LinearAlgebra.rmul!(a::PArray,b::Number)
  map(a) do a
    rmul!(a,b)
  end
end

function LinearAlgebra.lu(a::PArray)
  map(a) do a
    lu(a)
  end
end

function LinearAlgebra.lu!(a::PArray,b::PArray)
  map(a,b) do a,b
    lu!(a,b)
  end
end

function SparseArrays.resize!(a::PArray,args...)
  map(a) do a
    resize!(a,args...)
  end
end

function Arrays.CachedArray(a::PArray)
  ai = testitem(a)
  ci = CachedArray(ai)
  array = Vector{typeof(ci)}(undef,length(a))
  @inbounds for i in eachindex(a)
    array[i] = CachedArray(a[i])
  end
  PArray(array)
end

function Arrays.setsize!(
  a::PArray{<:CachedArray{T,N}},
  s::NTuple{N,Int}) where {T,N}

  map(a) do a
    setsize!(a,s)
  end
end

function Arrays.SubVector(a::PArray,pini::Int,pend::Int)
  map(a) do vector
    SubVector(vector,pini,pend)
  end
end

function Base.map(f,a::PArray...)
  PArray(map(f,map(get_array,a)...))
end

struct PBroadcast{D}
  dest::D
end
Arrays.get_array(a::PBroadcast) = a.dest

function Base.broadcasted(f,args::Union{PArray,PBroadcast}...)
  arrays = map(get_array,args)
  g = (x...) -> map(f,(x...))
  fargs = map((x...)->Base.broadcasted(g,x...),arrays...)
  PBroadcast(fargs)
end

function Base.broadcasted(f,a::Number,b::Union{PArray,PBroadcast})
  barray = get_array(b)
  fab = map(x->Base.broadcasted(f,a,x),barray)
  PBroadcast(fab)
end

function Base.broadcasted(f,a::Union{PArray,PBroadcast},b::Number)
  aarray = get_array(a)
  fab = map(x->Base.broadcasted(f,x,b),aarray)
  PBroadcast(fab)
end

function Base.broadcasted(
  f,
  a::Union{PArray,PBroadcast},
  b::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}})
  Base.broadcasted(f,a,Base.materialize(b))
end

function Base.broadcasted(
  f,
  a::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}},
  b::Union{PArray,PBroadcast})
  Base.broadcasted(f,Base.materialize(a),b)
end

function Base.materialize(b::PBroadcast)
  barray = get_array(b)
  dest = map(Base.materialize,barray)
  T = eltype(dest)
  L = length(dest)
  a = Vector{T}(undef,L)
  Base.materialize!(a,dest)
  PArray(a)
end

function Base.materialize!(a::PArray,b::PBroadcast)
  map(Base.materialize!,get_array(a),get_array(b))
  a
end

function Arrays.return_value(
  f::BroadcastingFieldOpMap,
  a::PArray,
  b::AbstractArray)

  vi = return_value(f,testitem(a),b)
  array = Vector{typeof(vi)}(undef,length(a))
  for i = eachindex(a)
    array[i] = return_value(f,a[i],b)
  end
  PArray(array)
end

function Arrays.return_value(
  f::BroadcastingFieldOpMap,
  a::AbstractArray,
  b::PArray)

  vi = return_value(f,a,testitem(b))
  array = Vector{typeof(vi)}(undef,length(b))
  for i = eachindex(b)
    array[i] = return_value(f,a,b[i])
  end
  PArray(array)
end

function Arrays.return_value(
  f::BroadcastingFieldOpMap,
  a::PArray,
  b::PArray)

  vi = return_value(f,testitem(a),testitem(b))
  array = Vector{typeof(vi)}(undef,length(a))
  for i = eachindex(a)
    array[i] = return_value(f,a[i],b[i])
  end
  PArray(array)
end

function Arrays.return_cache(
  f::BroadcastingFieldOpMap,
  a::PArray,
  b::AbstractArray)

  ci = return_cache(f,testitem(a),b)
  bi = evaluate!(ci,f,testitem(a),b)
  cache = Vector{typeof(ci)}(undef,length(a))
  array = Vector{typeof(bi)}(undef,length(a))
  for i = eachindex(a)
    cache[i] = return_cache(f,a[i],b)
  end
  cache,PArray(array)
end

function Arrays.return_cache(
  f::BroadcastingFieldOpMap,
  a::AbstractArray,
  b::PArray)

  ci = return_cache(f,a,testitem(b))
  bi = evaluate!(ci,f,a,testitem(b))
  cache = Vector{typeof(ci)}(undef,length(b))
  array = Vector{typeof(bi)}(undef,length(b))
  for i = eachindex(b)
    cache[i] = return_cache(f,a,b[i])
  end
  cache,PArray(array)
end

function Arrays.return_cache(
  f::BroadcastingFieldOpMap,
  a::PArray,
  b::PArray)

  ci = return_cache(f,testitem(a),testitem(b))
  bi = evaluate!(ci,f,testitem(a),testitem(b))
  cache = Vector{typeof(ci)}(undef,length(a))
  array = Vector{typeof(bi)}(undef,length(a))
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
      vi = return_value(testitem(a),x)
      array = Vector{typeof(vi)}(undef,length(a.values))
      for i = eachindex(a.values)
        array[i] = return_value(a[i],x)
      end
      PArray(array)
    end

    function Arrays.return_cache(a::LinearCombinationField{<:PArray},x::$T)
      ci = return_cache(testitem(a),x)
      bi = evaluate!(ci,testitem(a),x)
      cache = Vector{typeof(ci)}(undef,length(a.values))
      array = Vector{typeof(bi)}(undef,length(a.values))
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

        vi = return_value(k,testitem(v),fx)
        array = Vector{typeof(vi)}(undef,length(v))
        for i = eachindex(v)
          array[i] = return_value(k,v[i],fx)
        end
        PArray(array)
      end

      function Arrays.return_cache(
        k::LinearCombinationMap{<:Integer},
        v::PArray{<:$S},
        fx::$T)

        ci = return_cache(k,testitem(v),fx)
        bi = evaluate!(ci,k,testitem(v),fx)
        cache = Vector{typeof(ci)}(undef,length(v))
        array = Vector{typeof(bi)}(undef,length(v))
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
  abi = linear_combination(testitem(a),b)
  c = Vector{typeof(abi)}(undef,length(a))
  for i in eachindex(a)
    c[i] = linear_combination(a[i],b)
  end
  PArray(c)
end

function Arrays.return_value(
  f::IntegrationMap,
  a::PArray,
  w)

  vi = return_value(f,testitem(a),w)
  array = Vector{typeof(vi)}(undef,length(a))
  for i = eachindex(a)
    array[i] = return_value(f,a[i],w)
  end
  PArray(array)
end

function Arrays.return_cache(
  f::IntegrationMap,
  a::PArray,
  w)

  ci = return_cache(f,testitem(a),w)
  bi = evaluate!(ci,f,testitem(a),w)
  cache = Vector{typeof(ci)}(undef,length(a))
  array = Vector{typeof(bi)}(undef,length(a))
  for i = eachindex(a)
    cache[i] = return_cache(f,a[i],w)
  end
  cache,PArray(array)
end

function Arrays.evaluate!(
  cache,
  f::IntegrationMap,
  a::PArray,
  w)

  cx,array = cache
  @inbounds for i = eachindex(array)
    array[i] = evaluate!(cx[i],f,a[i],w)
  end
  array
end

function Arrays.return_value(
  f::IntegrationMap,
  a::PArray,
  w,
  j::AbstractVector)

  vi = return_value(f,testitem(a),w,j)
  array = Vector{typeof(vi)}(undef,length(a))
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

  ci = return_cache(f,testitem(a),w,j)
  bi = evaluate!(ci,f,testitem(a),w,j)
  cache = Vector{typeof(ci)}(undef,length(a))
  array = Vector{typeof(bi)}(undef,length(a))
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

function Arrays.return_cache(::Fields.ZeroBlockMap,a::AbstractArray,b::PArray)
  _a = allocate_parray(a,length(b))
  CachedArray(similar(_a,eltype(a),size(b)))
end

function Arrays.return_cache(::Fields.ZeroBlockMap,a::PArray,b::PArray)
  CachedArray(similar(a,eltype(a),size(b)))
end

function Arrays.evaluate!(cache::PArray,f::Fields.ZeroBlockMap,a,b::AbstractArray)
  _get_array(c::CachedArray) = c.array
  @inbounds for i = eachindex(cache)
    evaluate!(cache[i],f,a,b)
  end
  map(_get_array,cache)
end

function Arrays.evaluate!(cache::PArray,f::Fields.ZeroBlockMap,a,b::PArray)
  _get_array(c::CachedArray) = c.array
  @inbounds for i = eachindex(cache)
    evaluate!(cache[i],f,a,b[i])
  end
  map(_get_array,cache)
end

# function Arrays.return_cache(
#   f::Broadcasting{typeof(MultiField._sum_if_first_positive)},
#   dofs::PArray,
#   o)

#   dofsi = testitem(dofs)
#   ci = return_cache(f,dofsi,o)
#   bi = evaluate!(ci,f,dofsi,o)
#   cache = Vector{typeof(ci)}(undef,length(dofs))
#   array = Vector{typeof(bi)}(undef,length(dofs))
#   for i = eachindex(dofs)
#     cache[i] = return_cache(f,dofs[i],o)
#   end
#   cache,PArray(array)
# end

# function Arrays.evaluate!(
#   cache,
#   f::Broadcasting{typeof(MultiField._sum_if_first_positive)},
#   dofs::PArray,
#   o)

#   cx,array = cache
#   @inbounds for i = eachindex(dofs)
#     array[i] = evaluate!(cx[i],f,dofs[i],o)
#   end
#   array
# end

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
