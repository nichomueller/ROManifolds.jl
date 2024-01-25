struct ParamArray{T,N,A<:AbstractVector{<:AbstractArray{T,N}},L} <: AbstractParamContainer{T,N}
  array::A
  function ParamArray(array::A,::Val{L}) where {T,N,A<:AbstractVector{<:AbstractArray{T,N}},L}
    new{T,N,A,L}(array)
  end
end

const ParamVector{T,A,L} = ParamArray{T,1,A,L}
const ParamMatrix{T,A,L} = ParamArray{T,2,A,L}

const AffineParamArray{T,N,A} = ParamArray{T,N,A,1}
const AffineParamVector{T,A} = ParamVector{T,A,1}
const AffineParamMatrix{T,A} = ParamMatrix{T,A,1}

function ParamArray(array)
  ParamArray(array,Val(length(array)))
end

function ParamArray(array::AbstractVector{T}) where {T<:Number}
  array
end

function ParamVector{V}(::UndefInitializer,L::Integer) where V<:AbstractVector
  array = Vector{V}(undef,L)
  ParamArray(array)
end

function ParamMatrix{M}(::UndefInitializer,L::Integer) where M<:AbstractMatrix
  array = Vector{M}(undef,L)
  ParamArray(array)
end

Arrays.get_array(a::ParamArray) = a.array
Arrays.testitem(a::ParamArray) = testitem(get_array(a))
Base.length(::ParamArray{T,N,A,L}) where {T,N,A,L} = L
Base.length(::Type{ParamArray{T,N,A,L}}) where {T,N,A,L} = L
Base.size(a::ParamArray,i...) = size(testitem(a),i...)
Base.axes(a::ParamArray,i...) = axes(testitem(a))
Base.eltype(::ParamArray{T,N,A,L}) where {T,N,A,L} = T
Base.eltype(::Type{ParamArray{T,N,A,L}}) where {T,N,A,L} = T
Base.ndims(::ParamArray{T,N,A,L}) where {T,N,A,L} = N
Base.ndims(::Type{ParamArray{T,N,A,L}}) where {T,N,A,L} = N
Base.first(a::ParamArray) = testitem(a)
Base.eachindex(::ParamArray{T,N,A,L}) where {T,N,A,L} = Base.OneTo(L)
Base.lastindex(::ParamArray{T,N,A,L}) where {T,N,A,L} = L
Base.getindex(a::ParamArray,i...) = get_array(a)[i...]
Base.setindex!(a::ParamArray,v,i...) = get_array(a)[i...] = v
Base.iterate(a::ParamArray,i...) = iterate(get_array(a),i...)

function Base.show(io::IO,::MIME"text/plain",a::ParamArray{T,N,A,L}) where {T,N,A,L}
  println(io, "Parametric vector of types $(eltype(A)) and length $L, with entries:")
  show(io,a.array)
end

function Base.copy(a::ParamArray)
  ai = testitem(a)
  b = Vector{typeof(ai)}(undef,length(a))
  @inbounds for i = eachindex(a)
    b[i] = copy(a[i])
  end
  ParamArray(b)
end

function Base.similar(
  a::ParamArray{T},
  element_type::Type{S}=T,
  dims::Tuple{Int,Vararg{Int}}=size(a)) where {T,S}

  elb = similar(testitem(a),element_type,dims)
  b = Vector{typeof(elb)}(undef,length(a))
  @inbounds for i = eachindex(a)
    b[i] = similar(a[i],element_type,dims)
  end
  ParamArray(b)
end

function Base.zero(a::ParamArray)
  T = eltype(a)
  b = similar(a)
  b .= zero(T)
  b
end

function Base.zeros(a::ParamArray)
  get_array(zero(a))
end

function Arrays.testvalue(::Type{ParamArray{T,N,A,L}}) where {T,N,A,L}
  array = map(1:L) do i
    testvalue(eltype(A))
  end
  ParamArray(array,Val(L))
end

function allocate_param_array(a::AbstractArray,L::Integer)
  ParamArray([copy(a) for _ = 1:L])
end

function zero_param_array(a::AbstractArray{T},L::Integer) where T
  b = similar(a)
  fill!(b,zero(T))
  ParamArray([copy(b) for _ = 1:L])
end

function Base.sum(a::ParamArray)
  sum(get_array(a))
end

function Base.:+(a::T,b::T) where T<:ParamArray
  map(a,b) do a,b
    a+b
  end
end

function Base.:-(a::T,b::T) where T<:ParamArray
  map(a,b) do a,b
    a-b
  end
end

function Base.:+(a::ParamArray{T},b::S) where {T,S<:AbstractArray{T}}
  map(a) do a
    a+b
  end
end

function Base.:+(a::S,b::ParamArray{T}) where {T,S<:AbstractArray{T}}
  map(b) do b
    a+b
  end
end

function Base.:-(a::ParamArray{T},b::S) where {T,S<:AbstractArray{T}}
  map(a) do a
    a-b
  end
end

function Base.:-(a::S,b::ParamArray{T}) where {T,S<:AbstractArray{T}}
  map(b) do b
    a-b
  end
end

(Base.:-)(a::ParamArray) = a .* -1

function Base.:*(a::ParamArray,b::Number)
  ParamArray(get_array(a)*b)
end

function Base.:*(a::Number,b::ParamArray)
  b*a
end

function Base.:*(a::ParamMatrix,b::ParamVector)
  map(a,b) do a,b
    a*b
  end
end

function Base.:\(a::ParamMatrix,b::ParamVector)
  map(a,b) do a,b
    a\b
  end
end

function Base.:≈(a::ParamArray,b::ParamArray)
  bools = map(a,b) do a,b
    a ≈ b
  end
  all(bools)
end

function Base.:(==)(a::ParamArray,b::ParamArray)
  bools = map(a,b) do a,b
    a == b
  end
  all(bools)
end

function Base.transpose(a::ParamArray)
  map(transpose,a)
end

function Base.stack(a::Tuple{Vararg{ParamArray{T,N,A,L}}}) where {T,N,A,L}
  arrays = map(get_array,a)
  array = map(1:L) do i
    stack(map(b->getindex(b,i),arrays))
  end
  ParamArray(array)
end

function Base.hcat(a::ParamArray{T,N,A,L}...) where {T,N,A,L}
  arrays = map(get_array,a)
  array = map(1:L) do i
    hcat(map(b->getindex(b,i),arrays)...)
  end
  ParamArray(array)
end

function Base.vcat(a::ParamArray{T,N,A,L}...) where {T,N,A,L}
  arrays = map(get_array,a)
  array = map(1:L) do i
    vcat(map(b->getindex(b,i),arrays)...)
  end
  ParamArray(array)
end

function Base.hvcat(nblocks::Int,a::ParamArray...)
  nrows = Int(length(a)/nblocks)
  varray = map(1:nrows) do row
    vcat(a[(row-1)*nblocks+1:row*nblocks]...)
  end
  stack(varray)
end

function Base.fill!(a::ParamArray,z)
  map(a) do a
    fill!(a,z)
  end
end

function Base.maximum(f,a::ParamArray)
  map(a) do a
    maximum(f,a)
  end
end

function Base.minimum(f,a::ParamArray)
  map(a) do a
    minimum(f,a)
  end
end

function LinearAlgebra.fillstored!(a::ParamArray,z)
  map(a) do a
    fillstored!(a,z)
  end
end

function LinearAlgebra.mul!(
  c::ParamArray,
  a::ParamArray,
  b::ParamArray,
  α::Number,β::Number)

  map(c,a,b) do c,a,b
    mul!(c,a,b,α,β)
  end
end

function LinearAlgebra.ldiv!(a::ParamArray,m::LU,b::ParamArray)
  map(a,b) do a,b
    ldiv!(a,m,b)
  end
end

function LinearAlgebra.ldiv!(a::ParamArray,m::AbstractArray,b::ParamArray)
  @assert length(a) == length(m) == length(b)
  map(a,m,b) do a,m,b
    ldiv!(a,m,b)
  end
end

function LinearAlgebra.rmul!(a::ParamArray,b::Number)
  map(a) do a
    rmul!(a,b)
  end
end

function LinearAlgebra.lu(a::ParamArray)
  map(a) do a
    lu(a)
  end
end

function LinearAlgebra.lu!(a::ParamArray,b::ParamArray)
  map(a,b) do a,b
    lu!(a,b)
  end
end

function SparseArrays.resize!(a::ParamArray,args...)
  map(a) do a
    resize!(a,args...)
  end
end

function Arrays.CachedArray(a::ParamArray)
  map(a) do a
    CachedArray(a)
  end
end

function Arrays.setsize!(
  a::ParamArray{T,N,AbstractVector{CachedArray{T,N}}},
  s::NTuple{N,Int}) where {T,N}

  map(a) do a
    setsize!(a,s)
  end
end

function Arrays.SubVector(a::ParamArray,pini::Int,pend::Int)
  map(a) do vector
    SubVector(vector,pini,pend)
  end
end

function Base.map(f,a::ParamArray...)
  ParamArray(map(f,map(get_array,a)...))
end

struct PBroadcast{D}
  dest::D
end

Arrays.get_array(a::PBroadcast) = a.dest

function Base.broadcasted(f,args::Union{ParamArray,PBroadcast}...)
  arrays = map(get_array,args)
  g = (x...) -> map(f,(x...))
  fargs = map((x...)->Base.broadcasted(g,x...),arrays...)
  PBroadcast(fargs)
end

function Base.broadcasted(f,a::Number,b::Union{ParamArray,PBroadcast})
  barray = get_array(b)
  fab = map(x->Base.broadcasted(f,a,x),barray)
  PBroadcast(fab)
end

function Base.broadcasted(f,a::Union{ParamArray,PBroadcast},b::Number)
  aarray = get_array(a)
  fab = map(x->Base.broadcasted(f,x,b),aarray)
  PBroadcast(fab)
end

function Base.broadcasted(
  f,
  a::Union{ParamArray,PBroadcast},
  b::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}})
  Base.broadcasted(f,a,Base.materialize(b))
end

function Base.broadcasted(
  f,
  a::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}},
  b::Union{ParamArray,PBroadcast})
  Base.broadcasted(f,Base.materialize(a),b)
end

function Base.materialize(b::PBroadcast)
  barray = get_array(b)
  dest = map(Base.materialize,barray)
  T = eltype(dest)
  L = length(dest)
  a = Vector{T}(undef,L)
  Base.materialize!(a,dest)
  ParamArray(a)
end

function Base.materialize!(a::ParamArray,b::PBroadcast)
  map(Base.materialize!,get_array(a),get_array(b))
  a
end

function Arrays.return_value(
  f::BroadcastingFieldOpMap,
  a::ParamArray,
  b::AbstractArray)

  vi = return_value(f,testitem(a),b)
  array = Vector{typeof(vi)}(undef,length(a))
  for i = eachindex(a)
    array[i] = return_value(f,a[i],b)
  end
  ParamArray(array)
end

function Arrays.return_value(
  f::BroadcastingFieldOpMap,
  a::AbstractArray,
  b::ParamArray)

  vi = return_value(f,a,testitem(b))
  array = Vector{typeof(vi)}(undef,length(b))
  for i = eachindex(b)
    array[i] = return_value(f,a,b[i])
  end
  ParamArray(array)
end

function Arrays.return_value(
  f::BroadcastingFieldOpMap,
  a::ParamArray,
  b::ParamArray)

  vi = return_value(f,testitem(a),testitem(b))
  array = Vector{typeof(vi)}(undef,length(a))
  for i = eachindex(a)
    array[i] = return_value(f,a[i],b[i])
  end
  ParamArray(array)
end

function Arrays.return_cache(
  f::BroadcastingFieldOpMap,
  a::ParamArray,
  b::AbstractArray)

  ci = return_cache(f,testitem(a),b)
  bi = evaluate!(ci,f,testitem(a),b)
  cache = Vector{typeof(ci)}(undef,length(a))
  array = Vector{typeof(bi)}(undef,length(a))
  for i = eachindex(a)
    cache[i] = return_cache(f,a[i],b)
  end
  cache,ParamArray(array)
end

function Arrays.return_cache(
  f::BroadcastingFieldOpMap,
  a::AbstractArray,
  b::ParamArray)

  ci = return_cache(f,a,testitem(b))
  bi = evaluate!(ci,f,a,testitem(b))
  cache = Vector{typeof(ci)}(undef,length(b))
  array = Vector{typeof(bi)}(undef,length(b))
  for i = eachindex(b)
    cache[i] = return_cache(f,a,b[i])
  end
  cache,ParamArray(array)
end

function Arrays.return_cache(
  f::BroadcastingFieldOpMap,
  a::ParamArray,
  b::ParamArray)

  ci = return_cache(f,testitem(a),testitem(b))
  bi = evaluate!(ci,f,testitem(a),testitem(b))
  cache = Vector{typeof(ci)}(undef,length(a))
  array = Vector{typeof(bi)}(undef,length(a))
  for i = eachindex(a)
    cache[i] = return_cache(f,a[i],b[i])
  end
  cache,ParamArray(array)
end

function Arrays.evaluate!(
  cache,
  f::BroadcastingFieldOpMap,
  a::ParamArray{T,N},
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
  b::ParamArray{S,N}) where {T,S,N}

  cx,array = cache
  @inbounds for i = eachindex(array)
    array[i] = evaluate!(cx[i],f,a,b[i])
  end
  array
end

function Arrays.evaluate!(
  cache,
  f::BroadcastingFieldOpMap,
  a::ParamArray{T,N},
  b::ParamArray{S,N}) where {T,S,N}

  cx,array = cache
  @inbounds for i = eachindex(array)
    array[i] = evaluate!(cx[i],f,a[i],b[i])
  end
  array
end

for S in (:AbstractVector,:AbstractMatrix)
  T = S == :AbstractVector ? :ParamMatrix : :ParamVector
  U = S == :AbstractVector ? :ParamVector : :ParamMatrix
  @eval begin
    function Arrays.evaluate!(
      cache,
      f::BroadcastingFieldOpMap,
      a::$S,
      b::$T)

      cx,array = cache
      @inbounds for i = eachindex(array)
        array[i] = evaluate!(cx[i],f,a,b[i])
      end
      array
    end

    function Arrays.evaluate!(
      cache,
      f::BroadcastingFieldOpMap,
      a::$T,
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
      a::$T,
      b::$U)

      cx,array = cache
      @inbounds for i = eachindex(array)
        array[i] = evaluate!(cx[i],f,a[i],b[i])
      end
      array
    end

    function Arrays.evaluate!(
      cache,
      f::BroadcastingFieldOpMap,
      a::$T,
      b::AbstractArray{V,3} where V)

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
      b::ParamArray{V,3} where V)

      cx,array = cache
      @inbounds for i = eachindex(array)
        array[i] = evaluate!(cx[i],f,a,b[i])
      end
      array
    end

    function Arrays.evaluate!(
      cache,
      f::BroadcastingFieldOpMap,
      a::ParamArray{V,3} where V,
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
      a::AbstractArray{V,3} where V,
      b::$T)

      cx,array = cache
      @inbounds for i = eachindex(array)
        array[i] = evaluate!(cx[i],f,a,b[i])
      end
      array
    end
  end
end

for op in (:+,:-,:*)
  @eval begin
    function Arrays.return_value(f::Broadcasting{typeof($op)},a::ParamArray,b::ParamArray)
      return_value(Fields.BroadcastingFieldOpMap($op),a,b)
    end

    function Arrays.return_cache(f::Broadcasting{typeof($op)},a::ParamArray,b::ParamArray)
      return_cache(Fields.BroadcastingFieldOpMap($op),a,b)
    end

    function Arrays.evaluate!(cache,f::Broadcasting{typeof($op)},a::ParamArray,b::ParamArray)
      evaluate!(cache,Fields.BroadcastingFieldOpMap($op),a,b)
    end
  end
end

function Arrays.return_value(f::Broadcasting{typeof(*)},a::ParamArray,b::Number)
  vi = return_value(f,testitem(a),b)
  array = Vector{typeof(vi)}(undef,length(a))
  for i = eachindex(a)
    array[i] = return_value(f,a[i],b)
  end
  ParamArray(array)
end

function Arrays.return_cache(f::Broadcasting{typeof(*)},a::ParamArray,b::Number)
  ci = return_cache(f,testitem(a),b)
  bi = evaluate!(ci,f,testitem(a),b)
  cache = Vector{typeof(ci)}(undef,length(a))
  array = Vector{typeof(bi)}(undef,length(a))
  for i = eachindex(a)
    cache[i] = return_cache(f,a[i],b)
  end
  cache,ParamArray(array)
end

function Arrays.evaluate!(cache,f::Broadcasting{typeof(*)},a::ParamArray,b::Number)
  cx,array = cache
  @inbounds for i = eachindex(array)
    array[i] = evaluate!(cx[i],f,a[i],b)
  end
  array
end

function Arrays.return_value(k::Broadcasting{typeof(*)},a::Number,b::ParamArray)
  return_value(k,b,a)
end

function Arrays.return_cache(k::Broadcasting{typeof(*)},a::Number,b::ParamArray)
  return_cache(k,b,a)
end

function Arrays.evaluate!(cache,k::Broadcasting{typeof(*)},a::Number,b::ParamArray)
  evaluate!(cache,k,b,a)
end

# function Base.getindex(k::LinearCombinationField{<:ParamArray},i::Int)
#   LinearCombinationField(k.values[i],k.fields,k.column)
# end

# for T in (:(Point),:(AbstractVector{<:Point}))
#   @eval begin
#     function Arrays.return_value(a::LinearCombinationField{<:ParamArray},x::$T)
#       vi = return_value(testitem(a),x)
#       array = Vector{typeof(vi)}(undef,length(a.values))
#       for i = eachindex(a.values)
#         array[i] = return_value(a[i],x)
#       end
#       ParamArray(array)
#     end

#     function Arrays.return_cache(a::LinearCombinationField{<:ParamArray},x::$T)
#       ci = return_cache(testitem(a),x)
#       bi = evaluate!(ci,testitem(a),x)
#       cache = Vector{typeof(ci)}(undef,length(a.values))
#       array = Vector{typeof(bi)}(undef,length(a.values))
#       for i = eachindex(a.values)
#         cache[i] = return_cache(a[i],x)
#       end
#       cache,ParamArray(array)
#     end

#     function Arrays.evaluate!(cache,a::LinearCombinationField{<:ParamArray},x::$T)
#       cx,array = cache
#       @inbounds for i = eachindex(array)
#         array[i] = evaluate!(cx[i],a[i],x)
#       end
#       array
#     end
#   end
# end

# for S in (:AbstractVector,:AbstractMatrix,:AbstractArray)
#   for T in (:AbstractVector,:AbstractMatrix,:AbstractArray)
#     @eval begin
#       function Arrays.return_value(
#         k::LinearCombinationMap{<:Integer},
#         v::ParamArray{<:$S},
#         fx::$T)

#         vi = return_value(k,testitem(v),fx)
#         array = Vector{typeof(vi)}(undef,length(v))
#         for i = eachindex(v)
#           array[i] = return_value(k,v[i],fx)
#         end
#         ParamArray(array)
#       end

#       function Arrays.return_cache(
#         k::LinearCombinationMap{<:Integer},
#         v::ParamArray{<:$S},
#         fx::$T)

#         ci = return_cache(k,testitem(v),fx)
#         bi = evaluate!(ci,k,testitem(v),fx)
#         cache = Vector{typeof(ci)}(undef,length(v))
#         array = Vector{typeof(bi)}(undef,length(v))
#         for i = eachindex(v)
#           cache[i] = return_cache(k,v[i],fx)
#         end
#         cache,ParamArray(array)
#       end

#       function Arrays.evaluate!(
#         cache,
#         k::LinearCombinationMap{<:Integer},
#         v::ParamArray{<:$S},
#         fx::$T)

#         cx,array = cache
#         @inbounds for i = eachindex(array)
#           array[i] = evaluate!(cx[i],k,v[i],fx)
#         end
#         array
#       end
#     end
#   end
# end

# function Fields.linear_combination(a::ParamArray,b::AbstractArray)
#   abi = linear_combination(testitem(a),b)
#   c = Vector{typeof(abi)}(undef,length(a))
#   for i in eachindex(a)
#     c[i] = linear_combination(a[i],b)
#   end
#   ParamArray(c)
# end

function Arrays.return_value(
  f::IntegrationMap,
  a::ParamArray,
  w)

  vi = return_value(f,testitem(a),w)
  array = Vector{typeof(vi)}(undef,length(a))
  for i = eachindex(a)
    array[i] = return_value(f,a[i],w)
  end
  ParamArray(array)
end

function Arrays.return_cache(
  f::IntegrationMap,
  a::ParamArray,
  w)

  ci = return_cache(f,testitem(a),w)
  bi = evaluate!(ci,f,testitem(a),w)
  cache = Vector{typeof(ci)}(undef,length(a))
  array = Vector{typeof(bi)}(undef,length(a))
  for i = eachindex(a)
    cache[i] = return_cache(f,a[i],w)
  end
  cache,ParamArray(array)
end

function Arrays.evaluate!(
  cache,
  f::IntegrationMap,
  a::ParamArray,
  w)

  cx,array = cache
  @inbounds for i = eachindex(array)
    array[i] = evaluate!(cx[i],f,a[i],w)
  end
  array
end

function Arrays.return_value(
  f::IntegrationMap,
  a::ParamArray,
  w,
  j::AbstractVector)

  vi = return_value(f,testitem(a),w,j)
  array = Vector{typeof(vi)}(undef,length(a))
  for i = eachindex(a)
    array[i] = return_value(f,a[i],w,j)
  end
  ParamArray(array)
end

function Arrays.return_cache(
  f::IntegrationMap,
  a::ParamArray,
  w,
  j::AbstractVector)

  ci = return_cache(f,testitem(a),w,j)
  bi = evaluate!(ci,f,testitem(a),w,j)
  cache = Vector{typeof(ci)}(undef,length(a))
  array = Vector{typeof(bi)}(undef,length(a))
  for i = eachindex(a)
    cache[i] = return_cache(f,a[i],w,j)
  end
  cache,ParamArray(array)
end

function Arrays.evaluate!(
  cache,
  f::IntegrationMap,
  a::ParamArray,
  w,
  j::AbstractVector)

  cx,array = cache
  @inbounds for i = eachindex(array)
    array[i] = evaluate!(cx[i],f,a[i],w,j)
  end
  array
end

function Arrays.return_cache(::Fields.ZeroBlockMap,a::AbstractArray,b::ParamArray)
  _a = allocate_param_array(a,length(b))
  CachedArray(similar(_a,eltype(a),size(b)))
end

function Arrays.return_cache(::Fields.ZeroBlockMap,a::ParamArray,b::ParamArray)
  CachedArray(similar(a,eltype(a),size(b)))
end

function Arrays.evaluate!(cache::ParamArray,f::Fields.ZeroBlockMap,a,b::AbstractArray)
  _get_array(c::CachedArray) = c.array
  @inbounds for i = eachindex(cache)
    evaluate!(cache[i],f,a,b)
  end
  map(_get_array,cache)
end

function Arrays.evaluate!(cache::ParamArray,f::Fields.ZeroBlockMap,a,b::ParamArray)
  _get_array(c::CachedArray) = c.array
  @inbounds for i = eachindex(cache)
    evaluate!(cache[i],f,a,b[i])
  end
  map(_get_array,cache)
end

# function Arrays.return_cache(
#   f::Broadcasting{typeof(MultiField._sum_if_first_positive)},
#   dofs::ParamArray,
#   o)

#   dofsi = testitem(dofs)
#   ci = return_cache(f,dofsi,o)
#   bi = evaluate!(ci,f,dofsi,o)
#   cache = Vector{typeof(ci)}(undef,length(dofs))
#   array = Vector{typeof(bi)}(undef,length(dofs))
#   for i = eachindex(dofs)
#     cache[i] = return_cache(f,dofs[i],o)
#   end
#   cache,ParamArray(array)
# end

# function Arrays.evaluate!(
#   cache,
#   f::Broadcasting{typeof(MultiField._sum_if_first_positive)},
#   dofs::ParamArray,
#   o)

#   cx,array = cache
#   @inbounds for i = eachindex(dofs)
#     array[i] = evaluate!(cx[i],f,dofs[i],o)
#   end
#   array
# end

# function Utils.recenter(a::ParamArray{T},a0::ParamArray{T};kwargs...) where T
#   n = length(a)
#   n0 = length(a0)
#   ndiff = Int(n/n0)
#   array = Vector{T}(undef,n)
#   @inbounds for i = 1:n0
#     array[(i-1)*ndiff+1:i*ndiff] = recenter(a[(i-1)*ndiff+1:i*ndiff],a0[i];kwargs...)
#   end
#   ParamArray(array)
# end

# function get_at_offsets(x::ParamVector,offsets::Vector{Int},row::Int)
#   map(y->y[offsets[row]+1:offsets[row+1]],x)
# end

# function get_at_offsets(x::ParamMatrix,offsets::Vector{Int},row::Int,col::Int)
#   map(y->y[offsets[row]+1:offsets[row+1],offsets[col]+1:offsets[col+1]],x)
# end
