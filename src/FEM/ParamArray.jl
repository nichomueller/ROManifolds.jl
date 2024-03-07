struct ParamArray{T,N,A<:AbstractVector{<:AbstractArray{T,N}},L} <: AbstractParamContainer{T,N}
  array::A
  function ParamArray(array::A,::Val{L}) where {T,N,A<:AbstractVector{<:AbstractArray{T,N}},L}
    new{T,N,A,L}(array)
  end
end

const ParamVector{T,A,L} = ParamArray{T,1,A,L}
const ParamMatrix{T,A,L} = ParamArray{T,2,A,L}
const ParamSparseMatrix = ParamArray{T,2,A,L} where {T,A<:AbstractVector{<:AbstractSparseMatrix},L}

const AffineParamArray{T,N,A} = ParamArray{T,N,A,1}
const AffineParamVector{T,A} = ParamVector{T,A,1}
const AffineParamMatrix{T,A} = ParamMatrix{T,A,1}

function ParamArray(array)
  ParamArray(array,Val(length(array)))
end

function ParamArray(array::AbstractArray{T}) where {T<:Number}
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
Base.axes(a::ParamArray,i...) = axes(testitem(a),i...)
Base.eltype(::ParamArray{T,N,A,L}) where {T,N,A,L} = T
Base.eltype(::Type{ParamArray{T,N,A,L}}) where {T,N,A,L} = T
Base.ndims(::ParamArray{T,N,A,L}) where {T,N,A,L} = N
Base.ndims(::Type{ParamArray{T,N,A,L}}) where {T,N,A,L} = N
Base.first(a::ParamArray) = testitem(a)
Base.eachindex(::ParamArray{T,N,A,L}) where {T,N,A,L} = Base.OneTo(L)
Base.lastindex(::ParamArray{T,N,A,L}) where {T,N,A,L} = L
Base.getindex(a::ParamArray,i...) = ParamArray(getindex(get_array(a),i...))
Base.setindex!(a::ParamArray,v,i...) = setindex!(get_array(a),v,i...)
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

function Base.copy!(a::ParamArray,b::ParamArray)
  @assert length(a) == length(b)
  copyto!(a,b)
end

function Base.copyto!(a::ParamArray,b::ParamArray)
  map(copy!,get_array(a),get_array(b))
  a
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

function Base.similar(::Type{ParamArray{T,N,A,L}},n::Integer...) where {T,N,A,L}
  array = Vector{eltype(A)}(undef,L)
  @inbounds for i = eachindex(array)
    array[i] = similar(eltype(A),n...)
  end
  ParamArray(array)
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
  tv = testvalue(eltype(A))
  array = Vector{typeof(tv)}(undef,L)
  @inbounds for k = eachindex(array)
    array[k] = testvalue(eltype(A))
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

for op in (:+,:-)
  @eval begin
    function ($op)(a::T,b::T) where T<:ParamArray
      c = similar(a)
      @inbounds for i = eachindex(a)
        c[i] = ($op)(a[i],b[i])
      end
      c
    end
    function ($op)(a::ParamArray{T},b::S) where {T,S<:AbstractArray{T}}
      c = similar(a)
      @inbounds for i = eachindex(a)
        c[i] = ($op)(a[i],b)
      end
      c
    end
    function ($op)(a::S,b::ParamArray{T}) where {T,S<:AbstractArray{T}}
      c = similar(b)
      @inbounds for i = eachindex(b)
        c[i] = ($op)(a,b[i])
      end
      c
    end
  end
end

(Base.:-)(a::ParamArray) = a .* -1

function Base.:*(a::ParamArray,b::Number)
  ParamArray(get_array(a)*b)
end

function Base.:*(a::Number,b::ParamArray)
  b*a
end

function Base.:/(a::ParamArray,b::Number)
  a*(1/b)
end

function Base.:*(a::ParamMatrix,b::ParamVector)
  ci = testitem(a)*testitem(b)
  c = Vector{typeof(ci)}(undef,length(a))
  @inbounds for i = eachindex(a)
    c[i] = a[i] * b[i]
  end
  ParamArray(c)
end

function Base.:\(a::ParamMatrix,b::ParamVector)
  ci = testitem(a)*testitem(b)
  c = Vector{typeof(ci)}(undef,length(a))
  @inbounds for i = eachindex(a)
    c[i] = a[i] \ b[i]
  end
  ParamArray(c)
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
  at = map(transpose,a)
  ParamArray(at)
end

function Base.maximum(f,a::ParamArray)
  maxa = map(a) do a
    maximum(f,a)
  end
  maximum(f,maxa)
end

function Base.minimum(f,a::ParamArray)
  mina = map(a) do a
    minimum(f,a)
  end
  minimum(f,mina)
end

function Base.fill!(a::ParamArray,z)
  for ai in a
    fill!(ai,z)
  end
  return a
end

function LinearAlgebra.fillstored!(a::ParamArray,z)
  for ai in a
    fillstored!(ai,z)
  end
  return a
end

function LinearAlgebra.mul!(
  c::ParamArray,
  a::ParamArray,
  b::ParamArray,
  α::Number,β::Number)

  for i in eachindex(a)
    mul!(c[i],a[i],b[i],α,β)
  end
  return c
end

function LinearAlgebra.ldiv!(a::ParamArray,m::LU,b::ParamArray)
  for i in eachindex(a)
    ldiv!(a[i],m,b[i])
  end
  return a
end

function LinearAlgebra.ldiv!(a::ParamArray,m::ParamContainer,b::ParamArray)
  @assert length(a) == length(m) == length(b)
  for i in eachindex(a)
    ldiv!(a[i],m[i],b[i])
  end
  return a
end

function LinearAlgebra.rmul!(a::ParamArray,b::Number)
  for ai in a
    rmul!(ai,b)
  end
  return a
end

function LinearAlgebra.lu(a::ParamArray)
  lua = map(a) do a
    lu(a)
  end
  ParamContainer(lua)
end

function LinearAlgebra.lu!(a::ParamArray,b::ParamArray)
  for i in eachindex(a)
    lu!(a[i],b[i])
  end
  return a
end

function SparseArrays.resize!(a::ParamArray,args...)
  for ai in a
    resize!(ai,args...)
  end
  return a
end

SparseArrays.nnz(a::ParamMatrix) = nnz(first(a))
SparseArrays.nzrange(a::ParamMatrix,col::Int) = nzrange(first(a),col)
SparseArrays.rowvals(a::ParamMatrix) = rowvals(first(a))
SparseArrays.nonzeros(a::ParamMatrix) = ParamArray(map(nonzeros,a))
SparseMatricesCSR.colvals(a::ParamMatrix) = colvals(first(a))
SparseMatricesCSR.getoffset(a::ParamMatrix) = getoffset(first(a))

function Arrays.CachedArray(a::ParamArray)
  cache = map(a) do a
    CachedArray(a)
  end
  ParamArray(cache)
end

function Arrays.setsize!(
  a::ParamArray{T,N,AbstractVector{CachedArray{T,N}}},
  s::NTuple{N,Int}) where {T,N}

  for ai in a
    setsize!(ai,s)
  end
  return a
end

function Arrays.SubVector(a::ParamArray,pini::Int,pend::Int)
  svector = map(a) do vector
    SubVector(vector,pini,pend)
  end
  ParamArray(svector)
end

struct ParamBroadcast{D} <: AbstractParamBroadcast
  array::D
end

Arrays.get_array(a::ParamBroadcast) = a.array

function Base.broadcasted(f,a::Union{ParamArray,ParamBroadcast}...)
  bc = map((x...)->Base.broadcasted(f,x...),map(get_array,a)...)
  ParamBroadcast(bc)
end

function Base.broadcasted(f,a::Union{ParamArray,ParamBroadcast},b::Number)
  bc = map(a->Base.broadcasted(f,a,b),get_array(a))
  ParamBroadcast(bc)
end

function Base.broadcasted(f,a::Number,b::Union{ParamArray,ParamBroadcast})
  bc = map(b->Base.broadcasted(f,a,b),get_array(b))
  ParamBroadcast(bc)
end

function Base.broadcasted(f,
  a::Union{ParamArray,ParamBroadcast},
  b::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}})
  Base.broadcasted(f,a,Base.materialize(b))
end

function Base.broadcasted(
  f,
  a::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}},
  b::Union{ParamArray,ParamBroadcast})
  Base.broadcasted(f,Base.materialize(a),b)
end

function Base.materialize(b::ParamBroadcast)
  a = map(Base.materialize,get_array(b))
  ParamArray(a)
end

function Base.materialize!(a::ParamArray,b::Broadcast.Broadcasted)
  map(x->Base.materialize!(x,b),get_array(a))
  a
end

function Base.materialize!(a::ParamArray,b::ParamBroadcast)
  map(Base.materialize!,get_array(a),get_array(b))
  a
end

function Base.map(f,a::ParamArray...)
  map(f,get_array.(a)...)
end

function _to_param_array(a::ParamArray,b::AbstractArray)
  array = Vector{typeof(b)}(undef,length(a))
  @inbounds for i = eachindex(a)
    array[i] = b
  end
  ParamArray(array)
end
function _to_param_array(a::ParamArray,b::ParamArray)
  b
end
function _to_param_array!(a::ParamArray,b::AbstractArray)
  @inbounds for i = eachindex(a)
    a[i] = b
  end
  a
end
function _to_param_array!(a::ParamArray,b::ParamArray)
  b
end

function Arrays.return_value(
  f::BroadcastingFieldOpMap,
  a::ParamArray,
  b::AbstractArray)

  vi = return_value(f,testitem(a),b)
  array = Vector{typeof(vi)}(undef,length(a))
  @inbounds for i = eachindex(a)
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
  @inbounds for i = eachindex(b)
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
  @inbounds for i = eachindex(a)
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
  @inbounds for i = eachindex(a)
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
  @inbounds for i = eachindex(b)
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
  @inbounds for i = eachindex(a)
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

function Arrays.return_value(f::BroadcastingFieldOpMap,a::ParamArray...)
  evaluate(f,a...)
end

function Arrays.return_cache(f::BroadcastingFieldOpMap,a::ParamArray...)
  ai = first(a)
  @notimplementedif any(x->length(x)!=length(ai),a)
  ci = return_cache(f,map(testitem,a)...)
  bi = evaluate!(ci,f,map(testitem,a)...)
  cache = Vector{typeof(ci)}(undef,length(ai))
  array = allocate_param_array(bi,length(ai))
  @inbounds for i = eachindex(ai)
    ai = map(x->x[i],a)
    cache[i] = return_cache(f,ai...)
  end
  cache,array
end

function Arrays.evaluate!(cache,f::BroadcastingFieldOpMap,a::ParamArray...)
  ai = first(a)
  @notimplementedif any(x->length(x)!=length(ai),a)
  cx,array = cache
  @inbounds for i = eachindex(array)
    ai = map(x->x[i],a)
    array[i] = evaluate!(cx[i],f,ai...)
  end
  array
end

################################################################################
# cannot write Union{ParamArray,AbstractArray}... because of undesidered overloading;
# instead, writing a few mixed cases
function _return_value(f::BroadcastingFieldOpMap,args::Union{ParamArray,AbstractArray}...)
  evaluate(f,args...)
end

function _return_cache(f::BroadcastingFieldOpMap,args::Union{ParamArray,AbstractArray}...)
  inds = findall(ai->isa(ai,ParamArray),args)
  @notimplementedif length(inds) == 0
  ai = args[first(inds)]
  d = map(x->_to_param_array(ai,x),args)
  cx = return_cache(f,d...)
  cx,d
end

function _evaluate!(cache,f::BroadcastingFieldOpMap,args::Union{ParamArray,AbstractArray}...)
  inds = findall(ai->isa(ai,ParamArray),args)
  @notimplementedif length(inds) == 0
  cx,array = cache
  d = map(_to_param_array!,array,args)
  evaluate!(cx,f,d...)
end

function Arrays.return_value(f::BroadcastingFieldOpMap,a::ParamArray,b::AbstractArray...)
  _return_value(f,a,b...)
end

function Arrays.return_value(f::BroadcastingFieldOpMap,a::AbstractArray,b::ParamArray,c::AbstractArray...)
  _return_value(f,a,b,c...)
end

function Arrays.return_value(f::BroadcastingFieldOpMap,a::AbstractArray,b::AbstractArray,c::ParamArray,d::AbstractArray...)
  _return_value(f,a,b,c,d...)
end

function Arrays.return_cache(f::BroadcastingFieldOpMap,a::ParamArray,b::AbstractArray...)
  _return_cache(f,a,b...)
end

function Arrays.return_cache(f::BroadcastingFieldOpMap,a::AbstractArray,b::ParamArray,c::AbstractArray...)
  _return_cache(f,a,b,c...)
end

function Arrays.return_cache(f::BroadcastingFieldOpMap,a::AbstractArray,b::AbstractArray,c::ParamArray,d::AbstractArray...)
  _return_cache(f,a,b,c,d...)
end

function Arrays.evaluate!(cache,f::BroadcastingFieldOpMap,a::ParamArray,b::AbstractArray...)
  _evaluate!(cache,f,a,b...)
end

function Arrays.evaluate!(cache,f::BroadcastingFieldOpMap,a::AbstractArray,b::ParamArray,c::AbstractArray...)
  _evaluate!(cache,f,a,b,c...)
end

function Arrays.evaluate!(cache,f::BroadcastingFieldOpMap,a::AbstractArray,b::AbstractArray,c::ParamArray,d::AbstractArray...)
  _evaluate!(cache,f,a,b,c,d...)
end
################################################################################

function Arrays.return_value(
  ::typeof(*),
  a::ParamMatrix{T,A,L},
  b::ParamVector{S,B,L}
  ) where {T,A,S,B,L}
  array = Vector{eltype(B)}(undef,L)
  @inbounds for i = 1:L
    array[i] = return_value(*,a[i],b[i])
  end
  ParamArray(array)
end

function Arrays.return_value(f::Broadcasting,a::ParamArray)
  vi = return_value(f,testitem(a))
  array = Vector{typeof(vi)}(undef,length(a))
  @inbounds for i = eachindex(a)
    array[i] = return_value(f,a[i])
  end
  ParamArray(array)
end

function Arrays.return_cache(f::Broadcasting,a::ParamArray)
  ci = return_cache(f,testitem(a))
  bi = evaluate!(ci,f,testitem(a))
  cache = Vector{typeof(ci)}(undef,length(a))
  array = Vector{typeof(bi)}(undef,length(a))
  @inbounds for i = eachindex(a)
    cache[i] = return_cache(f,a[i])
  end
  cache,ParamArray(array)
end

function Arrays.evaluate!(cache,f::Broadcasting,a::ParamArray)
  cx,array = cache
  @inbounds for i = eachindex(array)
    array[i] = evaluate!(cx[i],f,a[i])
  end
  array
end

function Arrays.return_value(f::Broadcasting{typeof(∘)},a::ParamArray,b::Field)
  vi = return_value(f,testitem(a),b)
  array = Vector{typeof(vi)}(undef,length(a),b)
  @inbounds for i = eachindex(a)
    array[i] = return_value(f,a[i],b)
  end
  ParamArray(array)
end

function Arrays.return_cache(f::Broadcasting{typeof(∘)},a::ParamArray,b::Field)
  ci = return_cache(f,testitem(a),b)
  bi = evaluate!(ci,f,testitem(a),b)
  cache = Vector{typeof(ci)}(undef,length(a))
  array = Vector{typeof(bi)}(undef,length(a))
  @inbounds for i = eachindex(a)
    cache[i] = return_cache(f,a[i],b)
  end
  cache,ParamArray(array)
end

function Arrays.evaluate!(cache,f::Broadcasting{typeof(∘)},a::ParamArray,b::Field)
  cx,array = cache
  @inbounds for i = eachindex(array)
    array[i] = evaluate!(cx[i],f,a[i],b)
  end
  array
end

function Arrays.return_value(f::Broadcasting{<:Operation},a::ParamArray,b::Field)
  vi = return_value(f,testitem(a),b)
  array = Vector{typeof(vi)}(undef,length(a),b)
  @inbounds for i = eachindex(a)
    array[i] = return_value(f,a[i],b)
  end
  ParamArray(array)
end

function Arrays.return_cache(f::Broadcasting{<:Operation},a::ParamArray,b::Field)
  ci = return_cache(f,testitem(a),b)
  bi = evaluate!(ci,f,testitem(a),b)
  cache = Vector{typeof(ci)}(undef,length(a))
  array = Vector{typeof(bi)}(undef,length(a))
  @inbounds for i = eachindex(a)
    cache[i] = return_cache(f,a[i],b)
  end
  cache,ParamArray(array)
end

function Arrays.evaluate!(cache,f::Broadcasting{<:Operation},a::ParamArray,b::Field)
  cx,array = cache
  @inbounds for i = eachindex(array)
    array[i] = evaluate!(cx[i],f,a[i],b)
  end
  array
end

function Arrays.return_value(f::Broadcasting{<:Operation},a::Field,b::ParamArray)
  vi = return_value(f,a,testitem(b))
  array = Vector{typeof(vi)}(undef,a,length(b))
  @inbounds for i = eachindex(b)
    array[i] = return_value(f,a,b[i])
  end
  ParamArray(array)
end

function Arrays.return_cache(f::Broadcasting{<:Operation},a::Field,b::ParamArray)
  ci = return_cache(f,a,testitem(b))
  bi = evaluate!(ci,f,a,testitem(b))
  cache = Vector{typeof(ci)}(undef,length(b))
  array = Vector{typeof(bi)}(undef,length(b))
  @inbounds for i = eachindex(b)
    cache[i] = return_cache(f,a,b[i])
  end
  cache,ParamArray(array)
end

function Arrays.evaluate!(cache,f::Broadcasting{<:Operation},a::Field,b::ParamArray)
  cx,array = cache
  @inbounds for i = eachindex(array)
    array[i] = evaluate!(cx[i],f,a,b[i])
  end
  array
end

function Arrays.return_value(k::Broadcasting{<:Operation},a::ParamArray,b::ParamArray)
  evaluate(k,a,b)
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
  @inbounds for i = eachindex(a)
    array[i] = return_value(f,a[i],b)
  end
  ParamArray(array)
end

function Arrays.return_cache(f::Broadcasting{typeof(*)},a::ParamArray,b::Number)
  ci = return_cache(f,testitem(a),b)
  bi = evaluate!(ci,f,testitem(a),b)
  cache = Vector{typeof(ci)}(undef,length(a))
  array = Vector{typeof(bi)}(undef,length(a))
  @inbounds for i = eachindex(a)
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

function Fields.linear_combination(a::ParamArray,b::AbstractVector{<:Field})
  abi = linear_combination(testitem(a),b)
  c = Vector{typeof(abi)}(undef,length(a))
  @inbounds for i in eachindex(a)
    c[i] = linear_combination(a[i],b)
  end
  ParamContainer(c)
end

for T in (:AbstractVector,:AbstractMatrix,:AbstractArray)
  @eval begin
    function Arrays.return_value(
      k::LinearCombinationMap{<:Integer},
      v::ParamArray,
      fx::$T)

      vi = return_value(k,testitem(v),fx)
      array = Vector{typeof(vi)}(undef,length(v))
      @inbounds for i = eachindex(v)
        array[i] = return_value(k,v[i],fx)
      end
      ParamArray(array)
    end

    function Arrays.return_cache(
      k::LinearCombinationMap{<:Integer},
      v::ParamArray,
      fx::$T)

      ci = return_cache(k,testitem(v),fx)
      bi = evaluate!(ci,k,testitem(v),fx)
      cache = Vector{typeof(ci)}(undef,length(v))
      array = Vector{typeof(bi)}(undef,length(v))
      @inbounds for i = eachindex(v)
        cache[i] = return_cache(k,v[i],fx)
      end
      cache,ParamArray(array)
    end

    function Arrays.evaluate!(
      cache,
      k::LinearCombinationMap{<:Integer},
      v::ParamArray,
      fx::$T)

      cx,array = cache
      @inbounds for i = eachindex(array)
        array[i] = evaluate!(cx[i],k,v[i],fx)
      end
      array
    end
  end
end

function Arrays.return_value(
  f::IntegrationMap,
  a::ParamArray,
  w)

  vi = return_value(f,testitem(a),w)
  array = Vector{typeof(vi)}(undef,length(a))
  @inbounds for i = eachindex(a)
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
  @inbounds for i = eachindex(a)
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
  @inbounds for i = eachindex(a)
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
  @inbounds for i = eachindex(a)
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
  ParamArray(map(_get_array,cache))
end

function Arrays.evaluate!(cache::ParamArray,f::Fields.ZeroBlockMap,a,b::ParamArray)
  _get_array(c::CachedArray) = c.array
  @inbounds for i = eachindex(cache)
    evaluate!(cache[i],f,a,b[i])
  end
  ParamArray(map(_get_array,cache))
end

function Fields.unwrap_cached_array(a::ParamArray)
  cache = return_cache(unwrap_cached_array,a)
  evaluate!(cache,unwrap_cached_array,a)
end

function Arrays.return_cache(::typeof(Fields.unwrap_cached_array),a::ParamArray)
  ai = testitem(a)
  ci = return_cache(Fields.unwrap_cached_array,ai)
  ri = evaluate!(ci,Fields.unwrap_cached_array,ai)
  cache = Vector{typeof(ci)}(undef,length(a))
  array = Vector{typeof(ri)}(undef,length(a))
  @inbounds for i in eachindex(a)
    cache[i] = return_cache(Fields.unwrap_cached_array,a[i])
  end
  cache,ParamArray(array)
end

function Arrays.evaluate!(cache,::typeof(Fields.unwrap_cached_array),a::ParamArray)
  cx,array = cache
  @inbounds for i = eachindex(array)
    array[i] = evaluate!(cx[i],Fields.unwrap_cached_array,a[i])
  end
  array
end

function Fields._setsize_as!(d,a::ParamArray)
  @check size(d) == size(a)
  for i in eachindex(a)
    Fields._setsize_as!(d.array[i],a.array[i])
  end
  d
end

function Fields._setsize_mul!(c,a::ParamArray,b::ParamArray)
  @inbounds for i = eachindex(a)
    Fields._setsize_mul!(c[i],a[i],b[i])
  end
end

function  Fields._setsize_mul!(c,args::Union{ParamArray,AbstractArray}...)
  inds = findall(ai->isa(ai,ParamArray),args)
  @notimplementedif length(inds) == 0
  ai = args[first(inds)]
  b = map(x->_to_param_array(ai,x),args)
  Fields._setsize_mul!(c,b...)
end

function Arrays.return_value(k::MulAddMap,a::ParamArray,b::ParamArray,c::ParamArray)
  x = return_value(*,a,b)
  return_value(+,x,c)
end

function Arrays.return_cache(k::MulAddMap,a::ParamArray,b::ParamArray,c::ParamArray)
  c1 = CachedArray(a*b+c)
  c2 = return_cache(Fields.unwrap_cached_array,c1)
  (c1,c2)
end

function Arrays.evaluate!(cache,k::MulAddMap,a::ParamArray,b::ParamArray,c::ParamArray)
  c1,c2 = cache
  Fields._setsize_as!(c1,c)
  Fields._setsize_mul!(c1,a,b)
  d = evaluate!(c2,Fields.unwrap_cached_array,c1)
  copyto!(d,c)
  mul!(d,a,b,k.α,k.β)
  d
end
