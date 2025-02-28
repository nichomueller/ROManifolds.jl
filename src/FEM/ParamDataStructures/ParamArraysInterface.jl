"""
    abstract type AbstractParamArray{T,N,A<:AbstractArray{T,N}} <: AbstractParamContainer{A,N} end

Type representing parametric abstract arrays of type A.
Subtypes:
- [`ParamArray`](@ref)
- [`ParamSparseMatrix`](@ref)
"""
abstract type AbstractParamArray{T,N,A<:AbstractArray{T,N}} <: AbstractParamContainer{A,N} end

"""
    const AbstractParamVector{T} = AbstractParamArray{T,1,<:AbstractVector{T}}
"""
const AbstractParamVector{T} = AbstractParamArray{T,1,<:AbstractVector{T}}

"""
    const AbstractParamMatrix{T} = AbstractParamArray{T,2,<:AbstractMatrix{T}}
"""
const AbstractParamMatrix{T} = AbstractParamArray{T,2,<:AbstractMatrix{T}}

"""
    const AbstractParamArray3D{T} = AbstractParamArray{T,3,<:AbstractArray{T,3}}
"""
const AbstractParamArray3D{T} = AbstractParamArray{T,3,<:AbstractArray{T,3}}

"""
    abstract type ParamArray{T,N} <: AbstractParamArray{T,N,Array{T,N}} end

Type representing parametric arrays of type A.
Subtypes:
- [`TrivialParamArray`](@ref)
- [`ConsecutiveParamArray`](@ref)
- [`GenericParamVector`](@ref)
- [`GenericParamMatrix`](@ref)
- [`ArrayOfArrays`](@ref)
- [`BlockParamArray`](@ref)

Also acts as a constructor according to the following rules:
- ParamArray(A::AbstractArray{<:Number}) -> ParamNumber
- ParamArray(A::AbstractArray{<:Number},plength::Int) -> TrivialParamArray
- ParamArray(A::AbstractVector{<:AbstractArray}) -> ParamArray
- ParamArray(A::AbstractVector{<:AbstractSparseMatrix}) -> ParamSparseMatrix
- ParamArray(A::AbstractArray{<:ParamArray}) -> BlockParamArray
"""
abstract type ParamArray{T,N} <: AbstractParamArray{T,N,Array{T,N}} end

"""
    const ParamVector{T} = ParamArray{T,1}
"""
const ParamVector{T} = ParamArray{T,1}

"""
    const ParamMatrix{T} = ParamArray{T,2}
"""
const ParamMatrix{T} = ParamArray{T,2}

"""
    abstract type ParamSparseMatrix{Tv,Ti,A<:AbstractSparseMatrix{Tv,Ti}
      } <: AbstractParamArray{Tv,2,A} end

Type representing parametric abstract sparse matrices of type A.
Subtypes:
- [`ParamSparseMatrixCSC`](@ref)
- [`ParamSparseMatrixCSR`](@ref)
"""
abstract type ParamSparseMatrix{Tv,Ti,A<:AbstractSparseMatrix{Tv,Ti}} <: AbstractParamArray{Tv,2,A} end

"""
    abstract type MemoryLayoutStyle end

Subtypes:
- [`ConsecutiveMemory`](@ref)
- [`NonConsecutiveMemory`](@ref)
"""
abstract type MemoryLayoutStyle end

"""
    struct ConsecutiveMemory <: MemoryLayoutStyle end

Parametric objects with this trait store their values consecutively
"""
struct ConsecutiveMemory <: MemoryLayoutStyle end

"""
    struct NonConsecutiveMemory <: MemoryLayoutStyle end

Parametric objects with this trait do not store their values consecutively
"""
struct NonConsecutiveMemory <: MemoryLayoutStyle end

MemoryLayoutStyle(A::T) where T = MemoryLayoutStyle(T)
MemoryLayoutStyle(::Type{<:AbstractParamArray}) = ConsecutiveMemory()

ParamArray(args...;kwargs...) = @abstractmethod

"""
    param_array(a::AbstractArray,plength::Int;style=NonConsecutiveMemory()) -> ParamArray
    param_array(a::BlockArray,plength::Int;style=NonConsecutiveMemory()) -> BlockParamArray

Returns a [`AbstractParamArray`](@ref) of parametric length `plength` with entries
equal to those of `a`
"""
param_array(args...;kwargs...) = @abstractmethod

parameterize(a::AbstractArray,plength::Integer;kwargs...) = param_array(a,plength;kwargs...)

"""
    consecutive_parameterize(args...) -> ParamArray

Biulds of a [`AbstractParamArray`](@ref) with entries stored consecutively
in memory
"""
consecutive_parameterize(args...) = parameterize(args...;style=ConsecutiveMemory())

for f in (:param_array,:ParamArray)
  @eval begin
    $f(A::AbstractArray{<:Number};kwargs...) = ParamNumber(A)
    $f(A::AbstractParamArray;kwargs...) = A
  end
end

function param_array(a::Union{Number,AbstractArray{<:Number,0}},l::Integer;kwargs...)
  ParamNumber(fill(a,l))
end

param_getindex(A::AbstractParamArray{T,N},i::Integer) where {T,N} = getindex(A,tfill(i,Val{N}())...)
param_setindex!(A::AbstractParamArray{T,N},v,i::Integer) where {T,N} = setindex!(A,v,tfill(i,Val{N}())...)
to_param_quantity(a::AbstractArray,plength::Integer) = ParamArray(a,plength)

"""
    innerlength(A::AbstractParamArray) -> Int

Returns the length of `A` for a single parameter. Thus, the total entries of `A`
is equals to `param_length(A)*innerlength(A)`
"""
innerlength(A::AbstractParamArray) = prod(innersize(A))

"""
    inneraxes(A::AbstractParamArray) -> Tuple{Vararg{Base.OneTo}}

Returns the axes of `A` for a single parameter
"""
inneraxes(A::AbstractParamArray) = Base.OneTo.(innersize(A))

"""
    get_param_entry(A::AbstractParamArray{T},i...) where T -> Vector{eltype(T)}

Returns a vector of the entries of `A` at index `i`, for every parameter. The
length of the output is equals to `param_length(A)`
"""
get_param_entry(A::AbstractParamArray,i...) = @abstractmethod

# small hack, zero(::Type{<:AbstractArray}) is not implemented in Base
function Base.zero(::Type{<:AbstractArray{T,N}}) where {T<:Number,N}
  zeros(T,tfill(1,Val{N}()))
end
# small hack, one(::Type{<:AbstractArray}) is not implemented in Base
function Base.one(::Type{<:AbstractArray{T,N}}) where {T<:Number,N}
  ones(T,tfill(1,Val{N}()))
end

# small hack, we shouldn't be able to fill an abstract array with a non-scalar
for f in (:(Base.fill!),:(LinearAlgebra.fillstored!))
  @eval begin
    function $f(A::AbstractParamArray{T,N},z::AbstractArray{<:Number,N}) where {T,N}
      @check all(z.==first(z))
      $f(A,first(z))
      return A
    end
  end
end

function (*)(A::AbstractParamMatrix,x::AbstractParamVector)
  TS = LinearAlgebra.promote_op(LinearAlgebra.matprod,eltype(A),eltype(x))
  mul!(similar(x,TS,inneraxes(A)[1]),A,x)
end

function (*)(A::AbstractParamMatrix,B::AbstractParamMatrix)
  TS = LinearAlgebra.promote_op(LinearAlgebra.matprod,eltype(A),eltype(B))
  mul!(similar(B,TS,(innersize(A)[1],innersize(B)[2])),A,B)
end

function LinearAlgebra.mul!(
  C::AbstractParamArray,
  A::AbstractVecOrMat,
  B::AbstractParamArray,
  α::Number,β::Number)

  @check param_length(C) == param_length(B)
  @inbounds for i in param_eachindex(C)
    ci = param_getindex(C,i)
    bi = param_getindex(B,i)
    mul!(ci,A,bi,α,β)
  end
  return C
end

function LinearAlgebra.mul!(
  C::AbstractParamArray,
  A::AbstractParamArray,
  B::AbstractParamArray,
  α::Number,β::Number)

  @check param_length(C) == param_length(A) == param_length(B)
  @inbounds for i in param_eachindex(C)
    ci = param_getindex(C,i)
    ai = param_getindex(A,i)
    bi = param_getindex(B,i)
    mul!(ci,ai,bi,α,β)
  end
  return C
end

function (\)(A::AbstractParamMatrix,B::AbstractParamVector)
  TS = LinearAlgebra.promote_op(LinearAlgebra.matprod,eltype(A),eltype(B))
  C = similar(B,TS,axes(A,1))
  @inbounds for i in param_eachindex(C)
    param_getindex(C,i) .= param_getindex(A,i)\param_getindex(B,i)
  end
  return C
end

function LinearAlgebra.dot(A::AbstractParamArray,B::AbstractParamArray)
  @check size(A) == size(B)
  return map(dot,get_param_data(A),get_param_data(B))
end

function LinearAlgebra.norm(A::AbstractParamArray)
  return map(norm,get_param_data(A))
end

for factorization in (:LU,:Cholesky)
  @eval begin
    function LinearAlgebra.ldiv!(a::$factorization,B::AbstractParamArray)
      @inbounds for i in param_eachindex(B)
        bi = param_getindex(B,i)
        ldiv!(a,bi)
      end
      return B
    end
  end
end

function LinearAlgebra.ldiv!(A::AbstractParamArray,b::Factorization,C::AbstractParamArray)
  @check param_length(A) == param_length(C)
  @inbounds for i in param_eachindex(A)
    ai = param_getindex(A,i)
    ci = param_getindex(C,i)
    ldiv!(ai,b,ci)
  end
  return A
end

function LinearAlgebra.ldiv!(A::AbstractParamArray,B::ParamContainer,C::AbstractParamArray)
  @check param_length(A) == param_length(B) == param_length(C)
  @inbounds for i in param_eachindex(A)
    ai = param_getindex(A,i)
    bi = param_getindex(B,i)
    ci = param_getindex(C,i)
    ldiv!(ai,bi,ci)
  end
  return A
end

function LinearAlgebra.lu(A::AbstractParamArray;kwargs...)
  @notimplemented
end

function LinearAlgebra.lu!(A::AbstractParamArray,B::AbstractParamArray;kwargs...)
  @check param_length(A) == param_length(B)
  @inbounds for i in param_eachindex(A)
    ai = param_getindex(A,i)
    bi = param_getindex(B,i)
    lu!(ai,bi)
  end
  return A
end

# Gridap interface

Arrays.testitem(A::AbstractParamArray) = param_getindex(A,1)

function Arrays.testvalue(A::AbstractParamArray{T,N}) where {T,N}
  tv = testvalue(Array{T,N})
  plength = param_length(A)
  parameterize(tv,plength;style=MemoryLayoutStyle(A))
end

function Arrays.testvalue(::Type{A}) where {T,N,A<:AbstractParamArray{T,N}}
  tv = testvalue(Array{T,N})
  plength = one(Int)
  parameterize(tv,plength;style=MemoryLayoutStyle(A))
end

function Arrays.CachedArray(A::AbstractParamArray)
  @notimplemented
end

function Arrays.setsize!(A::AbstractParamArray{T,N},s::NTuple{N,Integer}) where {T,N}
  @notimplemented
end

"""
    param_return_value(f::Union{Function,Map},A...) -> Any

Generalization of the `Gridap` function `return_value` to the parametric case
"""
function param_return_value(f::Union{Function,Map},A...)
  pA = to_param_quantities(A...)
  c = return_value(f,map(testitem,pA)...)
  data = parameterize(c,param_length(first(pA)))
  return data
end

"""
    param_return_cache(f::Union{Function,Map},A...) -> Any

Generalization of the `Gridap` function `return_cache` to the parametric case
"""
function param_return_cache(f::Union{Function,Map},A...)
  pA = to_param_quantities(A...)
  pA1 = first(pA)
  item = map(testitem,pA)
  c = return_cache(f,item...)
  d = evaluate!(c,f,item...)
  cache = Vector{typeof(c)}(undef,param_length(pA1))
  data = parameterize(d,param_length(pA1))
  @inbounds for i in param_eachindex(pA1)
    cache[i] = return_cache(f,map(a -> param_getindex(a,i),pA)...)
  end
  return cache,data
end

"""
    param_evaluate!(C,f::Union{Function,Map},A...) -> Any

Generalization of the `Gridap` function `evaluate!` to the parametric case
"""
function param_evaluate!(C,f::Union{Function,Map},A...)
  cache,data = C
  pA = to_param_quantities(A...;plength=param_length(data))
  @inbounds for i in param_eachindex(data)
    vi = evaluate!(cache[i],f,map(a -> param_getindex(a,i),pA)...)
    param_setindex!(data,vi,i)
  end
  data
end

# optimizations

function param_return_value(
  f::Union{Function,Map},
  A::AbstractParamArray,
  b::Union{Field,AbstractArray{<:Number}}...)

  v = return_value(f,testitem(A),b...)
  pv = fill(v,param_length(A))
  return ParamArray(pv)
end

function param_return_value(
  f::Union{Function,Map},
  a::Union{Field,AbstractArray{<:Number}},
  B::AbstractParamArray,
  c::Union{Field,AbstractArray{<:Number}}...)

  v = return_value(f,a,testitem(B),c...)
  pv = fill(v,param_length(B))
  return ParamArray(pv)
end

function param_return_value(
  f::Union{Function,Map},
  a::Union{Field,AbstractArray{<:Number}},
  b::Union{Field,AbstractArray{<:Number}},
  C::AbstractParamArray,
  D::AbstractParamArray)

  @check param_length(C) == param_length(D)
  v = return_value(f,a,b,testitem(C),testitem(D))
  pv = fill(v,param_length(C))
  return ParamArray(pv)
end

function param_return_value(
  f::Union{Function,Map},
  a::Union{Field,AbstractArray{<:Number}},
  b::Union{Field,AbstractArray{<:Number}},
  C::AbstractParamArray,
  d::Union{Field,AbstractArray{<:Number}}...)

  v = return_value(f,a,b,testitem(C),d...)
  pv = fill(v,param_length(C))
  return ParamArray(pv)
end

function param_return_cache(
  f::Union{Function,Map},
  A::AbstractParamArray,
  b::Union{Field,AbstractArray{<:Number}}...)

  a = testitem(A)
  c = return_cache(f,a,b...)
  cx = evaluate!(c,f,a,b...)
  cache = Vector{typeof(c)}(undef,param_length(A))
  data = Vector{typeof(cx)}(undef,param_length(A))
  @inbounds for i = param_eachindex(A)
    cache[i] = return_cache(f,param_getindex(A,i),b...)
  end
  pdata = ParamArray(data)
  return cache,pdata
end

function param_return_cache(
  f::Union{Function,Map},
  a::Union{Field,AbstractArray{<:Number}},
  B::AbstractParamArray,
  c::Union{Field,AbstractArray{<:Number}}...)

  b = testitem(B)
  c′ = return_cache(f,a,b,c...)
  cx = evaluate!(c′,f,a,b,c...)
  cache = Vector{typeof(c′)}(undef,param_length(B))
  data = Vector{typeof(cx)}(undef,param_length(B))
  @inbounds for i = param_eachindex(B)
    cache[i] = return_cache(f,a,param_getindex(B,i),c...)
  end
  pdata = ParamArray(data)
  return cache,pdata
end

function param_return_cache(
  f::Union{Function,Map},
  a::Union{Field,AbstractArray{<:Number}},
  b::Union{Field,AbstractArray{<:Number}},
  C::AbstractParamArray,
  D::AbstractParamArray)

  @check param_length(C) == param_length(D)
  c = testitem(C)
  d = testitem(D)
  c′ = return_cache(f,a,b,c,d)
  cx = evaluate!(c′,f,a,b,c,d)
  cache = Vector{typeof(c′)}(undef,param_length(C))
  data = Vector{typeof(cx)}(undef,param_length(C))
  @inbounds for i = param_eachindex(C)
    cache[i] = return_cache(f,a,b,param_getindex(C,i),param_getindex(D,i))
  end
  pdata = ParamArray(data)
  return cache,pdata
end

function param_return_cache(
  f::Union{Function,Map},
  a::Union{Field,AbstractArray{<:Number}},
  b::Union{Field,AbstractArray{<:Number}},
  C::AbstractParamArray,
  d::Union{Field,AbstractArray{<:Number}}...)

  c = testitem(C)
  c′ = return_cache(f,a,b,c,d...)
  cx = evaluate!(c′,f,a,b,c,d...)
  cache = Vector{typeof(c′)}(undef,param_length(C))
  data = Vector{typeof(cx)}(undef,param_length(C))
  @inbounds for i = param_eachindex(C)
    cache[i] = return_cache(f,a,b,param_getindex(C,i),d...)
  end
  pdata = ParamArray(data)
  return cache,pdata
end

function param_evaluate!(
  C,
  f::Union{Function,Map},
  A::AbstractParamArray,
  b::Union{Field,AbstractArray{<:Number}}...)

  cache,data = C
  @inbounds for i = param_eachindex(A)
    vi = evaluate!(cache[i],f,param_getindex(A,i),b...)
    param_setindex!(data,vi,i)
  end
  return data
end

function param_evaluate!(
  C,
  f::Union{Function,Map},
  a::Union{Field,AbstractArray{<:Number}},
  B::AbstractParamArray,
  c::Union{Field,AbstractArray{<:Number}}...)

  cache,data = C
  @inbounds for i = param_eachindex(B)
    vi = evaluate!(cache[i],f,a,param_getindex(B,i),c...)
    param_setindex!(data,vi,i)
  end
  return data
end

function param_evaluate!(
  C′,
  f::Union{Function,Map},
  a::Union{Field,AbstractArray{<:Number}},
  b::Union{Field,AbstractArray{<:Number}},
  C::AbstractParamArray,
  D::AbstractParamArray)

  @check param_length(C) == param_length(D)
  cache,data = C′
  @inbounds for i = param_eachindex(C)
    vi = evaluate!(cache[i],f,a,b,param_getindex(C,i),param_getindex(D,i))
    param_setindex!(data,vi,i)
  end
  return data
end

function param_evaluate!(
  C′,
  f::Union{Function,Map},
  a::Union{Field,AbstractArray{<:Number}},
  b::Union{Field,AbstractArray{<:Number}},
  C::AbstractParamArray,
  d::Union{Field,AbstractArray{<:Number}}...)

  cache,data = C′
  @inbounds for i = param_eachindex(C)
    vi = evaluate!(cache[i],f,a,b,param_getindex(C,i),d...)
    param_setindex!(data,vi,i)
  end
  return data
end

# when the map is parametric

const ParamMap = Union{AbstractParamFunction,Broadcasting{<:AbstractParamFunction},BroadcastingFieldOpMap{<:AbstractParamFunction}}

param_length(F::BroadcastingFieldOpMap{<:AbstractParamFunction}) = param_length(F.op)
param_length(F::Broadcasting{<:AbstractParamFunction}) = param_length(F.f)
param_getindex(F::BroadcastingFieldOpMap{<:AbstractParamFunction},i::Int) = BroadcastingFieldOpMap(param_getindex(F.op,i))
param_getindex(F::Broadcasting{<:AbstractParamFunction},i::Int) = Broadcasting(param_getindex(F.f,i))
Arrays.testitem(F::BroadcastingFieldOpMap{<:AbstractParamFunction}) = param_getindex(F,1)
Arrays.testitem(F::Broadcasting{<:AbstractParamFunction}) = param_getindex(F,1)

function param_return_value(F::ParamMap,A...)
  plength = param_length(F)
  fitem = testitem(F)
  pA = to_param_quantities(A...;plength)
  c = return_value(fitem,map(testitem,pA)...)
  data = parameterize(c,plength)
  return data
end

function param_return_cache(F::ParamMap,A...)
  plength = param_length(F)
  fitem = testitem(F)
  pA = to_param_quantities(A...;plength)
  item = map(testitem,pA)
  c = return_cache(fitem,item...)
  d = evaluate!(c,fitem,item...)
  cache = Vector{typeof(c)}(undef,plength)
  data = parameterize(d,plength)
  @inbounds for i in 1:plength
    fi = param_getindex(F,i)
    cache[i] = return_cache(fi,map(a -> param_getindex(a,i),pA)...)
  end
  return cache,data
end

function param_evaluate!(C,F::ParamMap,A...)
  cache,data = C
  plength = param_length(data)
  pA = to_param_quantities(A...;plength)
  @inbounds for i in 1:plength
    fi = param_getindex(F,i)
    vi = evaluate!(cache[i],fi,map(a -> param_getindex(a,i),pA)...)
    param_setindex!(data,vi,i)
  end
  data
end

# optimizations

function param_return_value(
  F::ParamMap,
  A::AbstractParamArray,
  b::Union{Field,AbstractArray{<:Number}}...)

  @check param_length(F) == param_length(A)
  v = return_value(testitem(F),testitem(A),b...)
  pv = fill(v,param_length(F))
  return ParamArray(pv)
end

function param_return_value(
  F::ParamMap,
  a::Union{Field,AbstractArray{<:Number}},
  B::AbstractParamArray,
  c::Union{Field,AbstractArray{<:Number}}...)

  @check param_length(F) == param_length(B)
  v = return_value(testitem(F),a,testitem(B),c...)
  pv = fill(v,param_length(B))
  return ParamArray(pv)
end

function param_return_value(
  F::ParamMap,
  a::Union{Field,AbstractArray{<:Number}},
  b::Union{Field,AbstractArray{<:Number}},
  C::AbstractParamArray,
  D::AbstractParamArray)

  @check param_length(F) == param_length(C) == param_length(D)
  v = return_value(testitem(F),a,b,testitem(C),testitem(D))
  pv = fill(v,param_length(C))
  return ParamArray(pv)
end

function param_return_value(
  F::ParamMap,
  a::Union{Field,AbstractArray{<:Number}},
  b::Union{Field,AbstractArray{<:Number}},
  C::AbstractParamArray,
  d::Union{Field,AbstractArray{<:Number}}...)

  @check param_length(F) == param_length(C)
  v = return_value(testitem(F),a,b,testitem(C),d...)
  pv = fill(v,param_length(C))
  return ParamArray(pv)
end

function param_return_cache(
  F::ParamMap,
  A::AbstractParamArray,
  b::Union{Field,AbstractArray{<:Number}}...)

  @check param_length(F) == param_length(A)
  f = testitem(F)
  a = testitem(A)
  c = return_cache(f,a,b...)
  cx = evaluate!(c,f,a,b...)
  cache = Vector{typeof(c)}(undef,param_length(A))
  data = Vector{typeof(cx)}(undef,param_length(A))
  @inbounds for i = param_eachindex(A)
    cache[i] = return_cache(param_getindex(F,i),param_getindex(A,i),b...)
  end
  pdata = ParamArray(data)
  return cache,pdata
end

function param_return_cache(
  F::ParamMap,
  a::Union{Field,AbstractArray{<:Number}},
  B::AbstractParamArray,
  c::Union{Field,AbstractArray{<:Number}}...)

  @check param_length(F) == param_length(B)
  f = testitem(F)
  b = testitem(B)
  c′ = return_cache(f,a,b,c...)
  cx = evaluate!(c′,f,a,b,c...)
  cache = Vector{typeof(c′)}(undef,param_length(B))
  data = Vector{typeof(cx)}(undef,param_length(B))
  @inbounds for i = param_eachindex(B)
    cache[i] = return_cache(param_getindex(F,i),a,param_getindex(B,i),c...)
  end
  pdata = ParamArray(data)
  return cache,pdata
end

function param_return_cache(
  F::ParamMap,
  a::Union{Field,AbstractArray{<:Number}},
  b::Union{Field,AbstractArray{<:Number}},
  C::AbstractParamArray,
  D::AbstractParamArray)

  @check param_length(F) == param_length(C) == param_length(D)
  f = testitem(F)
  c = testitem(C)
  d = testitem(D)
  c′ = return_cache(f,a,b,c,d)
  cx = evaluate!(c′,f,a,b,c,d)
  cache = Vector{typeof(c′)}(undef,param_length(C))
  data = Vector{typeof(cx)}(undef,param_length(C))
  @inbounds for i = param_eachindex(C)
    cache[i] = return_cache(param_getindex(F,i),a,b,param_getindex(C,i),param_getindex(D,i))
  end
  pdata = ParamArray(data)
  return cache,pdata
end

function param_return_cache(
  F::ParamMap,
  a::Union{Field,AbstractArray{<:Number}},
  b::Union{Field,AbstractArray{<:Number}},
  C::AbstractParamArray,
  d::Union{Field,AbstractArray{<:Number}}...)

  @check param_length(F) == param_length(C)
  f = testitem(F)
  c = testitem(C)
  c′ = return_cache(f,a,b,c,d...)
  cx = evaluate!(c′,f,a,b,c,d...)
  cache = Vector{typeof(c′)}(undef,param_length(C))
  data = Vector{typeof(cx)}(undef,param_length(C))
  @inbounds for i = param_eachindex(C)
    cache[i] = return_cache(param_getindex(F,i),a,b,param_getindex(C,i),d...)
  end
  pdata = ParamArray(data)
  return cache,pdata
end

function param_evaluate!(
  C,
  F::ParamMap,
  A::AbstractParamArray,
  b::Union{Field,AbstractArray{<:Number}}...)

  cache,data = C
  @inbounds for i = param_eachindex(A)
    vi = evaluate!(cache[i],param_getindex(F,i),param_getindex(A,i),b...)
    param_setindex!(data,vi,i)
  end
  return data
end

function param_evaluate!(
  C,
  F::ParamMap,
  a::Union{Field,AbstractArray{<:Number}},
  B::AbstractParamArray,
  c::Union{Field,AbstractArray{<:Number}}...)

  cache,data = C
  @inbounds for i = param_eachindex(B)
    vi = evaluate!(cache[i],param_getindex(F,i),a,param_getindex(B,i),c...)
    param_setindex!(data,vi,i)
  end
  return data
end

function param_evaluate!(
  C′,
  F::ParamMap,
  a::Union{Field,AbstractArray{<:Number}},
  b::Union{Field,AbstractArray{<:Number}},
  C::AbstractParamArray,
  D::AbstractParamArray)

  @check param_length(C) == param_length(D)
  cache,data = C′
  @inbounds for i = param_eachindex(C)
    vi = evaluate!(cache[i],param_getindex(F,i),a,b,param_getindex(C,i),param_getindex(D,i))
    param_setindex!(data,vi,i)
  end
  return data
end

function param_evaluate!(
  C′,
  F::ParamMap,
  a::Union{Field,AbstractArray{<:Number}},
  b::Union{Field,AbstractArray{<:Number}},
  C::AbstractParamArray,
  d::Union{Field,AbstractArray{<:Number}}...)

  cache,data = C′
  @inbounds for i = param_eachindex(C)
    vi = evaluate!(cache[i],param_getindex(F,i),a,b,param_getindex(C,i),d...)
    param_setindex!(data,vi,i)
  end
  return data
end

for T in (:AbstractParamVector,:AbstractParamMatrix,:AbstractParamArray3D)
  for S in (:AbstractParamVector,:AbstractParamMatrix,:AbstractParamArray3D)
    @eval begin
      function Arrays.return_value(f::BroadcastingFieldOpMap,A::$T,B::$S)
        param_return_value(f,A,B)
      end

      function Arrays.return_cache(f::BroadcastingFieldOpMap,A::$T,B::$S)
        param_return_cache(f,A,B)
      end

      function Arrays.evaluate!(C,f::BroadcastingFieldOpMap,A::$T,B::$S)
        param_evaluate!(C,f,A,B)
      end
    end
  end
  for S in (:(AbstractVector{<:Number}),:(AbstractMatrix{<:Number}),:(AbstractArray{<:Number,3}))
    @eval begin
      function Arrays.return_value(f::BroadcastingFieldOpMap,A::$T,B::$S)
        param_return_value(f,A,B)
      end

      function Arrays.return_value(f::BroadcastingFieldOpMap,A::$S,B::$T)
        param_return_value(f,A,B)
      end

      function Arrays.return_cache(f::BroadcastingFieldOpMap,A::$T,B::$S)
        param_return_cache(f,A,B)
      end

      function Arrays.return_cache(f::BroadcastingFieldOpMap,A::$S,B::$T)
        param_return_cache(f,A,B)
      end

      function Arrays.evaluate!(C,f::BroadcastingFieldOpMap,A::$T,B::$S)
        param_evaluate!(C,f,A,B)
      end

      function Arrays.evaluate!(C,f::BroadcastingFieldOpMap,A::$S,B::$T)
        param_evaluate!(C,f,A,B)
      end
    end
  end
end

# more general cases

for T in (:BroadcastingFieldOpMap,:(BroadcastingFieldOpMap{<:AbstractParamFunction}))
  @eval begin
    function Arrays.return_value(f::$T,A::AbstractParamArray,B::Union{AbstractArray{<:Number},AbstractParamArray}...)
      param_return_value(f,A,B...)
    end

    function Arrays.return_cache(f::$T,A::AbstractParamArray,B::Union{AbstractArray{<:Number},AbstractParamArray}...)
      param_return_cache(f,A,B...)
    end

    function Arrays.evaluate!(c,f::$T,A::AbstractParamArray,B::Union{AbstractArray{<:Number},AbstractParamArray}...)
      param_evaluate!(c,f,A,B...)
    end

    function Arrays.return_value(f::$T,A::AbstractArray{<:Number},B::AbstractParamArray,C::Union{AbstractArray{<:Number},AbstractParamArray}...)
      param_return_value(f,A,B,C...)
    end

    function Arrays.return_cache(f::$T,A::AbstractArray{<:Number},B::AbstractParamArray,C::Union{AbstractArray{<:Number},AbstractParamArray}...)
      param_return_cache(f,A,B,C...)
    end

    function Arrays.evaluate!(c,f::$T,A::AbstractArray{<:Number},B::AbstractParamArray,C::Union{AbstractArray{<:Number},AbstractParamArray}...)
      param_evaluate!(c,f,A,B,C...)
    end

    function Arrays.return_value(f::$T,A::AbstractArray{<:Number},B::AbstractArray{<:Number},C::AbstractParamArray,D::Union{AbstractArray{<:Number},AbstractParamArray}...)
      param_return_value(f,A,B,C,D...)
    end

    function Arrays.return_cache(f::$T,A::AbstractArray{<:Number},B::AbstractArray{<:Number},C::AbstractParamArray,D::Union{AbstractArray{<:Number},AbstractParamArray}...)
      param_return_cache(f,A,B,C,D...)
    end

    function Arrays.evaluate!(c,f::$T,A::AbstractArray{<:Number},B::AbstractArray{<:Number},C::AbstractParamArray,D::Union{AbstractArray{<:Number},AbstractParamArray}...)
      param_evaluate!(c,f,A,B,C,D...)
    end
  end
end

for op in (:+,:-,:*)
  @eval begin
    function Arrays.return_value(f::Broadcasting{typeof($op)},A::AbstractParamArray,B::AbstractParamArray)
      param_return_value(Fields.BroadcastingFieldOpMap($op),A,B)
    end

    function Arrays.return_cache(f::Broadcasting{typeof($op)},A::AbstractParamArray,B::AbstractParamArray)
      param_return_cache(Fields.BroadcastingFieldOpMap($op),A,B)
    end

    function Arrays.evaluate!(C,f::Broadcasting{typeof($op)},A::AbstractParamArray,B::AbstractParamArray)
      param_evaluate!(C,Fields.BroadcastingFieldOpMap($op),A,B)
    end
  end
  for T in (:AbstractParamArray,:(AbstractArray{<:Number}))
    S = T == :AbstractParamArray ? :(AbstractArray{<:Number}) : :AbstractParamArray
    @eval begin
      function Arrays.return_value(f::Broadcasting{typeof($op)},A::$T,B::$S)
        param_return_value(f,A,B)
      end

      function Arrays.return_cache(f::Broadcasting{typeof($op)},A::$T,B::$S)
        param_return_cache(f,A,B)
      end

      function Arrays.evaluate!(C,f::Broadcasting{typeof($op)},A::$T,B::$S)
        param_evaluate!(C,f,A,B)
      end
    end
  end

end

# when map is parametric

for T in (:AbstractParamFunction,:(Broadcasting{<:AbstractParamFunction}),:(BroadcastingFieldOpMap{<:AbstractParamFunction}))
  @eval begin
    function Arrays.return_value(F::$T,args...)
      param_return_value(F,args...)
    end

    function Arrays.return_cache(F::$T,args...)
      param_return_cache(F,args...)
    end

    function Arrays.evaluate!(C,F::$T,args...)
      param_evaluate!(C,F,args...)
    end

    function Arrays.return_value(F::$T,args::AbstractArray...)
      param_return_value(F,args...)
    end

    function Arrays.return_cache(F::$T,args::AbstractArray...)
      param_return_cache(F,args...)
    end

    function Arrays.evaluate!(C,F::$T,args::AbstractArray...)
      param_evaluate!(C,F,args...)
    end
  end
end

for F in (:AbstractParamFunction,:(Broadcasting{<:AbstractParamFunction}),:(BroadcastingFieldOpMap{<:AbstractParamFunction}))
  for T in (:Number,:AbstractParamVector,:AbstractParamMatrix,:AbstractParamArray3D)
    for S in (:Number,:AbstractParamVector,:AbstractParamMatrix,:AbstractParamArray3D)
      @eval begin
        function Arrays.return_value(f::$F,A::$T,B::$S)
          param_return_value(f,A,B)
        end

        function Arrays.return_cache(f::$F,A::$T,B::$S)
          param_return_cache(f,A,B)
        end

        function Arrays.evaluate!(C,f::$F,A::$T,B::$S)
          param_evaluate!(C,f,A,B)
        end
      end
    end
    for S in (:(AbstractVector{<:Number}),:(AbstractMatrix{<:Number}),:(AbstractArray{<:Number,3}))
      @eval begin
        function Arrays.return_value(f::$F,A::$T,B::$S)
          param_return_value(f,A,B)
        end

        function Arrays.return_value(f::$F,A::$S,B::$T)
          param_return_value(f,A,B)
        end

        function Arrays.return_cache(f::$F,A::$T,B::$S)
          param_return_cache(f,A,B)
        end

        function Arrays.return_cache(f::$F,A::$S,B::$T)
          param_return_cache(f,A,B)
        end

        function Arrays.evaluate!(C,f::$F,A::$T,B::$S)
          param_evaluate!(C,f,A,B)
        end

        function Arrays.evaluate!(C,f::$F,A::$S,B::$T)
          param_evaluate!(C,f,A,B)
        end
      end
    end
  end
end

function Arrays.return_value(::typeof(*),A::AbstractParamArray,B::AbstractParamArray)
  param_return_value(*,A,B)
end

function Arrays.return_value(f::Broadcasting,A::AbstractParamArray)
  param_return_value(f,A)
end

function Arrays.return_cache(f::Broadcasting,A::AbstractParamArray)
  param_return_cache(f,A)
end

function Arrays.evaluate!(C,f::Broadcasting,A::AbstractParamArray)
  param_evaluate!(C,f,A)
end

for F in (:(typeof(∘)),:Operation)
  for T in (:AbstractParamArray,:Field)
    S = T == :AbstractParamArray ? :Field : :AbstractParamArray
    @eval begin
      function Arrays.return_value(f::Broadcasting{<:$F},A::$T,B::$S)
        param_return_value(f,A,B)
      end

      function Arrays.return_cache(f::Broadcasting{<:$F},A::$T,B::$S)
        param_return_cache(f,A,B)
      end

      function Arrays.evaluate!(C,f::Broadcasting{<:$F},A::$T,B::$S)
        param_evaluate!(C,f,A,B)
      end
    end
  end
end

for T in (:AbstractParamArray,:Number)
  S = T == :AbstractParamArray ? :Number : :AbstractParamArray
  @eval begin
    function Arrays.return_value(f::Broadcasting{typeof(*)},A::$T,B::$S)
      param_return_value(f,A,B)
    end

    function Arrays.return_cache(f::Broadcasting{typeof(*)},A::$T,B::$S)
      param_return_cache(f,A,B)
    end

    function Arrays.evaluate!(C,f::Broadcasting{typeof(*)},A::$T,B::$S)
      param_evaluate!(C,f,A,B)
    end
  end
end

for Q in (:Integer,:Colon)
  F = Q==:Integer ? :(LinearCombinationMap{<:Integer}) : :(LinearCombinationMap{Colon})
  for T in (:AbstractParamVector,:AbstractParamMatrix,:AbstractParamArray3D)
    for S in (:AbstractParamVector,:AbstractParamMatrix,:AbstractParamArray3D)
      @eval begin
        function Arrays.return_value(f::$F,A::$S,b::$T)
          param_return_value(f,A,b)
        end

        function Arrays.return_cache(f::$F,A::$S,b::$T)
          param_return_cache(f,A,b)
        end

        function Arrays.evaluate!(C,f::$F,A::$S,b::$T)
          param_evaluate!(C,f,A,b)
        end
      end
    end
    for S in (:AbstractVector,:AbstractMatrix,:AbstractArray)
      @eval begin
        function Arrays.return_value(f::$F,A::$S,b::$T)
          param_return_value(f,A,b)
        end

        function Arrays.return_cache(f::$F,A::$S,b::$T)
          param_return_cache(f,A,b)
        end

        function Arrays.evaluate!(C,f::$F,A::$S,b::$T)
          param_evaluate!(C,f,A,b)
        end

        function Arrays.return_value(f::$F,a::$T,B::$S)
          param_return_value(f,a,B)
        end

        function Arrays.return_cache(f::$F,a::$T,B::$S)
          param_return_cache(f,a,B)
        end

        function Arrays.evaluate!(C,f::$F,a::$T,B::$S)
          param_evaluate!(C,f,a,B)
        end
      end
    end
  end
end

function Arrays.return_value(f::IntegrationMap,A::AbstractParamArray,w::AbstractVector{<:Real})
  param_return_value(f,A,w)
end

function Arrays.return_cache(f::IntegrationMap,A::AbstractParamArray,w::AbstractVector{<:Real})
  param_return_cache(f,A,w)
end

function Arrays.evaluate!(C,f::IntegrationMap,A::AbstractParamArray,w::AbstractVector{<:Real})
  param_evaluate!(C,f,A,w)
end

function Arrays.return_value(f::IntegrationMap,A::AbstractParamArray,w::AbstractVector{<:Real},jq::AbstractVector)
  param_return_value(f,A,w,jq)
end

function Arrays.return_cache(f::IntegrationMap,A::AbstractParamArray,w::AbstractVector{<:Real},jq::AbstractVector)
  param_return_cache(f,A,w,jq)
end

function Arrays.evaluate!(C,f::IntegrationMap,A::AbstractParamArray,w::AbstractVector{<:Real},jq::AbstractVector)
  param_evaluate!(C,f,A,w,jq)
end

function Arrays.return_cache(f::CellData.ConstrainRowsMap,A::AbstractParamArray,constr,mask)
  param_return_cache(f,A,constr,mask)
end

function Arrays.evaluate!(C,f::CellData.ConstrainRowsMap,A::AbstractParamArray,constr,mask)
  param_evaluate!(C,f,A,constr,mask)
end

function Arrays.return_cache(f::CellData.ConstrainColsMap,A::AbstractParamArray,constr_t,mask)
  return_cache(f,A,constr_t,mask)
end

function Arrays.evaluate!(C,f::CellData.ConstrainColsMap,A::AbstractParamArray,constr_t,mask)
  param_evaluate!(C,f,A,constr_t,mask)
end

for T in (:AbstractParamArray,:AbstractArray,:Nothing), S in (:AbstractParamArray,:AbstractArray)
  (T∈(:AbstractArray,:Nothing) && S==:AbstractArray) && continue
  @eval begin
    function Arrays.return_cache(f::Fields.ZeroBlockMap,A::$T,B::$S)
      param_return_cache(f,A,B)
    end
  end
end

function Arrays.evaluate!(
  C::Tuple{<:Any,<:AbstractParamArray},
  f::Fields.ZeroBlockMap,
  a,
  b::AbstractArray)

  cache,data = C
  @inbounds for i in param_eachindex(data)
    vi = evaluate!(cache[i],f,a,b)
    param_setindex!(data,vi,i)
  end
  data
end

function Arrays.evaluate!(
  C::Tuple{<:Any,<:AbstractParamArray},
  f::Fields.ZeroBlockMap,
  A::AbstractParamArray,
  b::AbstractArray)

  cache,data = C
  @inbounds for i in param_eachindex(data)
    ai = param_getindex(A,i)
    vi = evaluate!(cache[i],f,ai,b)
    param_setindex!(data,vi,i)
  end
  data
end

function Arrays.evaluate!(
  C::Tuple{<:Any,<:AbstractParamArray},
  f::Fields.ZeroBlockMap,
  A::AbstractParamArray,
  B::AbstractParamArray)

  cache,data = C
  @inbounds for i in param_eachindex(data)
    ai = param_getindex(A,i)
    bi = param_getindex(B,i)
    vi = evaluate!(cache[i],f,ai,bi)
    param_setindex!(data,vi,i)
  end
  data
end

function Arrays.evaluate!(
  C::Tuple{<:Any,<:AbstractParamArray},
  f::Fields.ZeroBlockMap,
  a,
  B::AbstractParamArray)

  cache,data = C
  @inbounds for i in param_eachindex(data)
    bi = param_getindex(B,i)
    vi = evaluate!(cache[i],f,a,bi)
    param_setindex!(data,vi,i)
  end
  data
end

function Arrays.return_cache(
  k::BroadcastingFieldOpMap,
  f::ArrayBlock{<:AbstractArray,N},
  g::ArrayBlock{<:AbstractArray,N}
  ) where N

  @notimplementedif size(f) != size(g)
  fi = testvalue(testitem(f))
  gi = testvalue(testitem(g))
  ci = return_cache(k,fi,gi)
  hi = evaluate!(ci,k,fi,gi)
  m = Fields.ZeroBlockMap()
  a = Array{typeof(hi),N}(undef,size(f.array))
  b = Array{typeof(ci),N}(undef,size(f.array))
  zf = Array{typeof(return_cache(m,fi,gi))}(undef,size(f.array))
  zg = Array{typeof(return_cache(m,gi,fi))}(undef,size(f.array))
  t = map(|,f.touched,g.touched)
  for i in eachindex(f.array)
    if f.touched[i] && g.touched[i]
      b[i] = return_cache(k,f.array[i],g.array[i])
    elseif f.touched[i]
      _fi = f.array[i]
      zg[i] = return_cache(m,gi,_fi)
      _gi = evaluate!(zg[i],m,gi,_fi)
      b[i] = return_cache(k,_fi,_gi)
    elseif g.touched[i]
      _gi = g.array[i]
      zf[i] = return_cache(m,fi,_gi)
      _fi = evaluate!(zf[i],m,fi,_gi)
      b[i] = return_cache(k,_fi,_gi)
    end
  end
  ArrayBlock(a,t), b, zf, zg
end

function Arrays.return_cache(
  k::BroadcastingFieldOpMap,
  f::ArrayBlock{<:AbstractArray,1},
  g::ArrayBlock{<:AbstractArray,2}
  )

  fi = testvalue(testitem(f))
  gi = testvalue(testitem(g))
  ci = return_cache(k,fi,gi)
  hi = evaluate!(ci,k,fi,gi)
  @check size(g.array,1) == 1 || size(g.array,2) == 0
  s = (size(f.array,1),size(g.array,2))
  a = Array{typeof(hi),2}(undef,s)
  b = Array{typeof(ci),2}(undef,s)
  t = fill(false,s)
  for j in 1:s[2]
    for i in 1:s[1]
      if f.touched[i] && g.touched[1,j]
        t[i,j] = true
        b[i,j] = return_cache(k,f.array[i],g.array[1,j])
      end
    end
  end
  ArrayBlock(a,t), b
end

function Arrays.return_cache(
  k::BroadcastingFieldOpMap,
  f::ArrayBlock{<:AbstractArray,2},
  g::ArrayBlock{<:AbstractArray,1}
  )

  fi = testvalue(testitem(f))
  gi = testvalue(testitem(g))
  ci = return_cache(k,fi,gi)
  hi = evaluate!(ci,k,fi,gi)
  @check size(f.array,1) == 1 || size(f.array,2) == 0
  s = (size(g.array,1),size(f.array,2))
  a = Array{typeof(hi),2}(undef,s)
  b = Array{typeof(ci),2}(undef,s)
  t = fill(false,s)
  for j in 1:s[2]
    for i in 1:s[1]
      if f.touched[1,j] && g.touched[i]
        t[i,j] = true
        b[i,j] = return_cache(k,f.array[1,j],g.array[i])
      end
    end
  end
  ArrayBlock(a,t), b
end

function Arrays.return_cache(
  k::BroadcastingFieldOpMap,
  a::(ArrayBlock{<:AbstractArray,N})...
  ) where N

  a1 = first(a)
  @notimplementedif any(ai->size(ai)!=size(a1),a)
  ais = map(ai->testvalue(testitem(ai)),a)
  ci = return_cache(k,ais...)
  bi = evaluate!(ci,k,ais...)
  c = Array{typeof(ci),N}(undef,size(a1))
  array = Array{typeof(bi),N}(undef,size(a1))
  for i in eachindex(a1.array)
    @notimplementedif any(ai->ai.touched[i]!=a1.touched[i],a)
    if a1.touched[i]
      _ais = map(ai->ai.array[i],a)
      c[i] = return_cache(k,_ais...)
    end
  end
  ArrayBlock(array,a1.touched), c
end

for A in (:ArrayBlock,:AbstractArray)
  for B in (:ArrayBlock,:AbstractArray)
    for C in (:ArrayBlock,:AbstractArray)
      if !(A == B == C)
        @eval begin
          function Arrays.evaluate!(cache,k::Fields.BroadcastingFieldOpMap,a::$A,b::$B,c::$C)
            function _replace_nz_blocks!(cache::ArrayBlock,vali::AbstractArray)
              for i in eachindex(cache.array)
                if cache.touched[i]
                  cache.array[i] = vali
                end
              end
              cache
            end

            function _replace_nz_blocks!(cache::ArrayBlock,val::ArrayBlock)
              for i in eachindex(cache.array)
                if cache.touched[i]
                  cache.array[i] = val.array[i]
                end
              end
              cache
            end

            eval_cache,replace_cache = cache
            cachea,cacheb,cachec = replace_cache

            _replace_nz_blocks!(cachea,a)
            _replace_nz_blocks!(cacheb,b)
            _replace_nz_blocks!(cachec,c)

            evaluate!(eval_cache,k,cachea,cacheb,cachec)
          end
        end
      end
      for D in (:ArrayBlock,:AbstractArray)
        if !(A == B == C == D)
          @eval begin
            function Arrays.evaluate!(cache,k::Fields.BroadcastingFieldOpMap,a::$A,b::$B,c::$C,d::$D)
              function _replace_nz_blocks!(cache::ArrayBlock,vali::AbstractArray)
                for i in eachindex(cache.array)
                  if cache.touched[i]
                    cache.array[i] = vali
                  end
                end
                cache
              end

              function _replace_nz_blocks!(cache::ArrayBlock,val::ArrayBlock)
                for i in eachindex(cache.array)
                  if cache.touched[i]
                    cache.array[i] = val.array[i]
                  end
                end
                cache
              end

              eval_cache,replace_cache = cache
              cachea,cacheb,cachec,cached = replace_cache

              _replace_nz_blocks!(cachea,a)
              _replace_nz_blocks!(cacheb,b)
              _replace_nz_blocks!(cachec,c)
              _replace_nz_blocks!(cached,d)

              evaluate!(eval_cache,k,cachea,cacheb,cachec,cached)
            end
          end
        end
      end
    end
  end
end

function Fields.unwrap_cached_array(A::AbstractParamArray)
  C = param_return_cache(Fields.unwrap_cached_array,A)
  param_evaluate!(C,Fields.unwrap_cached_array,A)
end

function Fields._setsize_as!(A::AbstractParamArray,B::AbstractParamArray)
  @check param_length(A) == param_length(B)
  @inbounds for i in param_eachindex(A)
    Fields._setsize_as!(param_getindex(A,i),param_getindex(B,i))
  end
  A
end

function Fields._setsize_mul!(C::AbstractParamArray,A::AbstractParamArray,B::AbstractParamArray)
  @check param_length(A) == param_length(B)
  @inbounds for i = eachindex(C)
    Fields._setsize_mul!(param_getindex(C,i),param_getindex(A,i),param_getindex(B,i))
  end
end

function Fields._setsize_mul!(C,A::Union{AbstractParamArray,AbstractArray}...)
  pA = to_param_quantities(A...)
  Fields._setsize_mul!(C,pA...)
end

function Arrays.return_value(f::MulAddMap,A::AbstractParamArray,B::AbstractParamArray,C::AbstractParamArray)
  x = return_value(*,A,B)
  return_value(+,x,C)
end

function Arrays.return_cache(f::MulAddMap,A::AbstractParamArray,B::AbstractParamArray,C::AbstractParamArray)
  c1 = CachedArray(A*B+C)
  c2 = return_cache(Fields.unwrap_cached_array,c1)
  (c1,c2)
end

function Arrays.evaluate!(cache,f::MulAddMap,A::AbstractParamArray,B::AbstractParamArray,C::AbstractParamArray)
  c1,c2 = cache
  Fields._setsize_as!(c1,C)
  Fields._setsize_mul!(c1,A,B)
  d = evaluate!(c2,Fields.unwrap_cached_array,c1)
  copyto!(d,C)
  mul!(d,A,B,f.α,f.β)
  d
end

function Arrays.return_cache(f::ConfigMap{typeof(ForwardDiff.gradient)},A::AbstractParamArray)
  return_cache(f,testitem(A))
end

function Arrays.return_cache(f::ConfigMap{typeof(ForwardDiff.jacobian)},A::AbstractParamArray)
  return_cache(f,testitem(A))
end

function Arrays.return_value(f::DualizeMap,A::AbstractParamArray)
  param_return_value(f,A)
end

function Arrays.return_cache(f::DualizeMap,A::AbstractParamArray)
  param_return_cache(f,A)
end

function Arrays.evaluate!(C,f::DualizeMap,A::AbstractParamArray)
  param_evaluate!(C,f,A)
end

for T in (:(ForwardDiff.GradientConfig),:(ForwardDiff.JacobianConfig))
  @eval begin
    function Arrays.return_value(
      f::AutoDiffMap,
      ydual::AbstractVector,
      x::AbstractParamArray,
      cfg::$T)

      @check length(ydual) == param_length(x)
      vi = return_value(f,testitem(ydual),testitem(x),cfg)
      A = parameterize(vi,length(ydual))
      return A
    end

    function Arrays.return_cache(
      f::AutoDiffMap,
      ydual::AbstractVector,
      x::AbstractParamArray,
      cfg::$T)

      @check length(ydual) == param_length(x)
      ci = return_cache(f,testitem(ydual),testitem(x),cfg)
      A = parameterize(ci,length(ydual))
      return A
    end

    function Arrays.evaluate!(
      cache::AbstractParamArray,
      f::AutoDiffMap,
      ydual::AbstractVector,
      x::AbstractParamArray,
      cfg::$T)

      @inbounds for i = param_eachindex(cache)
        ci = param_getindex(cache,i)
        evaluate!(ci,f,ydual[i],param_getindex(x,i),cfg)
      end
      cache
    end
  end
end
