"""
    abstract type AbstractParamArray{T,N,L,A<:AbstractArray{T,N}} <: AbstractParamContainer{A,N,L} end

Type representing parametric abstract arrays of type A. L encodes the parametric length.
Subtypes:
- [`ParamArray`](@ref).
- [`ParamSparseMatrix`](@ref).

"""
abstract type AbstractParamArray{T,N,L,A<:AbstractArray{T,N}} <: AbstractParamContainer{A,N,L} end

const AbstractParamVector{T,L} = AbstractParamArray{T,1,L,<:AbstractVector{T}}
const AbstractParamMatrix{T,L} = AbstractParamArray{T,2,L,<:AbstractMatrix{T}}
const AbstractParamArray3D{T,L} = AbstractParamArray{T,3,L,<:AbstractArray{T,3}}

"""
    abstract type ParamArray{T,N,L} <: AbstractParamArray{T,N,L,Array{T,N}} end

Type representing parametric arrays of type A. L encodes the parametric length.
Subtypes:
- [`ParamArray`](@ref).
- [`ConsecutiveParamArray`](@ref).
- [`TrivialParamArray`](@ref).
- [`BlockParamArray`](@ref).

"""
abstract type ParamArray{T,N,L} <: AbstractParamArray{T,N,L,Array{T,N}} end
const ParamVector{T,L} = ParamArray{T,1,L}
const ParamMatrix{T,L} = ParamArray{T,2,L}

"""
    abstract type ParamSparseMatrix{Tv,Ti,L,A<:AbstractSparseMatrix{Tv,Ti}
      } <: AbstractParamArray{Tv,2,L,A} end

Type representing parametric abstract sparse matrices of type A. L encodes the parametric length.
Subtypes:
- [`ParamSparseMatrixCSC`](@ref).

"""
abstract type ParamSparseMatrix{Tv,Ti,L,A<:AbstractSparseMatrix{Tv,Ti}} <: AbstractParamArray{Tv,2,L,A} end


"""
    ParamArray(A::AbstractArray{<:Number}) -> ParamNumber
    ParamArray(A::AbstractArray{<:Number},plength::Int) -> TrivialParamArray
    ParamArray(A::AbstractVector{<:AbstractArray}) -> ParamArray
    ParamArray(A::AbstractVector{<:AbstractSparseMatrix}) -> ParamSparseMatrix
    ParamArray(A::AbstractArray{<:ParamArray}) -> BlockParamArray

Generic constructor of a AbstractParamArray

"""
ParamArray(A) = @abstractmethod
param_array(A,args...) = @abstractmethod

# Numbers interface
ParamArray(A::AbstractArray{<:Number}) = ParamNumber(A)

function param_array(a::Union{Number,AbstractArray{<:Number,0}},l::Integer)
  ParamNumber(fill(a,l))
end

param_getindex(A::AbstractParamArray{T,N},i::Integer) where {T,N} = getindex(A,tfill(i,Val{N}())...)
param_setindex!(A::AbstractParamArray{T,N},v,i::Integer) where {T,N} = setindex!(A,v,tfill(i,Val{N}())...)
get_param_entry(A::AbstractParamArray,i...) = @abstractmethod
to_param_quantity(a::AbstractArray,plength::Integer) = ParamArray(a,plength)

innerlength(A::AbstractParamArray) = prod(innersize(A))
inneraxes(A::AbstractParamArray) = Base.OneTo.(innersize(A))

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

function Arrays.testvalue(::Type{<:AbstractParamArray{T,N,L}}) where {T,N,L}
  tv = testvalue(Array{T,N})
  param_array(tv,L)
end

function Arrays.CachedArray(A::AbstractParamArray)
  @notimplemented
end

function Arrays.setsize!(A::AbstractParamArray{T,N},s::NTuple{N,Integer}) where {T,N}
  @notimplemented
end

"""
    param_return_value(f::Union{Function,Map},A...) -> Any

Generalization of [`return_value`](@ref) to the parametric case

"""
function param_return_value(f::Union{Function,Map},A...)
  pA = to_param_quantities(A...)
  c = return_value(f,testitem.(pA)...)
  data = param_array(c,param_length(first(pA)))
  return data
end

"""
    param_return_cache(f::Union{Function,Map},A...) -> Any

Generalization of [`return_cache`](@ref) to the parametric case

"""
function param_return_cache(f::Union{Function,Map},A...)
  pA = to_param_quantities(A...)
  c = return_cache(f,testitem.(pA)...)
  d = evaluate!(c,f,testitem.(pA)...)
  cache = Vector{typeof(c)}(undef,param_length(first(pA)))
  data = param_array(d,param_length(first(pA)))
  @inbounds for i in param_eachindex(first(pA))
    cache[i] = return_cache(f,param_getindex.(pA,i)...)
  end
  return cache,data
end


"""
    param_evaluate!(C,f::Union{Function,Map},A...) -> Any

Generalization of [`evaluate!`](@ref) to the parametric case

"""
function param_evaluate!(C,f::Union{Function,Map},A...)
  cache,data = C
  pA = to_param_quantities(A...;plength=param_length(data))
  @inbounds for i in param_eachindex(data)
    vi = evaluate!(cache[i],f,param_getindex.(pA,i)...)
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
  return param_array(v,param_length(A))
end

function param_return_value(
  f::Union{Function,Map},
  a::Union{Field,AbstractArray{<:Number}},
  B::AbstractParamArray,
  c::Union{Field,AbstractArray{<:Number}}...)

  v = return_value(f,a,testitem(B),c...)
  return param_array(v,param_length(B))
end

function param_return_value(
  f::Union{Function,Map},
  a::Union{Field,AbstractArray{<:Number}},
  b::Union{Field,AbstractArray{<:Number}},
  C::AbstractParamArray,
  d::Union{Field,AbstractArray{<:Number}}...)

  v = return_value(f,a,b,testitem(C),d...)
  return param_array(v,param_length(C))
end

function param_return_cache(
  f::Union{Function,Map},
  A::AbstractParamArray,
  b::Union{Field,AbstractArray{<:Number}}...)

  c = return_cache(f,testitem(A),b...)
  cx = evaluate!(c,f,testitem(A),b...)
  cache = Vector{typeof(c)}(undef,param_length(A))
  data = param_array(cx,param_length(A))
  @inbounds for i = param_eachindex(A)
    cache[i] = return_cache(f,param_getindex(A,i),b...)
  end
  return cache,data
end

function param_return_cache(
  f::Union{Function,Map},
  a::Union{Field,AbstractArray{<:Number}},
  B::AbstractParamArray,
  c::Union{Field,AbstractArray{<:Number}}...)

  c′ = return_cache(f,a,testitem(B),c...)
  cx = evaluate!(c′,f,a,testitem(B),c...)
  cache = Vector{typeof(c′)}(undef,param_length(B))
  data = param_array(cx,param_length(B))
  @inbounds for i = param_eachindex(B)
    cache[i] = return_cache(f,a,param_getindex(B,i),c...)
  end
  return cache,data
end

function param_return_cache(
  f::Union{Function,Map},
  a::Union{Field,AbstractArray{<:Number}},
  b::Union{Field,AbstractArray{<:Number}},
  C::AbstractParamArray,
  d::Union{Field,AbstractArray{<:Number}}...)

  c′ = return_cache(f,a,b,testitem(C),d...)
  cx = evaluate!(c′,f,a,b,testitem(C),d...)
  cache = Vector{typeof(c′)}(undef,param_length(C))
  data = param_array(cx,param_length(C))
  @inbounds for i = param_eachindex(C)
    cache[i] = return_cache(f,a,b,param_getindex(C,i),d...)
  end
  return cache,data
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
  d::Union{Field,AbstractArray{<:Number}}...)

  cache,data = C′
  @inbounds for i = param_eachindex(C)
    vi = evaluate!(cache[i],f,a,b,param_getindex(C,i),d...)
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

function Arrays.return_value(f::BroadcastingFieldOpMap,A::AbstractParamArray,B::Union{AbstractArray{<:Number},AbstractParamArray}...)
  param_return_value(f,A,B...)
end

function Arrays.return_cache(f::BroadcastingFieldOpMap,A::AbstractParamArray,B::Union{AbstractArray{<:Number},AbstractParamArray}...)
  param_return_cache(f,A,B...)
end

function Arrays.evaluate!(c,f::BroadcastingFieldOpMap,A::AbstractParamArray,B::Union{AbstractArray{<:Number},AbstractParamArray}...)
  param_evaluate!(c,f,A,B...)
end

function Arrays.return_value(f::BroadcastingFieldOpMap,A::AbstractArray{<:Number},B::AbstractParamArray,C::Union{AbstractArray{<:Number},AbstractParamArray}...)
  param_return_value(f,A,B,C...)
end

function Arrays.return_cache(f::BroadcastingFieldOpMap,A::AbstractArray{<:Number},B::AbstractParamArray,C::Union{AbstractArray{<:Number},AbstractParamArray}...)
  param_return_cache(f,A,B,C...)
end

function Arrays.evaluate!(c,f::BroadcastingFieldOpMap,A::AbstractArray{<:Number},B::AbstractParamArray,C::Union{AbstractArray{<:Number},AbstractParamArray}...)
  param_evaluate!(c,f,A,B,C...)
end

function Arrays.return_value(f::BroadcastingFieldOpMap,A::AbstractArray{<:Number},B::AbstractArray{<:Number},C::AbstractParamArray,D::Union{AbstractArray{<:Number},AbstractParamArray}...)
  param_return_value(f,A,B,C,D...)
end

function Arrays.return_cache(f::BroadcastingFieldOpMap,A::AbstractArray{<:Number},B::AbstractArray{<:Number},C::AbstractParamArray,D::Union{AbstractArray{<:Number},AbstractParamArray}...)
  param_return_cache(f,A,B,C,D...)
end

function Arrays.evaluate!(c,f::BroadcastingFieldOpMap,A::AbstractArray{<:Number},B::AbstractArray{<:Number},C::AbstractParamArray,D::Union{AbstractArray{<:Number},AbstractParamArray}...)
  param_evaluate!(c,f,A,B,C,D...)
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

function Arrays.return_cache(
  f::BroadcastingFieldOpMap{<:AbstractParamFunction},
  x::Union{Number,AbstractArray{<:Number}}...)

  g = BroadcastingFieldOpMap(testitem(f.op))
  c = return_cache(g,x...)
  d = evaluate!(c,g,x...)
  cache = Vector{typeof(c)}(undef,param_length(f.op))
  data = param_array(d,param_length(f.op))
  @inbounds for i in param_eachindex(f.op)
    g = BroadcastingFieldOpMap(getindex(f.op,i))
    cache[i] = return_cache(g,x...)
  end
  return cache,data
end

function Arrays.evaluate!(
  C,
  f::BroadcastingFieldOpMap{<:AbstractParamFunction},
  x::Union{Number,AbstractArray{<:Number}}...)

  cache,data = C
  @inbounds for i in param_eachindex(data)
    g = BroadcastingFieldOpMap(getindex(f.op,i))
    vi = evaluate!(cache[i],g,x...)
    param_setindex!(data,vi,i)
  end
  data
end

function Arrays.return_cache(
  f::BroadcastingFieldOpMap{<:AbstractParamFunction},
  A::AbstractParamArray,
  b::Union{AbstractParamArray,AbstractArray{<:Number}}...)

  plength = param_length(f.op)
  g = BroadcastingFieldOpMap(testitem(f.op))
  B = to_param_quantities(b...;plength)
  c = return_cache(g,testitem(A),testitem.(B)...)
  d = evaluate!(c,g,testitem(A),testitem.(B)...)
  cache = Vector{typeof(c)}(undef,plength)
  data = param_array(d,plength)
  @inbounds for i in 1:plength
    g = BroadcastingFieldOpMap(getindex(f.op,i))
    cache[i] = return_cache(g,param_getindex(A,i),param_getindex.(B,i)...)
  end
  return cache,data
end

function Arrays.evaluate!(
  C,
  f::BroadcastingFieldOpMap{<:AbstractParamFunction},
  A::AbstractParamArray,
  b::Union{AbstractParamArray,AbstractArray{<:Number}}...)

  cache,data = C
  B = to_param_quantities(b...;plength=param_length(f.op))
  @inbounds for i in param_eachindex(f.op)
    g = BroadcastingFieldOpMap(getindex(f.op,i))
    vi = evaluate!(cache[i],g,param_getindex(A,i),param_getindex.(B,i)...)
    param_setindex!(data,vi,i)
  end
  data
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

function Fields.linear_combination(A::AbstractParamArray,b::AbstractVector{<:Field})
  ab = linear_combination(testitem(A),b)
  data = Vector{typeof(ab)}(undef,param_length(A))
  @inbounds for i in param_eachindex(A)
    data[i] = linear_combination(param_getindex(A,i),b)
  end
  ParamContainer(data)
end

function Arrays.return_cache(
  f::ParamContainer{<:Union{Field,ParamField,AbstractArray{<:Field}}},
  x::AbstractArray{<:Point})

  ci = return_cache(testitem(f),x)
  bi = evaluate!(ci,testitem(f),x)
  cache = Vector{typeof(ci)}(undef,param_length(f))
  array = Vector{typeof(bi)}(undef,param_length(f))
  @inbounds for i = param_eachindex(f)
    cache[i] = return_cache(param_getindex(f,i),x)
  end
  cache,ParamArray(array)
end

function Arrays.evaluate!(
  cache,
  f::ParamContainer{<:Union{Field,ParamField,AbstractArray{<:Field}}},
  x::AbstractArray{<:Point})

  cx,array = cache
  @inbounds for i = param_eachindex(array)
    array[i] = evaluate!(cx[i],param_getindex(f,i),x)
  end
  return array
end

for T in (:AbstractVector,:AbstractMatrix,:AbstractArray)
  @eval begin
    function Arrays.return_value(f::LinearCombinationMap{<:Integer},A::AbstractParamArray,b::$T)
      param_return_value(f,A,b)
    end

    function Arrays.return_cache(f::LinearCombinationMap{<:Integer},A::AbstractParamArray,b::$T)
      param_return_cache(f,A,b)
    end

    function Arrays.evaluate!(C,f::LinearCombinationMap{<:Integer},A::AbstractParamArray,b::$T)
      param_evaluate!(C,f,A,b)
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

for T in (:AbstractParamArray,:AbstractArray,:Nothing), S in (:AbstractParamArray,:AbstractArray)
  (T∈(:AbstractArray,:Nothing) && S==:AbstractArray) && continue
  @eval begin
    function Arrays.return_cache(f::Fields.ZeroBlockMap,A::$T,B::$S)
      pA,pB = to_param_quantities(A,B)
      map(get_param_data(pA),get_param_data(pB)) do a,b
        CachedArray(similar(a,eltype(a),size(b)))
      end |> ParamContainer
    end
  end
end

function Arrays.evaluate!(C::ParamContainer,f::Fields.ZeroBlockMap,A,B::AbstractArray)
  pA,pB = to_param_quantities(A,B;plength=param_length(C))
  map(C,get_param_data(pA),get_param_data(pB)) do c,a,b
    evaluate!(c,f,a,b)
  end |> ParamArray
end

# for T in (:AbstractParamArray,:AbstractArray,:Nothing), S in (:AbstractParamArray,:AbstractArray)
#   (T∈(:AbstractArray,:Nothing) && S==:AbstractArray) && continue
#   @eval begin
#     function Arrays.return_cache(f::Fields.ZeroBlockMap,A::$T,B::$S)
#       pA,pB = to_param_quantities(A,B)
#       c = return_cache(f,testitem(pA),testitem(pB))
#       cache = Vector{typeof(c)}(undef,param_length(pA))
#       @inbounds for i in param_eachindex(pA)
#         cache[i] = return_cache(f,param_getindex(pA,i),param_getindex(pB,i))
#       end
#       return ParamArray(cache)
#     end
#   end
# end

# for T in (:AbstractArray,:ArrayBlock)
#   @eval begin
#     function Arrays.evaluate!(C::AbstractParamArray,f::Fields.ZeroBlockMap,A,B::AbstractParamArray)
#       setsize!(C,size(B))
#       r = C.array
#       fill!(r,zero(eltype(r)))
#       ConsecutiveParamArray(r)
#     end
#   end
# end

function Fields.unwrap_cached_array(A::AbstractParamArray)
  C = param_return_cache(unwrap_cached_array,A)
  param_evaluate!(C,unwrap_cached_array,A)
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
      A = param_array(vi,length(ydual))
      return A
    end

    function Arrays.return_cache(
      f::AutoDiffMap,
      ydual::AbstractVector,
      x::AbstractParamArray,
      cfg::$T)

      @check length(ydual) == param_length(x)
      ci = return_cache(f,testitem(ydual),testitem(x),cfg)
      A = param_array(ci,length(ydual))
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
