abstract type AbstractParamArray{T,N,L,A<:AbstractArray{T,N}} <: AbstractParamContainer{A,N,L} end
abstract type ParamArray{T,N,L} <: AbstractParamArray{T,N,L,Array{T,N}} end
abstract type ParamCachedArray{T,N,L} <: AbstractParamArray{T,N,L,CachedArray{T,N}} end
abstract type ParamSparseMatrix{Tv,Ti,L,A<:AbstractSparseMatrix{Tv,Ti}} <: AbstractParamArray{Tv,2,L,A} end
abstract type ParamSparseMatrixCSC{Tv,Ti,L} <: ParamSparseMatrix{Tv,Ti,L,SparseMatrixCSC{Tv,Ti}} end

ParamArray(A::AbstractVector{<:AbstractArray}) = ArrayOfArrays(A)
ParamArray(A::AbstractVector{<:SparseMatrixCSC}) = MatrixOfSparseMatricesCSC(A)
ParamArray(A::AbstractArray{<:Number},plength=1) = ArrayOfTrivialArrays(A,plength)
ParamArray(A::AbstractArray{<:ParamArray}) = mortar(A)

function param_array(f,A::AbstractArray{<:AbstractArray}...)
  ParamArray(map(f,A...))
end

function array_of_similar_arrays(a::AbstractArray{<:Number},l::Integer)
  ParamArray([similar(a) for _ = 1:l])
end

function array_of_zero_arrays(a::AbstractArray{<:Number},l::Integer)
  A = array_of_similar_arrays(a,l)
  fill!(A,zero(eltype(a)))
  return A
end

to_param_quantity(A::AbstractParamArray,plength::Integer) = A
to_param_quantity(a::AbstractArray,plength::Integer) = ParamArray(a,plength)

Base.:(==)(A::AbstractParamArray,B::AbstractParamArray) = all(param_data(A) .== param_data(B))
Base.:(≈)(A::AbstractParamArray,B::AbstractParamArray) = all(param_data(A) .≈ param_data(B))

for op in (:+,:-)
  @eval begin
    function ($op)(A::AbstractParamArray,b::AbstractArray{<:Number})
      B = ParamArray(b,param_length(A))
      ($op)(A,B)
    end

    function ($op)(a::AbstractArray{<:Number},B::AbstractParamArray)
      A = ParamArray(a,param_length(B))
      ($op)(A,B)
    end
  end
end

const AbstractParamVector{T,L} = AbstractParamArray{T,1,L}
const AbstractParamMatrix{T,L} = AbstractParamArray{T,2,L}
const AbstractParamTensor3D{T,L} = AbstractParamArray{T,3,L}

for f in (:(Base.maximum),:(Base.minimum))
  @eval begin
    $f(A::AbstractParamArray) = $f(map($f,param_data(A)))
    $f(g,A::AbstractParamArray) = $f(map(a -> $f(g,a),param_data(A)))
  end
end

function Base.transpose(A::AbstractParamArray)
  param_array(param_data(A)) do v
    transpose(v)
  end
end

# small hack
function Base.zero(::Type{<:AbstractArray{T,N}}) where {T<:Number,N}
  zeros(T,tfill(1,Val{N}()))
end

for f in (:(Base.fill!),:(LinearAlgebra.fillstored!))
  @eval begin
    function $f(A::AbstractParamArray,z::Number)
      map(a -> $f(a,z),param_data(A))
      return A
    end

    # small hack
    function $f(A::AbstractParamArray{T,N},z::AbstractArray{<:Number,N}) where {T,N}
      @check all(z.==first(z))
      $f(A,first(z))
      return A
    end
  end
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

function LinearAlgebra.axpy!(α::Number,A::AbstractParamArray,B::AbstractParamArray)
  @check param_length(A) == param_length(B)
  @inbounds for i in param_eachindex(A)
    axpy!(α,param_getindex(A,i),param_getindex(B,i))
  end
  return B
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
  ParamContainer(lu.(param_data(A);kwargs...))
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
  array_of_similar_arrays(tv,L)
end

function Arrays.setsize!(A::AbstractParamArray{T,N},s::NTuple{N,Integer}) where {T,N}
  @inbounds for i in param_eachindex(A)
    setsize!(param_getindex(A,i),s)
  end
  return A
end

function param_return_value(f::Union{Function,Map},A...)
  pA = to_param_quantities(A...)
  c = return_value(f,testitem.(pA)...)
  data = array_of_similar_arrays(c,param_length(first(pA)))
  return data
end

function param_return_cache(f::Union{Function,Map},A...)
  pA = to_param_quantities(A...)
  c = return_cache(f,testitem.(pA)...)
  d = evaluate!(c,f,testitem.(pA)...)
  cache = Vector{typeof(c)}(undef,param_length(first(pA)))
  data = array_of_similar_arrays(d,param_length(first(pA)))
  @inbounds for i in param_eachindex(first(pA))
    cache[i] = return_cache(f,param_getindex.(pA,i)...)
  end
  return cache,data
end

function param_evaluate!(C,f::Union{Function,Map},A...)
  cache,data = C
  pA = to_param_quantities(A...;plength=param_length(data))
  @inbounds for i in param_eachindex(data)
    vi = evaluate!(cache[i],f,param_getindex.(pA,i)...)
    param_setindex!(data,vi,i)
  end
  data
end

for T in (:AbstractParamVector,:AbstractParamMatrix,:AbstractParamTensor3D)
  for S in (:AbstractParamVector,:AbstractParamMatrix,:AbstractParamTensor3D)
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
      param_array(param_data(pA),param_data(pB)) do a,b
        CachedArray(similar(a,eltype(a),size(b)))
      end
    end
  end
end

function Arrays.evaluate!(C::AbstractParamArray,f::Fields.ZeroBlockMap,A::AbstractArray,B::AbstractArray)
  _,pA,pB = to_param_quantities(C,A,B)
  param_array(param_data(C),param_data(pA),param_data(pB)) do c,a,b
    evaluate!(c,f,collect(a),collect(b))
  end
end

function Arrays.evaluate!(C::AbstractParamArray,f::Fields.ZeroBlockMap,a::Nothing,B::AbstractArray)
  _,pB = to_param_quantities(C,B)
  param_array(param_data(C),param_data(pB)) do c,b
    evaluate!(c,f,a,collect(b))
  end
end

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
      A = array_of_similar_arrays(vi,length(ydual))
      return A
    end

    function Arrays.return_cache(
      f::AutoDiffMap,
      ydual::AbstractVector,
      x::AbstractParamArray,
      cfg::$T)

      @check length(ydual) == param_length(x)
      ci = return_cache(f,testitem(ydual),testitem(x),cfg)
      A = array_of_similar_arrays(ci,length(ydual))
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
