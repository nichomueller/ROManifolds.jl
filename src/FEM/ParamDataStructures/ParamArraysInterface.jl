abstract type AbstractParamArray{T,N,L} <: AbstractParamContainer{AbstractArray{T,N},N,L} end

ParamArray(A::AbstractVector{<:AbstractVector}) = VectorOfVectors(A)
ParamArray(A::AbstractVector{<:AbstractMatrix}) = VectorOfVectors(A)
ParamArray(A::AbstractVector{<:SparseMatrixCSC}) = MatrixOfSparseMatricesCSC(A)
ParamArray(A::AbstractArray{<:Number},plength=1) = ArrayOfTrivialArrays(A,plength)

function param_array(f,A::AbstractArray{<:AbstractArray})
  ParamArray(map(f,A))
end

function array_of_similar_arrays(a,l::Integer)
  @abstractmethod
end

function array_of_similar_arrays(a::Union{Number,AbstractArray{<:Number,0}},l::Integer)
  VectorOfScalars(fill(zero(eltype(a)),l))
end

function array_of_similar_arrays(a::AbstractVecOrMat{<:Number},l::Integer)
  ParamArray([similar(a) for _ = 1:l])
end

_to_param_quantity(A::AbstractParamArray,plength::Integer) = A
_to_param_quantity(a::AbstractArray,plength::Integer) = ParamArray(a,plength)

const AbstractParamVector{T,L} = AbstractParamArray{T,1,L}
const AbstractParamMatrix{T,L} = AbstractParamArray{T,2,L}
const AbstractParamTensor3D{T,L} = AbstractParamArray{T,3,L}

Arrays.testitem(A::AbstractParamArray) = param_getindex(A,1)

function Base.maximum(f,A::AbstractParamArray)
  maxa = -Inf
  @inbounds for i in param_eachindex(A)
    maxa = max(maxa,maximum(f,param_getindex(A,i)))
  end
  return maxa
end

function Base.minimum(f,A::AbstractParamArray)
  maxa = Inf
  @inbounds for i in param_eachindex(A)
    maxa = min(maxa,minimum(f,param_getindex(A,i)))
  end
  return maxa
end

function Base.fill!(A::AbstractParamArray,z)
  @inbounds for i in param_eachindex(A)
    fill!(param_getindex(A,i),z)
  end
  return A
end

function LinearAlgebra.fillstored!(A::AbstractParamArray,z)
  @inbounds for i in param_eachindex(A)
    fillstored!(param_getindex(A,i),z)
  end
  return A
end

function LinearAlgebra.mul!(
  C::AbstractParamArray,
  A::AbstractParamArray,
  B::AbstractParamArray,
  α::Number,β::Number)

  @check param_length(C) == param_length(A) == param_length(B)
  @inbounds for i in param_eachindex(C)
    mul!(param_getindex(C,i),param_getindex(A,i),param_getindex(B,i),α,β)
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
        ldiv!(a,param_getindex(B,i))
      end
      return B
    end
  end
end

function LinearAlgebra.ldiv!(A::AbstractParamArray,b::Factorization,C::AbstractParamArray)
  @check param_length(A) == param_length(C)
  @inbounds for i in param_eachindex(A)
    ldiv!(param_getindex(A,i),b,param_getindex(C,i))
  end
  return A
end

function LinearAlgebra.ldiv!(A::AbstractParamArray,B::ParamContainer,C::AbstractParamArray)
  @check param_length(A) == param_length(B) == length(C)
  @inbounds for i in param_eachindex(A)
    ldiv!(param_getindex(A,i),param_getindex(B,i),param_getindex(C,i))
  end
  return A
end

function LinearAlgebra.lu(A::AbstractParamArray)
  ParamContainer(lu.(A))
end

function LinearAlgebra.lu!(A::AbstractParamArray,B::AbstractParamArray)
  @check param_length(A) == param_length(B)
  @inbounds for i in param_eachindex(A)
    lu!(param_getindex(A,i),param_getindex(B,i))
  end
  return A
end

function Arrays.CachedArray(A::AbstractParamArray)
  ParamArray(CachedArray.([param_getindex(A,i) for i in param_eachindex(A)]))
end

function Arrays.setsize!(A::AbstractParamArray{T,N},s::NTuple{N,Integer}) where {T,N}
  @inbounds for i in param_eachindex(A)
    setsize!(param_getindex(A,i),s)
  end
  return A
end

# Gridap interface

function param_return_value(f::Union{Function,Map},A::Union{AbstractParamArray,AbstractArray{<:Number},Field}...)
  pA = _to_param_quantities(A...)
  c = return_value(f,testitem.(pA)...)
  data = Vector{typeof(c)}(undef,param_length(first(pA)))
  @inbounds for i in param_eachindex(first(pA))
    data[i] = return_value(f,param_getindex.(pA,i)...)
  end
  return ParamArray(data)
end

function param_return_cache(f::Union{Function,Map},A::Union{AbstractParamArray,AbstractArray{<:Number},Field}...)
  pA = _to_param_quantities(A...)
  c = return_cache(f,testitem.(pA)...)
  d = evaluate!(c,f,testitem.(pA)...)
  cache = Vector{typeof(c)}(undef,param_length(first(pA)))
  data = array_of_similar_arrays(d,param_length(first(pA)))
  @inbounds for i in param_eachindex(first(pA))
    cache[i] = return_cache(f,param_getindex.(pA,i)...)
  end
  return cache,data
end

function param_evaluate!(B,f::Union{Function,Map},A::Union{AbstractParamArray,AbstractArray{<:Number},Field}...)
  cache,data = B
  pA = map(a->ArrayOfTrivialArrays(a,param_length(data)),A) #faster
  @inbounds for i in param_eachindex(data)
    data[i] = evaluate!(cache[i],f,param_getindex.(pA,i)...)
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

function Arrays.return_value(f::BroadcastingFieldOpMap,A::Union{AbstractArray{<:Number},AbstractParamArray}...)
  param_return_value(f,A...)
end

function Arrays.return_cache(f::BroadcastingFieldOpMap,A::Union{AbstractArray{<:Number},AbstractParamArray}...)
  param_return_cache(f,A...)
end

function Arrays.evaluate!(C,f::BroadcastingFieldOpMap,A::Union{AbstractArray{<:Number},AbstractParamArray}...)
  param_evaluate!(C,f,A...)
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

function Arrays.return_cache(::Fields.ZeroBlockMap,a::AbstractArray,B::AbstractParamArray)
  A = array_of_similar_arrays(a,param_length(B))
  CachedArray(similar(A,eltype(A),innersize(B)))
end

function Arrays.return_cache(::Fields.ZeroBlockMap,A::AbstractParamArray,B::AbstractParamArray)
  @check param_length(A) == param_length(B)
  CachedArray(similar(A,eltype(A),innersize(B)))
end

function Arrays.evaluate!(C::AbstractParamArray,f::Fields.ZeroBlockMap,a,b::AbstractArray)
  _get_array(c::CachedArray) = c.array
  @inbounds for i = param_eachindex(C)
    evaluate!(param_getindex(C,i),f,a,b)
  end
  param_array(param_data(C)) do data
    _get_array(data)
  end
end

function Arrays.evaluate!(C::AbstractParamArray,f::Fields.ZeroBlockMap,a,B::AbstractParamArray)
  _get_array(c::CachedArray) = c.array
  @inbounds for i = param_eachindex(C)
    evaluate!(param_getindex(C,i),f,a,param_getindex(B,i))
  end
  param_array(param_data(C)) do data
    _get_array(data)
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
  pA = _to_param_quantities(A...)
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
      vi = return_value(f,first(ydual),testitem(x),cfg)
      A = array_of_similar_arrays(vi,length(ydual))
      return A
    end

    function Arrays.return_cache(
      f::AutoDiffMap,
      ydual::AbstractVector,
      x::AbstractParamArray,
      cfg::$T)

      @check length(ydual) == param_length(x)
      vi = return_cache(f,first(ydual),testitem(x),cfg)
      A = array_of_similar_arrays(vi,length(ydual))
      return A
    end

    function Arrays.evaluate!(
      cache::AbstractParamArray,
      f::AutoDiffMap,
      ydual::AbstractVector,
      x::AbstractParamArray,
      cfg::$T)

      @inbounds for i = param_eachindex(cache)
        evaluate!(param_getindex(cache,i),f,ydual[i],param_getindex(x,i),cfg)
      end
      cache
    end
  end
end
