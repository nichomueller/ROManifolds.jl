const ParamArray{T,N} = AbstractArrayOfSimilarArrays{T,N,N}
const ParamVector{T} = ParamArray{T,1}
const ParamMatrix{T} = ParamArray{T,2}
const ParamTensor3D{T} = ParamArray{T,3}

param_data(A::AbstractArrayOfSimilarArrays) = A.data
param_data(A::MatrixOfSparseMatricesCSC) = nonzeros(A)

param_length(A::ParamArray) = length(param_data(A))

Base.@propagate_inbounds function param_getindex(A::VectorOfSimilarArrays,i::Integer)
  getindex(A,i)
end

Base.@propagate_inbounds function param_getindex(A::MatrixOfSparseMatricesCSC,i::Integer)
  @boundscheck Base.checkbounds(A.nzval,:,i)
  SparseMatrixCSC(A.m,A.n,A.colptr,A.rowval,_nonzeros(A,i))
end

function array_of_similar_arrays(a::AbstractArray{<:Number},l::Integer)
  ArrayOfSimilarArrays([copy(a) for _ = 1:l])
end

function Base.maximum(f,A::ParamArray)
  maxa = -Inf
  @inbounds for a in param_data(A)
    maxa = max(maxa,maximum(f,a))
  end
  return maxa
end

function Base.minimum(f,A::ParamArray)
  maxa = Inf
  @inbounds for a in param_data(A)
    maxa = min(maxa,minimum(f,a))
  end
  return maxa
end

function Base.fill!(A::ParamArray,z)
  @inbounds for a in param_data(A)
    fill!(a,z)
  end
  return A
end

function LinearAlgebra.fillstored!(A::ParamArray,z)
  @inbounds for a in param_data(A)
    fillstored!(a,z)
  end
  return A
end

function LinearAlgebra.mul!(
  C::ParamArray,
  A::ParamArray,
  B::ParamArray,
  α::Number,β::Number)

  @check param_length(C) == param_length(A) == param_length(B)
  @inbounds for (c,a,b) in zip(param_data(C),param_data(A),param_data(B))
    mul!(c,a,b,α,β)
  end
  return C
end

function LinearAlgebra.axpy!(α::Number,A::ParamArray,B::ParamArray)
  @check param_length(A) == param_length(B)
  @inbounds for (a,b) in zip(param_data(A),param_data(B))
    axpy!(α,a,b)
  end
  return B
end

for factorization in (:LU,:Cholesky)
  @eval begin
    function LinearAlgebra.ldiv!(a::$factorization,B::ParamArray)
      @inbounds for b in param_data(B)
        ldiv!(a,b)
      end
      return B
    end
  end
end

function LinearAlgebra.ldiv!(A::ParamArray,b::Factorization,C::ParamArray)
  @check param_length(A) == param_length(B)
  @inbounds for (a,c) in zip(param_data(A),param_data(C))
    ldiv!(a,b,c)
  end
  return A
end

function LinearAlgebra.ldiv!(A::ParamArray,B::ParamContainer,C::ParamArray)
  @check param_length(A) == length(B) == length(C)
  @inbounds for (a,b,c) in zip(param_data(A),param_data(C),param_data(C))
    ldiv!(a,b,c)
  end
  return A
end

function LinearAlgebra.lu(A::ParamArray)
  ParamContainer(lu.(param_data(A)))
end

function LinearAlgebra.lu!(A::ParamArray,B::ParamArray)
  @inbounds for (a,b) in zip(param_data(A),param_data(B))
    lu!(a,b)
  end
  return A
end

function Arrays.CachedArray(A::ParamArray)
  ArrayOfSimilarArrays(CachedArray.(param_data(A)))
end

function Arrays.setsize!(A::ParamArray{T,N},s::NTuple{N,Integer}) where {T,N}
  @inbounds for a in param_data(A)
    setsize!(a,s)
  end
  return A
end

# Gridap interface

for T in (:ParamArray,:ParamField)
  S = T == :ParamArray ? :ParamField : :ParamArray
  @eval begin
    function _find_param_length(A::Union{$T,AbstractArray{<:Number}}...)
      B::Tuple{Vararg{$T}} = filter(a->isa(a,$T),A)
      @check all(param_length(first(B)) .== param_length.(B))
      return param_length(first(B))
    end

    function _find_param_length(A::Union{$T,$S}...)
      B::Tuple{Vararg{$T}} = filter(a->isa(a,$T),A)
      C::Tuple{Vararg{$S}} = filter(a->isa(a,$S),A)
      @check all(param_length(first(B)) .== param_length.(B))
      @check all(param_length(first(B)) .== param_length.(C))
      return param_length(first(B))
    end

    function _find_param_length(A::Union{$T,Field,AbstractArray{<:Number}}...)
      B::Tuple{Vararg{$T}} = filter(a->isa(a,$T),A)
      @check all(param_length(first(B)) .== param_length.(B))
      return param_length(first(B))
    end

    function _find_param_length(A::Union{$T,$S,AbstractArray{<:Number}}...)
      B::Tuple{Vararg{$T}} = filter(a->isa(a,$T),A)
      C::Tuple{Vararg{$S}} = filter(a->isa(a,$S),A)
      @check all(param_length(first(B)) .== param_length.(B))
      @check all(param_length(first(B)) .== param_length.(C))
      return param_length(first(B))
    end

    function _to_param_quantities(A::Union{$T,AbstractArray{<:Number}}...)
      plength = _find_param_length(A...)
      pA = map(a->ArrayOfTrivialArrays(a,plength),A)
      return pA
    end
  end
end

function _to_param_quantities(A::Union{ParamArray,Field}...)
  plength = _find_param_length(A...)
  pA = map(f->TrivialParamField(f,plength),A)
  return pA
end

function param_return_value(f::Union{Function,Map},A::Union{ParamArray,AbstractArray{<:Number},Field}...)
  pA = _to_param_quantities(A...)
  c = return_value(f,first.(param_data.(pA))...)
  data = Vector{typeof(c)}(undef,param_length(first(pA)))
  @inbounds for i in param_eachindex(first(pA))
    data[i] = return_value(f,param_getindex.(pA,i)...)
  end
  return ArrayOfSimilarArrays(data)
end

function param_return_cache(f::Union{Function,Map},A::Union{ParamArray,AbstractArray{<:Number},Field}...)
  pA = _to_param_quantities(A...)
  c = return_cache(f,first.(param_data.(pA))...)
  d = evaluate!(c,f,first.(param_data.(pA))...)
  cache = Vector{typeof(c)}(undef,param_length(first(pA)))
  data = Vector{typeof(d)}(undef,param_length(first(pA)))
  @inbounds for i in param_eachindex(first(pA))
    cache[i] = return_cache(f,param_getindex.(pA,i)...)
  end
  return cache,ArrayOfSimilarArrays(data)
end

function param_evaluate!(B,f::Union{Function,Map},A::Union{ParamArray,AbstractArray{<:Number},Field}...)
  cache,data = B
  pA = map(a->ArrayOfTrivialArrays(a,param_length(data)),A) #faster
  @inbounds for i in param_eachindex(data)
    data[i] = evaluate!(cache[i],f,param_getindex.(pA,i)...)
  end
  data
end

for T in (:ParamVector,:ParamMatrix,:ParamTensor3D)
  for S in (:ParamVector,:ParamMatrix,:ParamTensor3D)
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

function Arrays.return_value(f::BroadcastingFieldOpMap,A::Union{AbstractArray{<:Number},ParamArray}...)
  param_return_value(f,A...)
end

function Arrays.return_cache(f::BroadcastingFieldOpMap,A::Union{AbstractArray{<:Number},ParamArray}...)
  param_return_cache(f,A...)
end

function Arrays.evaluate!(C,f::BroadcastingFieldOpMap,A::Union{AbstractArray{<:Number},ParamArray}...)
  param_evaluate!(C,f,A...)
end

for op in (:+,:-,:*)
  @eval begin
    function Arrays.return_value(f::Broadcasting{typeof($op)},A::ParamArray,B::ParamArray)
      param_return_value(Fields.BroadcastingFieldOpMap($op),A,B)
    end

    function Arrays.return_cache(f::Broadcasting{typeof($op)},A::ParamArray,B::ParamArray)
      param_return_cache(Fields.BroadcastingFieldOpMap($op),A,B)
    end

    function Arrays.evaluate!(C,f::Broadcasting{typeof($op)},A::ParamArray,B::ParamArray)
      param_evaluate!(C,Fields.BroadcastingFieldOpMap($op),A,B)
    end
  end
  for T in (:ParamArray,:(AbstractArray{<:Number}))
    S = T == :ParamArray ? :(AbstractArray{<:Number}) : :ParamArray
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

function Arrays.return_value(::typeof(*),A::ParamArray,B::ParamArray)
  param_return_value(*,A,B)
end

function Arrays.return_value(f::Broadcasting,A::ParamArray)
  param_return_value(f,A)
end

function Arrays.return_cache(f::Broadcasting,A::ParamArray)
  param_return_cache(f,A)
end

function Arrays.evaluate!(C,f::Broadcasting,A::ParamArray)
  param_evaluate!(C,f,A)
end

for F in (:(typeof(∘)),:Operation)
  for T in (:ParamArray,:Field)
    S = T == :ParamArray ? :Field : :ParamArray
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

for T in (:ParamArray,:Number)
  S = T == :ParamArray ? :Number : :ParamArray
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

function Fields.linear_combination(A::ParamArray,b::AbstractVector{<:Field})
  ab = linear_combination(first(param_data(A)),b)
  data = Vector{typeof(ab)}(undef,param_length(A))
  @inbounds for i in param_eachindex(A)
    data[i] = linear_combination(param_getindex(A,i),b)
  end
  ParamContainer(data)
end

for T in (:AbstractVector,:AbstractMatrix,:AbstractArray)
  @eval begin
    function Arrays.return_value(f::LinearCombinationMap{<:Integer},A::ParamArray,b::$T)
      param_return_value(f,A,b)
    end

    function Arrays.return_cache(f::LinearCombinationMap{<:Integer},A::ParamArray,b::$T)
      param_return_cache(f,A,b)
    end

    function Arrays.evaluate!(C,f::LinearCombinationMap{<:Integer},A::ParamArray,b::$T)
      param_evaluate!(C,f,A,b)
    end
  end
end

function Arrays.return_value(f::IntegrationMap,A::ParamArray,w::AbstractVector{<:Real})
  param_return_value(f,A,w)
end

function Arrays.return_cache(f::IntegrationMap,A::ParamArray,w::AbstractVector{<:Real})
  param_return_cache(f,A,w)
end

function Arrays.evaluate!(C,f::IntegrationMap,A::ParamArray,w::AbstractVector{<:Real})
  param_evaluate!(C,f,A,w)
end

function Arrays.return_value(f::IntegrationMap,A::ParamArray,w::AbstractVector{<:Real},jq::AbstractVector)
  param_return_value(f,A,w,jq)
end

function Arrays.return_cache(f::IntegrationMap,A::ParamArray,w::AbstractVector{<:Real},jq::AbstractVector)
  param_return_cache(f,A,w,jq)
end

function Arrays.evaluate!(C,f::IntegrationMap,A::ParamArray,w::AbstractVector{<:Real},jq::AbstractVector)
  param_evaluate!(C,f,A,w,jq)
end

function Arrays.return_cache(::Fields.ZeroBlockMap,a::AbstractArray,B::ParamArray)
  A = array_of_similar_arrays(a,param_length(B))
  CachedArray(similar(A,eltype(A),innersize(B)))
end

function Arrays.return_cache(::Fields.ZeroBlockMap,A::ParamArray,B::ParamArray)
  CachedArray(similar(A,eltype(A),innersize(B)))
end

function Arrays.evaluate!(C::ParamArray,f::Fields.ZeroBlockMap,a,b::AbstractArray)
  _get_array(c::CachedArray) = c.array
  @inbounds for i = param_eachindex(C)
    evaluate!(param_getindex(C,i),f,a,b)
  end
  ArrayOfSimilarArrays(map(_get_array,cache))
end

function Fields.unwrap_cached_array(A::ParamArray)
  C = param_return_cache(unwrap_cached_array,A)
  param_evaluate!(C,unwrap_cached_array,A)
end

function Fields._setsize_as!(A::ParamArray,B::ParamArray)
  @check param_length(A) == param_length(B)
  @inbounds for i in param_eachindex(A)
    Fields._setsize_as!(param_getindex(A,i),param_getindex(B,i))
  end
  A
end

function Fields._setsize_mul!(C::ParamArray,A::ParamArray,B::ParamArray)
  @check param_length(A) == param_length(B)
  @inbounds for i = eachindex(C)
    Fields._setsize_mul!(param_getindex(C,i),param_getindex(A,i),param_getindex(B,i))
  end
end

function Fields._setsize_mul!(C,A::Union{ParamArray,AbstractArray}...)
  pA = _to_param_quantities(A...)
  Fields._setsize_mul!(C,pA...)
end

function Arrays.return_value(f::MulAddMap,A::ParamArray,B::ParamArray,C::ParamArray)
  x = return_value(*,A,B)
  return_value(+,x,C)
end

function Arrays.return_cache(f::MulAddMap,A::ParamArray,B::ParamArray,C::ParamArray)
  c1 = CachedArray(A*B+C)
  c2 = return_cache(Fields.unwrap_cached_array,c1)
  (c1,c2)
end

function Arrays.evaluate!(cache,f::MulAddMap,A::ParamArray,B::ParamArray,C::ParamArray)
  c1,c2 = cache
  Fields._setsize_as!(c1,C)
  Fields._setsize_mul!(c1,A,B)
  d = evaluate!(c2,Fields.unwrap_cached_array,c1)
  copyto!(d,C)
  mul!(d,A,B,f.α,f.β)
  d
end

function Arrays.return_cache(f::ConfigMap{typeof(ForwardDiff.gradient)},A::ParamArray)
  return_cache(f,first(param_data(A)))
end

function Arrays.return_cache(f::ConfigMap{typeof(ForwardDiff.jacobian)},A::ParamArray)
  return_cache(f,first(param_data(A)))
end

function Arrays.return_value(f::DualizeMap,A::ParamArray)
  param_return_value(f,A)
end

function Arrays.return_cache(f::DualizeMap,A::ParamArray)
  param_return_cache(f,A)
end

function Arrays.evaluate!(C,f::DualizeMap,A::ParamArray)
  param_evaluate!(C,f,A)
end

for T in (:(ForwardDiff.GradientConfig),:(ForwardDiff.JacobianConfig))
  @eval begin
    function Arrays.return_value(
      f::AutoDiffMap,
      ydual::AbstractVector,
      x::ParamArray,
      cfg::$T)

      @check length(ydual) == param_length(x)
      vi = return_value(f,first(ydual),first(param_data(x)),cfg)
      A = array_of_similar_arrays(vi,length(ydual))
      return A
    end

    function Arrays.return_cache(
      f::AutoDiffMap,
      ydual::AbstractVector,
      x::ParamArray,
      cfg::$T)

      @check length(ydual) == param_length(x)
      vi = return_cache(f,first(ydual),first(param_data(x)),cfg)
      A = array_of_similar_arrays(vi,length(ydual))
      return A
    end

    function Arrays.evaluate!(
      cache::ParamArray,
      f::AutoDiffMap,
      ydual::AbstractVector,
      x::ParamArray,
      cfg::$T)

      @inbounds for i = param_eachindex(cache)
        evaluate!(param_getindex(cache,i),f,ydual[i],param_getindex(x,i),cfg)
      end
      cache
    end
  end
end
