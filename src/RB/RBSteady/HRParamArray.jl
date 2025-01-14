"""
    struct HRParamArray{T,N,A,B,C<:ParamArray{T,N}} <: ParamArray{T,N}
      fe_quantity::A
      coeff::B
      hypred::C
    end

Parametric vector returned after the online phase of a hyper-reduction strategy.
Fields:

- `fe_quantity`: represents a parametric residual/jacobian computed via integration
  on an AbstractIntegrationDomain
- `coeff`: parameter-dependent coefficient computed during the online phase
  according to the formula `coeff = Φi⁻¹ * fe_quantity[i,:]`, where (Φi,i) are
  stored in a `HyperReduction` object
- `hypred`: the ouptut of the online phase of a hyper-reduction strategy, acoording
  to the formula `hypred = Φrb * coeff`, where Φrb is stored in a
  `HyperReduction` object
"""
struct HRParamArray{T,N,A,B,C<:ParamArray{T,N}} <: ParamArray{T,N}
  fe_quantity::A
  coeff::B
  hypred::C
end

Base.size(a::HRParamArray) = size(a.hypred)
Base.getindex(a::HRParamArray{T,N},i::Vararg{Integer,N}) where {T,N} = getindex(a.hypred,i...)
Base.setindex!(a::HRParamArray{T,N},v,i::Vararg{Integer,N}) where {T,N} = setindex!(a.hypred,v,i...)
ParamDataStructures.param_length(a::HRParamArray) = param_length(a.hypred)
ParamDataStructures.get_all_data(a::HRParamArray) = get_all_data(a.hypred)
ParamDataStructures.param_getindex(a::HRParamArray,i::Integer) = param_getindex(a.hypred,i)

for f in (:(Base.copy),:(Base.similar))
  @eval begin
    function $f(a::HRParamArray)
      fe_quantity′ = $f(a.fe_quantity)
      coeff′ = $f(a.coeff)
      hypred′ = $f(a.hypred)
      HRParamArray(fe_quantity′,coeff′,hypred′)
    end
  end
end

function Base.copyto!(a::HRParamArray,b::HRParamArray)
  copyto!(a.fe_quantity,b.fe_quantity)
  copyto!(a.coeff,b.coeff)
  copyto!(a.hypred,b.hypred)
  a
end

function Base.fill!(a::HRParamArray,b::Number)
  fill!(a.hypred,b)
end

function LinearAlgebra.rmul!(a::HRParamArray,b::Number)
  rmul!(a.hypred,b)
end

function LinearAlgebra.axpy!(α::Number,a::HRParamArray,b::HRParamArray)
  axpy!(α,a.hypred,b.hypred)
end

function LinearAlgebra.axpy!(α::Number,a::HRParamArray,b::ParamArray)
  axpy!(α,a.hypred,b)
end

function LinearAlgebra.norm(a::HRParamArray)
  norm(a.hypred)
end

for (T,S) in zip((:AffineContribution,:BlockHyperReduction),(:ArrayContribution,:ArrayBlock))
  @eval begin
    function inv_project!(cache::HRParamArray,a::$T,b::$S)
      hypred = cache.hypred
      coeff = cache.coeff
      inv_project!(hypred,coeff,a,b)
    end
  end
end
