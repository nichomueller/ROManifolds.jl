struct TensorProductFactors{T,N,A,I} <: AbstractArray{T,N}
  factors::A
  index_map::I
  function TensorProductFactors(
    factors::A,index_map::I=identity
    ) where {T,N,A<:AbstractVector{<:AbstractArray{T,N}},I}
    new{T,N,A,I}(factors,index_map)
  end
end

function TensorProductFactors(factors::NTuple{D,<:AbstractArray}) where D
  TensorProductFactors(factors,identity)
end

Base.size(a::TensorProductFactors,d::Integer) = prod(size.(a.factors,d))
Base.size(a::TensorProductFactors{D}) where D = ntuple(d->size(a,d),D)
Base.axes(a::TensorProductFactors,d::Integer) = Base.OneTo(prod(size.(a.factors,d)))
Base.axes(a::TensorProductFactors) = ntuple(d->axes(a,d),D)

get_factors(a::TensorProductFactors) = a.factors

Base.IndexStyle(::TensorProductFactors) = IndexLinear()

function Base.getindex(a::TensorProductFactors,i::Integer)
  getindex(kronecker(a.factors...),a.index_map(i))
end
