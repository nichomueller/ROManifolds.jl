struct TensorProductArray{D,T,N,I} <: AbstractArray{T,N}
  factors::NTuple{D,AbstractArray{T,N}}
  index_map::I
end

function TensorProductArray(factors::NTuple{D,<:AbstractArray}) where D
  TensorProductArray(factors,identity)
end

Base.size(a::TensorProductArray,d::Integer) = prod(size.(a.factors,d))
Base.size(a::TensorProductArray{D}) where D = ntuple(d->size(a,d),D)
Base.axes(a::TensorProductArray,d::Integer) = Base.OneTo(prod(size.(a.factors,d)))
Base.axes(a::TensorProductArray) = ntuple(d->axes(a,d),D)

get_factors(a::TensorProductArray) = a.factors

Base.IndexStyle(::TensorProductArray) = IndexLinear()

function Base.getindex(a::TensorProductArray,i::Integer)
  getindex(kronecker(a.factors...),a.index_map(i))
end
