abstract type TensorProductFactors{T} <: AbstractArray{T,2} end

struct FieldFactors{I,T,A,B} <: TensorProductFactors{T}
  factors::A
  indices_map::B
  isotropy::I
  function FieldFactors(
    factors::A,indices_map::B,isotropy::I=Isotropy(factors)
    ) where {T,A<:AbstractVector{<:AbstractMatrix{T}},B<:IndexMap,I}
    new{I,T,A,B}(factors,indices_map,isotropy)
  end
end

get_factors(a::FieldFactors) = a.factors
get_indices_map(a::FieldFactors) = a.indices_map

Base.size(a::FieldFactors) = size(a.indices_map.rmatrix)

function Base.getindex(a::FieldFactors,i::Integer,j::Integer)
  dot(a.indices_map.rmatrix[i,:],kronecker(a.factors...)[:,j])
end
