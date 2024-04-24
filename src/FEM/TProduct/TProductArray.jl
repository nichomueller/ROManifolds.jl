abstract type TensorProductFactors{T} <: AbstractArray{T,2} end

struct FieldFactors{I,T,A,B,C} <: TensorProductFactors{T}
  factors::A
  product::B
  row_map::C
  col_map::C
  isotropy::I
  function FieldFactors(
    factors::A,row_map::C,col_map::C=trivial(row_map),isotropy::I=Isotropy(factors)
    ) where {T,A<:AbstractVector{<:AbstractMatrix{T}},C<:IndexMap,I}

    product = kronecker(factors...)
    B = typeof(product)
    new{I,T,A,B,C}(factors,product,row_map,col_map,isotropy)
  end
end

get_factors(a::FieldFactors) = a.factors
get_indices_map(a::FieldFactors) = (a.row_map,a.col_map)

Base.size(a::FieldFactors) = size(a.product)

function Base.getindex(a::FieldFactors,i::Integer,j::Integer)
  rowi = a.row_map.rmatrix[i,:]
  colj = a.col_map.rmatrix[:,j]
  rowi'*a.product*colj
end
