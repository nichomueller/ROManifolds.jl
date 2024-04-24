abstract type TensorProductFactors{D,T,N} <: AbstractArray{T,N} end

struct FieldFactors{D,I,T,N,A,B,C,E} <: TensorProductFactors{D,T,N}
  factors::A
  product::B
  row_map::C
  col_map::E
  isotropy::I
  function FieldFactors(
    factors::A,row_map::C,col_map::E=trivial(row_map),isotropy::I=Isotropy(factors)
    ) where {T,A<:AbstractVector{<:AbstractArray{T,N}},C<:IndexMap,E<:IndexMap,I}

    product = kronecker(factors...)
    D = length(factors)
    B = typeof(product)
    new{D,I,T,N,A,B,C,E}(factors,product,row_map,col_map,isotropy)
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

function Base.copy(a::FieldFactors)
  FieldFactors(copy.(a.factors),a.row_map,a.col_map,a.isotropy)
end
