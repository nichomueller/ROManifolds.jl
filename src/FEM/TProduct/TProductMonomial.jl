function _basis_from_factors(::Type{T},::Val{D},factors) where {D,T}
  orders = Tuple(map(get_order,factors))
  terms = _get_terms(orders)
  return MonomialBasis{D}(T,orders,terms)
end

struct TensorProductMonomial{D,I} <: TensorProductField{D,I} end

struct TensorProductMonomialBasis{D,I,T,A,B} <: AbstractVector{TensorProductMonomial{D,I}}
  factors::A
  basis::MonomialBasis{D,T}
  indices_map::B
  isotropy::I
end

function TensorProductMonomialBasis(::Type{T},p::Polytope{D},orders) where {T,D}
  function _compute_1d_mbasis(order=first(orders))
    compute_monomial_basis(T,SEGMENT,(order,))
  end
  isotropy = Isotropy(orders)
  indices_map = compute_nodes_map(;polytope=p,orders)
  factors = isotropy==Isotropic() ? Fill(_compute_1d_mbasis(),D) : map(_compute_1d_mbasis,orders)
  basis = _basis_from_factors(T,Val(D),factors)
  TensorProductMonomialBasis(factors,basis,indices_map,isotropy)
end

function TensorProductMonomialBasis(::Type{T},p::Polytope{D},order::Integer=1) where {T,D}
  orders = tfill(order,Val(D))
  TensorProductMonomialBasis(T,p,orders)
end

Base.size(a::TensorProductMonomialBasis{D,T}) where {D,T} = size(a.basis)
Base.getindex(a::TensorProductMonomialBasis{D,I},i::Integer) where {D,I} = TensorProductMonomial{D,I}()
Base.IndexStyle(::TensorProductMonomialBasis) = IndexLinear()

ReferenceFEs.get_order(a::TensorProductMonomialBasis) = get_order(a.basis)
ReferenceFEs.get_orders(a::TensorProductMonomialBasis) = get_orders(a.basis)
ReferenceFEs.get_exponents(a::TensorProductMonomialBasis) = get_exponents(a.basis)

get_field(b::TensorProductMonomialBasis) = b.basis
get_factors(b::TensorProductMonomialBasis) = b.factors
get_indices_map(b::TensorProductMonomialBasis) = b.indices_map

Arrays.get_array(a::CachedArray) = a.array

function Arrays.return_cache(
  f::TensorProductMonomialBasis{D},
  x::AbstractTensorProductPoints{D}
  ) where D

  factors = get_factors(f)
  points = get_factors(x)
  r,v,c = return_cache(factors[1],points[1])
  vr = Vector{typeof(get_array(r))}(undef,D)
  return vr,(r,v,c)
end

function Arrays.evaluate!(
  cache,
  f::TensorProductMonomialBasis{D},
  x::AbstractTensorProductPoints{D}
  ) where D

  vr,c = cache
  factors = get_factors(f)
  points = get_factors(x)
  indices_map = get_indices_map(f)
  @inbounds for d = 1:D
    vr[d] = evaluate!(c,factors[d],points[d])
  end
  return FieldFactors(vr,indices_map,Anisotropic())
end

# Isotropic shortcuts

function Arrays.return_cache(
  f::TensorProductMonomialBasis{D,Isotropic},
  x::TensorProductNodes{D,Isotropic}) where D

  factors = get_factors(f)
  points = get_factors(x)
  return return_cache(factors[1],points[1])
end

function Arrays.evaluate!(
  cache,
  f::TensorProductMonomialBasis{D,Isotropic},
  x::TensorProductNodes{D,Isotropic}
  ) where D

  factors = get_factors(f)
  points = get_factors(x)
  indices_map = get_indices_map(f)
  r = evaluate!(cache,factors[1],points[1])
  return FieldFactors(Fill(r,D),indices_map,Isotropic())
end
