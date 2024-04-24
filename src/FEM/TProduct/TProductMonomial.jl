function _basis_from_factors(::Type{T},::Val{D},factors) where {D,T}
  orders = Tuple(map(get_order,factors))
  terms = _get_terms(orders)
  return MonomialBasis{D}(T,orders,terms)
end

struct TPMonomial{D} <: TPField end

struct TensorProductMonomialBasis{D,T,A} <: TensorProductField{D,Isotropic}
  factors::A
  basis::MonomialBasis{D,T}
end

function TensorProductMonomialBasis(::Type{T},p::Polytope{D},orders) where {T,D}
  function _compute_1d_mbasis(order=first(orders))
    compute_monomial_basis(T,SEGMENT,(order,))
  end
  factors = Fill(_compute_1d_mbasis(),D)
  basis = _basis_from_factors(T,Val(D),factors)
  TensorProductMonomialBasis(factors,basis)
end

function TensorProductMonomialBasis(::Type{T},p::Polytope{D},order::Integer=1) where {T,D}
  orders = tfill(order,Val(D))
  TensorProductMonomialBasis(T,p,orders)
end

Base.size(a::TensorProductMonomialBasis{D,T}) where {D,T} = size(a.basis)
Base.getindex(a::TensorProductMonomialBasis{D},i::Integer) where D = TPMonomial{D}()
Base.IndexStyle(::TensorProductMonomialBasis) = IndexLinear()

ReferenceFEs.get_order(a::TensorProductMonomialBasis) = get_order(a.basis)
ReferenceFEs.get_orders(a::TensorProductMonomialBasis) = get_orders(a.basis)
ReferenceFEs.get_exponents(a::TensorProductMonomialBasis) = get_exponents(a.basis)

get_field(b::TensorProductMonomialBasis) = b.basis
get_factors(b::TensorProductMonomialBasis) = b.factors

Arrays.get_array(a::CachedArray) = a.array

function Arrays.return_cache(
  f::TensorProductMonomialBasis{D},
  x::AbstractTensorProductPoints{D}
  ) where D

  factors = get_factors(f)
  points = get_factors(x)
  indices_map = get_indices_map(x)
  cache = return_cache(factors[1],points[1])
  r = _return_vec_cache(cache,D)
  return indices_map,r,cache
end

function Arrays.evaluate!(
  _cache,
  f::TensorProductMonomialBasis{D},
  x::AbstractTensorProductPoints{D}
  ) where D

  indices_map,r,cache = _cache
  factors = get_factors(f)
  points = get_factors(x)
  @inbounds for d = 1:D
    r[d] = evaluate!(cache,factors[d],points[d])
  end
  return FieldFactors(r,indices_map)
end

# Isotropic shortcuts

function Arrays.return_cache(
  f::TensorProductMonomialBasis{D},
  x::TensorProductNodes{D,Isotropic}) where D

  factors = get_factors(f)
  points = get_factors(x)
  indices_map = get_indices_map(x)
  cache = return_cache(factors[1],points[1])
  return indices_map,cache
end

function Arrays.evaluate!(
  _cache,
  f::TensorProductMonomialBasis{D},
  x::TensorProductNodes{D,Isotropic}
  ) where D

  indices_map,cache = _cache
  factors = get_factors(f)
  points = get_factors(x)
  r = evaluate!(cache,factors[1],points[1])
  return FieldFactors(Fill(r,D),indices_map)
end

# # gradients

# function get_factors(a::FieldGradientArray{N,<:TensorProductMonomialBasis}) where N
#   FieldGradientArray{N}.(get_factors(a.fa))
# end

# function Arrays.return_cache(
#   fg::FieldGradientArray{N,<:TensorProductMonomialBasis{D}},
#   x::TensorProductNodes{D}) where {N,D}

#   factors = get_factors(fg)
#   points = get_factors(x)
#   indices_map = get_indices_map(x)
#   r,v,c,g = return_cache(factors[1],points[1])
#   vr = Vector{typeof(get_array(r))}(undef,D)
#   return indices_map,vr,(r,v,c,g)
# end

# function Arrays.evaluate!(
#   cache,
#   fg::FieldGradientArray{N,<:TensorProductMonomialBasis{D}},
#   x::AbstractTensorProductPoints{D}
#   ) where {N,D}

#   indices_map,vr,c = cache
#   factors = get_factors(fg)
#   points = get_factors(x)
#   @inbounds for d = 1:D
#     vr[d] = evaluate!(c,factors[d],points[d])
#   end
#   return FieldFactors(vr,indices_map)
# end

# # Isotropic shortcuts

# function Arrays.return_cache(
#   fg::FieldGradientArray{N,<:TensorProductMonomialBasis{D}},
#   x::TensorProductNodes{D,Isotropic}
#   ) where {N,D}

#   factors = get_factors(fg)
#   points = get_factors(x)
#   indices_map = get_indices_map(x)
#   cache = return_cache(factors[1],points[1])
#   return indices_map,cache
# end

# function Arrays.evaluate!(
#   _cache,
#   fg::FieldGradientArray{N,<:TensorProductMonomialBasis{D}},
#   x::TensorProductNodes{D,Isotropic}
#   ) where {N,D}

#   indices_map,cache = _cache
#   factors = get_factors(fg)
#   points = get_factors(x)
#   r = evaluate!(cache,factors[1],points[1])
#   return FieldFactors(Fill(r,D),indices_map)
# end
