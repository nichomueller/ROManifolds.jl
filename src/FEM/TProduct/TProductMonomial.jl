function _basis_from_factors(::Type{T},::Val{D},factors) where {D,T}
  get_inner_terms(b::MonomialBasis) = b.terms
  orders = Tuple(map(get_order,factors))
  _terms = map(get_inner_terms,factors)
  nterms = Tuple(map(length,_terms))
  terms = collect1d(CartesianIndices(nterms))
  return MonomialBasis{D}(T,orders,terms)
end

struct TensorProductMonomial <: TensorProductField end

struct TensorProductMonomialBasis{D,T,A} <: AbstractVector{TensorProductMonomial}
  factors::A
  basis::MonomialBasis{D,T}
end

function TensorProductMonomialBasis(::Type{T},p::Polytope{D},orders) where {T,D}
  function _compute_1d_mbasis(order)
    compute_monomial_basis(eltype(T),SEGMENT,(order,))
  end
  o1 = first(orders)
  isotropy = all(orders .== o1)
  factors = isotropy ? Fill(_compute_1d_mbasis(o1),D) : map(_compute_1d_mbasis,orders)
  basis = _basis_from_factors(T,Val(D),factors)
  TensorProductMonomialBasis(factors,basis)
end

function TensorProductMonomialBasis(::Type{T},p::Polytope{D},order::Integer=1) where {T,D}
  orders = tfill(order,Val(D))
  TensorProductMonomialBasis(T,p,orders)
end

Base.size(a::TensorProductMonomialBasis{D,T}) where {D,T} = size(a.basis)
Base.getindex(a::TensorProductMonomialBasis,i::Integer) = TensorProductMonomial()
Base.IndexStyle(::TensorProductMonomialBasis) = IndexLinear()

ReferenceFEs.get_order(a::TensorProductMonomialBasis) = get_order(a.basis)
ReferenceFEs.get_orders(a::TensorProductMonomialBasis) = get_orders(a.basis)
ReferenceFEs.get_exponents(a::TensorProductMonomialBasis) = get_exponents(a.basis)

get_factors(b::TensorProductMonomialBasis) = b.factors

Arrays.get_array(a::CachedArray) = a.array

function Arrays.return_cache(
  f::TensorProductMonomialBasis{D,T},
  x::AbstractVector{<:Point}
  ) where {D,T}

  return_cache(f.basis,x)
end

function Arrays.evaluate!(
  cache,
  f::TensorProductMonomialBasis{D,T},
  x::AbstractVector{<:Point}
  ) where {D,T}

  evaluate!(cache,f.basis,x)
end

function Arrays.return_cache(
  fg::FieldGradientArray{Ng,TensorProductMonomialBasis{D,T}},
  x::AbstractVector{<:Point}) where {Ng,D,T}

  return_cache(FieldGradientArray{Ng}(fg.fa.basis),x)
end

function Arrays.evaluate!(
  cache,
  fg::FieldGradientArray{Ng,TensorProductMonomialBasis{D,T}},
  x::AbstractVector{<:Point}) where {Ng,D,T}

  evaluate!(cache,FieldGradientArray{Ng}(fg.fa.basis),x)
end

function Arrays.return_cache(
  f::TensorProductMonomialBasis{D,T},
  x::TensorProductNodes{D,T}
  ) where {D,T}

  r,v,c = return_cache(f.factors[1],x.nodes[1])
  vr = map(_->similar(typeof(r),size(r)),1:D)
  return (vr,v,c)
end

function Arrays.evaluate!(
  cache,
  f::TensorProductMonomialBasis{D,T},
  x::TensorProductNodes{D,T}
  ) where {D,T}

  vr,v,c = cache
  @inbounds for d = 1:D
    cached = vr[d],v,c
    evaluate!(cached,f.factors[d],x.nodes[d])
  end
  return map(get_array,vr)
end

# Fill shortcut

function Arrays.return_cache(
  f::TensorProductMonomialBasis{D,T,<:Fill},
  x::TensorProductNodes{D,T,<:Fill}) where {D,T}

  return_cache(f.factors[1],x.nodes[1])
end

function Arrays.evaluate!(
  cache,
  f::TensorProductMonomialBasis{D,T,<:Fill},
  x::TensorProductNodes{D,T,<:Fill}
  ) where {D,T}

  r = evaluate!(cache,f.factors[1],x.nodes[1])
  Fill(r,D)
end
