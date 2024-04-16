struct TensorProductMonomial <: TensorProductField end

struct TensorProductMonomialBasis{D,T} <: AbstractVector{TensorProductMonomial}
  bases::NTuple{D,MonomialBasis{1,T}}
  basis::MonomialBasis{D,T}
end

function TensorProductMonomialBasis(bases::NTuple{D,MonomialBasis{1,T}}) where {D,T}
  get_inner_terms(b::MonomialBasis) = b.terms
  _terms = get_inner_terms.(bases)
  nterms = map(length,_terms)
  terms = collect1d(CartesianIndices(nterms))
  orders = ntuple(i->get_order(bases[i]),D)
  basis = MonomialBasis{D}(T,orders,terms)
  TensorProductMonomialBasis(bases,basis)
end

Base.size(a::TensorProductMonomialBasis{D,T}) where {D,T} = size(a.basis)
Base.getindex(a::TensorProductMonomialBasis,i::Integer) = TensorProductMonomial()
Base.IndexStyle(::TensorProductMonomialBasis) = IndexLinear()

ReferenceFEs.get_order(a::TensorProductMonomialBasis) = get_order(a.basis)
ReferenceFEs.get_orders(a::TensorProductMonomialBasis) = get_orders(a.basis)
ReferenceFEs.get_exponents(a::TensorProductMonomialBasis) = get_exponents(a.basis)

function Arrays.return_cache(f::TensorProductMonomialBasis{D,T},x::AbstractVector{<:Point}) where {D,T}
  return_cache(f.basis,x)
end

function Arrays.evaluate!(cache,f::TensorProductMonomialBasis{D,T},x::AbstractVector{<:Point}) where {D,T}
  evaluate!(cache,f.basis,x)
end

function Arrays.return_cache(f::TensorProductMonomialBasis{D,T},x::TensorProductNodes) where {D,T}
  r,v,c = return_cache(f.bases[1],x.nodes[1])
  vr = Vector{typeof(r.array)}(undef,D)
  tr = Vector{typeof(r.array)}(undef,D)
  tr,(r,v,c)
end

function Arrays.evaluate!(cache,f::TensorProductMonomialBasis{D,T},x::TensorProductNodes) where {D,T}
  ntuple(d->evaluate!(cache[d],f.bases[d],x.nodes[d]),D)
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
