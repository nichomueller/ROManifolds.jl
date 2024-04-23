struct TensorProductShapefuns{D,I,A} <: AbstractVector{TensorProductField{D,I}}
  factors::A
  function TensorProductShapefuns{D,I}(factors::A) where {D,I,A}
    new{D,I,A}(factors)
  end
end

function ReferenceFEs.compute_shapefuns(
  dofs::TensorProductDofBases{D,I},
  prebasis::TensorProductMonomialBasis{D,I}) where {D,I}

  factors = map(compute_shapefuns,dofs.factors,prebasis.factors)
  TensorProductShapefuns{D,I}(factors)
end

get_factors(a::TensorProductShapefuns) = a.factors

Base.size(a::TensorProductShapefuns{D}) where D = ntuple(d->prod(size.(a.factors,d)),D)
Base.getindex(a::TensorProductShapefuns,i::Integer...) = getindex(kronecker(get_factors(a)),i...)

function Arrays.return_cache(a::TensorProductShapefuns,x::TensorProductNodes)
  factors = get_factors(a)
  points = get_factors(x)
  s,c = return_cache(factors[1],points[1])
  r = Vector{typeof(get_array(s))}(undef,D)
  return r,(s,c)
end

function Arrays.evaluate!(cache,a::TensorProductShapefuns,x::TensorProductNodes{D}) where D
  r,c = cache
  factors = get_factors(a)
  points = get_factors(x)
  @inbounds for d = 1:D
    r[d] = evaluate!(c,factors[d],points[d])
  end
  return r
end

function Arrays.return_cache(
  a::TensorProductShapefuns{D,Isotropic},
  x::TensorProductNodes{D,Isotropic}
  ) where D

  factors = get_factors(a)
  points = get_factors(x)
  return return_cache(factors[1],points[1])
end

function Arrays.evaluate!(
  cache,
  a::TensorProductShapefuns{D,Isotropic},
  x::TensorProductNodes{D,Isotropic}
  ) where D

  factors = get_factors(a)
  points = get_factors(x)
  r = evaluate!(cache,factors[1],points[1])
  return Fill(r,D)
end
