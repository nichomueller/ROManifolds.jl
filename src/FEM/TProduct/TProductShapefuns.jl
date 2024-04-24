struct TensorProductShapefuns{D,I,A,B,C} <: TensorProductField{D,I}
  factors::A
  shapefuns::B
  indices_map::C
  function TensorProductShapefuns{D,I}(factors::A,shapefuns::B,indices_map::C) where {D,I,A,B,C}
    new{D,I,A,B,C}(factors,shapefuns,indices_map)
  end
end

function ReferenceFEs.compute_shapefuns(
  dofs::TensorProductDofBases{D,I},
  prebasis::TensorProductMonomialBasis{D}) where {D,I}

  factors = map(compute_shapefuns,get_factors(dofs),get_factors(prebasis))
  shapefuns = compute_shapefuns(get_field(dofs),get_field(prebasis))
  indices_map = inv(get_indices_map(dofs))
  TensorProductShapefuns{D,I}(factors,shapefuns,indices_map)
end

get_factors(a::TensorProductShapefuns) = a.factors
get_indices_map(a::TensorProductShapefuns) = a.indices_map
get_field(a::TensorProductShapefuns{D,I}) where {D,I} = a.shapefuns

Base.length(a::TensorProductShapefuns) = prod(length.(a.factors))
Base.size(a::TensorProductShapefuns) = (length(a),)

function Base.getindex(a::TensorProductShapefuns,i::Integer)
  iH = OneHotVector(i,length(a))
  ids = get_indices_map(a)
  ri = findfirst(ids.rmatrix*iH .== 1)
  GenericTPField(a.factors,ri)
end

function Arrays.return_cache(a::TensorProductShapefuns{D},x::TensorProductNodes{D}) where D
  factors = get_factors(a)
  points = get_factors(x)
  row_map = get_indices_map(x)
  col_map = get_indices_map(a)
  cache = return_cache(factors[1],points[1])
  r = _return_vec_cache(cache,D)
  return row_map,col_map,r,cache
end

function Arrays.evaluate!(_cache,a::TensorProductShapefuns,x::TensorProductNodes{D}) where D
  row_map,col_map,r,cache = _cache
  factors = get_factors(a)
  points = get_factors(x)
  @inbounds for d = 1:D
    r[d] = evaluate!(cache,factors[d],points[d])
  end
  return FieldFactors(r,row_map,col_map)
end

function Arrays.return_cache(
  a::TensorProductShapefuns{D,Isotropic},
  x::TensorProductNodes{D,Isotropic}
  ) where D

  factors = get_factors(a)
  points = get_factors(x)
  row_map = get_indices_map(x)
  col_map = get_indices_map(a)
  cache = return_cache(factors[1],points[1])
  return row_map,col_map,cache
end

function Arrays.evaluate!(
  _cache,
  a::TensorProductShapefuns{D,Isotropic},
  x::TensorProductNodes{D,Isotropic}
  ) where D

  row_map,col_map,cache = _cache
  factors = get_factors(a)
  points = get_factors(x)
  r = evaluate!(cache,factors[1],points[1])
  return FieldFactors(Fill(r,D),row_map,col_map)
end
