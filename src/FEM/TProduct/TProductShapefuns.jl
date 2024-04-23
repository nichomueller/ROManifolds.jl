struct TensorProductShapefuns{D,I,A,B,C,E} <: AbstractVector{TensorProductField{D,I}}
  factors::A
  shapefuns::B
  indices_map::C
  shapes_indices_map::E
  function TensorProductShapefuns{D,I}(
    factors::A,
    shapefuns::B,
    indices_map::C,
    shapes_indices_map::E) where {D,I,A,B,C,E}
    new{D,I,A,B,C,E}(factors,shapefuns,indices_map,shapes_indices_map)
  end
end

function ReferenceFEs.compute_shapefuns(
  dofs::TensorProductDofBases{D,I},
  prebasis::TensorProductMonomialBasis{D,I}) where {D,I}

  factors = map(compute_shapefuns,dofs.factors,prebasis.factors)
  shapefuns = compute_shapefuns(dofs.basis,prebasis.basis)
  indices_map = get_indices_map(prebasis)
  shapes_indices_map = factors2shapefuns(factors,shapefuns,indices_map)
  TensorProductShapefuns{D,I}(factors,shapefuns,indices_map,shapes_indices_map)
end

get_factors(a::TensorProductShapefuns) = a.factors
get_field(a::TensorProductShapefuns) = a.shapefuns
get_indices_map(a::TensorProductShapefuns) = a.indices_map
get_shapes_indices_map(a::TensorProductShapefuns) = a.shapes_indices_map

Base.size(a::TensorProductShapefuns) = size(a.shapefuns)
Base.getindex(a::TensorProductShapefuns,i::Integer) = getindex(a.shapefuns,i)

function factors2shapefuns(factors::Fill,shapefuns,indices_map)
  _get_values(a::Fields.LinearCombinationFieldVector) = a.values
  ϕ = _get_values(shapefuns)
  ψ = TProduct.FieldFactors(map(_get_values,factors),indices_map,Isotropic())
  @check size(ϕ) == size(ψ)

  keep_ids_ψ = findall(ψ .!= ϕ)
  keep_ids_ϕ = copy(keep_ids_ψ)

  f2s = collect(CartesianIndices(ψ))
  for i in keep_ids_ψ
    ids = findall(ϕ[i] .== ψ)
    intersect!(ids,keep_ids_ϕ)
    @check !isempty(ids)
    f2s[i] = first(ids)
    deleteat!(keep_ids_ϕ,findfirst([f2s[i]] .== keep_ids_ϕ))
  end

  return f2s
end

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
  indices_map = get_indices_map(a)
  shapes_indices_map = get_shapes_indices_map(a)
  @inbounds for d = 1:D
    r[d] = evaluate!(c,factors[d],points[d])
  end
  tpr = FieldFactors(r,indices_map,Anisotropic())
  return compose(tpr,shapes_indices_map)
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
  indices_map = get_indices_map(a)
  shapes_indices_map = get_shapes_indices_map(a)
  r = evaluate!(cache,factors[1],points[1])
  tpr = FieldFactors(Fill(r,D),indices_map,Isotropic())
  return compose(tpr,shapes_indices_map)
end
