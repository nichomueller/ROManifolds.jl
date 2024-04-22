# shapes

struct TensorProductShapefuns{D,I,A,B} <: AbstractVector{TensorProductField{D,I}}
  factors::A
  shapefuns::B
  function TensorProductShapefuns{D,I}(factors::A,shapefuns::B) where {D,I,A,B}
    new{D,I,A,B}(factors,shapefuns)
  end
end

function ReferenceFEs.compute_shapefuns(
  dofs::TensorProductDofBases{D,I},
  prebasis::TensorProductMonomialBasis{D,I}) where {D,I}

  factors = map(compute_shapefuns,dofs.factors,prebasis.factors)
  shapefuns = compute_shapefuns(dofs.basis,prebasis.basis)
  TensorProductShapefuns{D,I}(factors,shapefuns)
end

get_factors(a::TensorProductShapefuns) = a.factors
get_field(a::TensorProductShapefuns) = a.shapefuns

Base.size(a::TensorProductShapefuns) = size(a.shapefuns)
Base.getindex(a::TensorProductShapefuns,i::Integer) = getindex(a.shapefuns,i)

function Arrays.return_cache(a::TensorProductShapefuns,x::TensorProductNodes)
  field = get_field(a)
  factors = get_factors(a)
  points = get_factors(x)
  nodes_map = get_indices_map(x)
  indices_map = compute_nodes_and_comps_2_dof_map(field.fields,nodes_map)
  s,c = return_cache(factors[1],points[1])
  r = Vector{typeof(get_array(s))}(undef,D)
  return indices_map,r,(s,c)
end

function Arrays.evaluate!(cache,a::TensorProductShapefuns,x::TensorProductNodes{D}) where D
  indices_map,r,c = cache
  factors = get_factors(a)
  points = get_factors(x)
  @inbounds for d = 1:D
    r[d] = evaluate!(c,factors[d],points[d])
  end
  tpr = FieldFactors(r,indices_map,Anisotropic())
  return tpr
end

function Arrays.return_cache(
  a::TensorProductShapefuns{D,Isotropic},
  x::TensorProductNodes{D,Isotropic}
  ) where D

  field = get_field(a)
  factors = get_factors(a)
  points = get_factors(x)
  nodes_map = get_indices_map(x)
  indices_map = compute_nodes_and_comps_2_dof_map(field.fields,nodes_map)
  cache = return_cache(factors[1],points[1])
  return indices_map,cache
end

function Arrays.evaluate!(
  cache,
  a::TensorProductShapefuns{D,Isotropic},
  x::TensorProductNodes{D,Isotropic}
  ) where D

  indices_map,c = cache
  factors = get_factors(a)
  points = get_factors(x)
  r = evaluate!(c,factors[1],points[1])
  tpr = FieldFactors(Fill(r,D),indices_map,Isotropic())
  return tpr
end
