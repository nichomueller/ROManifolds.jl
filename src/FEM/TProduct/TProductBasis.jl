function _lagrangian_dof_basis_from_factors(::Type{T},nodes) where T
  dof_to_node = _get_dof_to_node(T,nodes)
  dof_to_comp = _get_dof_to_comp(T,nodes)
  node_and_comp_to_dof = _get_node_and_comp_to_dof(T,nodes)
  return LagrangianDofBasis(collect(nodes),dof_to_node,dof_to_comp,node_and_comp_to_dof)
end

struct TensorProductDofBases{D,I,T,A,B,C} <: AbstractVector{T}
  factors::A
  basis::B
  nodes::C
  isotropy::I

  function TensorProductDofBases(
    factors::A,
    basis::B,
    nodes::C,
    isotropy::I=Isotropy(factors)
    ) where {D,I,T,A,B<:AbstractVector{T},C<:TensorProductNodes{D}}

    new{D,I,T,A,B,C}(factors,basis,nodes,isotropy)
  end
end

function TensorProductDofBases(::Type{T},p::Polytope{D},::Lagrangian,orders) where {T,D}
  function _compute_1d_dbasis(order=first(orders))
    LagrangianDofBasis(T,SEGMENT,(order,))
  end
  isotropy = Isotropy(orders)
  factors = isotropy==Isotropic() ? Fill(_compute_1d_dbasis(),D) : map(_compute_1d_dbasis,orders)
  nodes = TensorProductNodes(p,orders)
  basis = _lagrangian_dof_basis_from_factors(T,nodes)
  TensorProductDofBases(factors,basis,nodes,isotropy)
end

function TensorProductDofBases(::Type{T},p::Polytope{D},name=lagrangian,order::Integer=1) where {T,D}
  orders = tfill(order,Val(D))
  TensorProductDofBases(T,p,name,orders)
end

Base.size(a::TensorProductDofBases) = size(a.basis)
Base.axes(a::TensorProductDofBases) = axes(a.basis)
Base.IndexStyle(::TensorProductDofBases) = IndexLinear()
Base.getindex(a::TensorProductDofBases,i::Integer) = ReferenceFEs.PointValue(a.nodes[i])

get_factors(a::TensorProductDofBases) = a.factors

ReferenceFEs.get_nodes(a::TensorProductDofBases) = a.nodes

function Arrays.return_cache(a::TensorProductDofBases,field)
  return_cache(a.basis,field)
end

function Arrays.evaluate!(cache,a::TensorProductDofBases,field)
  evaluate!(cache,a.basis,field)
end

function Arrays.return_cache(
  a::TensorProductDofBases{D},
  field::TensorProductMonomialBasis{D}
  ) where D

  bfactors = get_factors(a)
  ffactors = get_factors(field)
  s,v,c = return_cache(bfactors[1],ffactors[1])
  r = Vector{typeof(get_array(s))}(undef,D)
  return r,(s,v,c)
end

function Arrays.evaluate!(
  _cache,
  a::TensorProductDofBases{D},
  field::TensorProductMonomialBasis{D}
  ) where D

  r,cache = _cache
  bfactors = get_factors(a)
  ffactors = get_factors(field)
  indices_map = get_indices_map(field)
  @inbounds for d = 1:D
    r[d] = evaluate!(cache,bfactors[d],ffactors[d])
  end
  tpr = FieldFactors(r,indices_map,Anisotropic())
  return tpr
end

# Isotropy shortcuts

function Arrays.return_cache(
  a::TensorProductDofBases{D,Isotropic},
  field::TensorProductMonomialBasis{D,Isotropic}
  ) where D

  bfactors = get_factors(a)
  ffactors = get_factors(field)
  return return_cache(bfactors[1],ffactors[1])
end

function Arrays.evaluate!(
  cache,
  a::TensorProductDofBases{D,Isotropic},
  field::TensorProductMonomialBasis{D,Isotropic}
  ) where D

  bfactors = get_factors(a)
  ffactors = get_factors(field)
  indices_map = get_indices_map(field)
  r = evaluate!(cache,bfactors[1],ffactors[1])
  tpr = FieldFactors(Fill(r,D),indices_map,Isotropic())
  return tpr
end
