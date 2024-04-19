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
    isotropy::I
    ) where {D,I,T,A,B<:AbstractVector{T},C<:TensorProductNodes{D}}

    new{D,I,T,A,B,C}(factors,basis,nodes,isotropy)
  end
end

function TensorProductDofBases(::Type{T},p::Polytope{D},::Lagrangian,orders) where {T,D}
  function _compute_1d_dbasis(order)
    LagrangianDofBasis(eltype(T),SEGMENT,(order,))
  end
  o1 = first(orders)
  isotropy = Isotropy(all(orders .== o1))
  factors = isotropy==Isotropic() ? Fill(_compute_1d_dbasis(o1),D) : map(_compute_1d_dbasis,orders)
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
get_index_map(a::TensorProductDofBases) = get_index_map(a.nodes)

ReferenceFEs.get_nodes(a::TensorProductDofBases) = a.nodes

function Arrays.return_cache(a::TensorProductDofBases,field)
  return_cache(a.basis,field)
end

function Arrays.evaluate!(cache,a::TensorProductDofBases,field)
  evaluate!(cache,a.basis,field)
end

function Arrays.return_cache(
  a::TensorProductDofBases{D},
  field::TensorProductField{D}
  ) where D

  index_map = get_index_map(a)
  bfactors = get_factors(a)
  ffactors = get_factors(field)
  s,v,c = return_cache(bfactors[1],ffactors[1])
  r = Vector{typeof(get_array(s))}(undef,D)
  return (index_map,r,(s,v,c))
end

function Arrays.evaluate!(
  _cache,
  a::TensorProductDofBases{D},
  field::TensorProductField{D}
  ) where D

  index_map,r,cache = _cache
  bfactors = get_factors(a)
  ffactors = get_factors(field)
  @inbounds for d = 1:D
    r[d] = evaluate!(cache,bfactors[d],ffactors[d])
  end
  tpr = FieldFactors(r,index_map,Anisotropic())
  return tpr
end

function Arrays.return_cache(
  a::TensorProductDofBases{D,Isotropic},
  field::TensorProductField{D,Isotropic}
  ) where D

  index_map = get_index_map(a)
  bfactors = get_factors(a)
  ffactors = get_factors(field)
  cache = return_cache(bfactors[1],ffactors[1])
  return index_map,cache
end

function Arrays.evaluate!(
  _cache,
  a::TensorProductDofBases{D,Isotropic},
  field::TensorProductField{D,Isotropic}
  ) where D

  index_map,cache = _cache
  bfactors = get_factors(a)
  ffactors = get_factors(field)
  r = evaluate!(cache,bfactors[1],ffactors[1])
  tpr = FieldFactors(Fill(r,D),index_map,Isotropic())
  return tpr
end

function Arrays.return_cache(
  a::TensorProductDofBases{D},
  field::TensorProductMonomialBasis{D}
  ) where D

  nodes_map = get_index_map(a)
  orders = get_orders(field)
  ndofs = size(field,1)
  index_map = compute_nodes_and_comps_2_dof_map(nodes_map;orders,ndofs)
  bfactors = get_factors(a)
  ffactors = get_factors(field)
  s,v,c = return_cache(bfactors[1],ffactors[1])
  r = Vector{typeof(get_array(s))}(undef,D)
  return (index_map,r,(s,v,c))
end

function Arrays.evaluate!(
  _cache,
  a::TensorProductDofBases{D},
  field::TensorProductMonomialBasis{D}
  ) where D

  index_map,r,cache = _cache
  bfactors = get_factors(a)
  ffactors = get_factors(field)
  @inbounds for d = 1:D
    r[d] = evaluate!(cache,bfactors[d],ffactors[d])
  end
  tpr = BasisFactors(r,index_map,Anisotropic())
  return tpr
end

# Isotropy shortcuts

function Arrays.return_cache(
  a::TensorProductDofBases{D,Isotropic},
  field::TensorProductMonomialBasis{D,Isotropic}
  ) where D

  nodes_map = get_index_map(a)
  orders = get_orders(field)
  ndofs = size(field,1)
  index_map = compute_nodes_and_comps_2_dof_map(nodes_map;orders,ndofs)
  bfactors = get_factors(a)
  ffactors = get_factors(field)
  cache = return_cache(bfactors[1],ffactors[1])
  return index_map,cache
end

function Arrays.evaluate!(
  _cache,
  a::TensorProductDofBases{D,Isotropic},
  field::TensorProductMonomialBasis{D,Isotropic}
  ) where D

  index_map,cache = _cache
  bfactors = get_factors(a)
  ffactors = get_factors(field)
  r = evaluate!(cache,bfactors[1],ffactors[1])
  tpr = BasisFactors(Fill(r,D),index_map,Isotropic())
  return tpr
end
