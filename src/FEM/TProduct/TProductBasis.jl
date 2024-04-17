function _dof_basis_from_factors(::Type{T},nodes) where T
  dof_to_node = _get_dof_to_node(T,nodes)
  dof_to_comp = _get_dof_to_comp(T,nodes)
  node_and_comp_to_dof = _get_node_and_comp_to_dof(T,nodes)
  return LagrangianDofBasis(collect(nodes),dof_to_node,dof_to_comp,node_and_comp_to_dof)
end

struct TensorProductDofBases{D,T,A,B,C} <: AbstractVector{T}
  factors::A
  basis::B
  nodes::C

  function TensorProductDofBases(
    factors::A,
    basis::B,
    nodes::C) where {D,T,A,B<:AbstractVector{T},C<:TensorProductNodes{D}}

    new{D,T,A,B,C}(factors,basis,nodes)
  end
end

function TensorProductDofBases(::Type{T},p::Polytope{D},orders;lagrangian=true) where {T,D}
  function _compute_1d_dbasis(order)
    @check lagrangian
    LagrangianDofBasis(eltype(T),SEGMENT,(order,))
  end
  o1 = first(orders)
  isotropy = all(orders .== o1)
  factors = isotropy ? Fill(_compute_1d_dbasis(o1),D) : map(_compute_1d_dbasis,orders)
  nodes = TensorProductNodes(p,orders)
  basis = _dof_basis_from_factors(T,nodes)
  TensorProductDofBases(factors,basis,nodes)
end

function TensorProductDofBases(::Type{T},p::Polytope{D},order::Integer=1;kwargs...) where {T,D}
  orders = tfill(order,Val(D))
  TensorProductDofBases(T,p,orders;kwargs...)
end

Base.size(a::TensorProductDofBases) = size(a.basis)
Base.axes(a::TensorProductDofBases) = axes(a.basis)
Base.IndexStyle(::TensorProductDofBases) = IndexLinear()
Base.getindex(a::TensorProductDofBases,i::Integer) = ReferenceFEs.PointValue(a.nodes[i])

get_factors(b::TensorProductDofBases) = b.factors

ReferenceFEs.get_nodes(a::TensorProductDofBases) = a.nodes

function Arrays.return_cache(a::TensorProductDofBases,field)
  return_cache(a.basis,field)
end

function Arrays.evaluate!(cache,a::TensorProductDofBases,field)
  evaluate!(cache,a.basis,field)
end

function Arrays.return_cache(
  a::TensorProductDofBases{D},
  field::Union{TensorProductField,TensorProductMonomialBasis}
  ) where D

  nodes_map = a.nodes.nodes_map
  orders = field.basis.orders
  ndofs = num_dofs(a.basis)
  index_map = compute_nodes_and_comps_2_dof_map(nodes_map;orders,ndofs)
  b = first(get_factors(a))
  f = first(get_factors(field))
  cache = return_cache(b,f)
  r = Vector{typeof(get_array(c))}(undef,D)
  (index_map,r,cache)
end

function Arrays.evaluate!(
  _cache,
  a::TensorProductDofBases{D},
  field::Union{TensorProductField,TensorProductMonomialBasis}
  ) where D

  index_map,r,cache = _cache
  b = get_factors(a)
  f = get_factors(field)
  @inbounds for d = 1:D
    r[d] = evaluate!(cache,b[d],f[d])
  end
  tpr = BasisFactors(r,index_map)
  return tpr
end

# Fill shortcut

function Arrays.return_cache(
  a::TensorProductDofBases{D,S,<:Fill},
  field::TensorProductMonomialBasis{D,T,<:Fill}) where {D,S,T}

  nodes_map = a.nodes.nodes_map
  orders = field.basis.orders
  ndofs = num_dofs(a.basis)
  index_map = compute_nodes_and_comps_2_dof_map(nodes_map;orders,ndofs)
  cache = return_cache(a.factors[1],field.factors[1])
  return index_map,cache
end

function Arrays.evaluate!(
  _cache,
  a::TensorProductDofBases{D,S,<:Fill},
  field::TensorProductMonomialBasis{D,T,<:Fill}
  ) where {D,S,T}

  index_map,cache = _cache
  r = evaluate!(cache,a.factors[1],field.factors[1])
  tpr = BasisFactors(Fill(r,D),index_map)
  return tpr
end
