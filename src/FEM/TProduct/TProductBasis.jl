function _get_dof_to_node(::Type{T},nodes) where T
  nnodes = length(nodes)
  ncomps = num_components(T)
  ids = ntuple(i->collect(Base.OneTo(nnodes)),ncomps)
  collect1d(ids)
end

function _get_dof_to_comp(::Type{T},nodes) where T
  nnodes = length(nodes)
  ncomps = num_components(T)
  ids = ntuple(i->fill(i,nnodes),ncomps)
  collect1d(ids)
end

function _get_node_and_comp_to_dof(::Type{T},nodes) where T<:MultiValue
  nnodes = length(nodes)
  ncomps = num_components(T)
  dof_to_node = _get_dof_to_node(T,nodes)
  dof_to_comp = _get_dof_to_comp(T,nodes)
  ids = (dof_to_comp .- 1)*ncomps .+ dof_to_node
  rids = reshape(ids,nnodes,ncomps)
  [VectorValue[rids[i,:]] for i = axes(rids,1)]
end

function _get_node_and_comp_to_dof(::Type{T},nodes) where T
  _get_dof_to_node(T,nodes)
end

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
  a::TensorProductDofBases{D,T,<:LagrangianDofBasis},
  field::Union{TensorProductField,TensorProductMonomialBasis}
  ) where {D,T}

  b = first(get_factors(a))
  f = first(get_factors(field))
  c,cf = return_cache(b,f)
  vc = Vector{typeof(get_array(c))}(undef,D)
  (vc,c,cf)
end

function Arrays.evaluate!(
  cache,
  a::TensorProductDofBases{D,T,<:LagrangianDofBasis},
  field::Union{TensorProductField,TensorProductMonomialBasis}
  ) where {D,T}

  b = get_factors(a)
  f = get_factors(field)
  vc,_cache... = cache
  @inbounds for d = 1:D
    vc[d] = evaluate!(_cache,b[d],f[d])
  end
  return vc
end
