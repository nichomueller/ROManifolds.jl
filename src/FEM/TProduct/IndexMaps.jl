abstract type IndexMap <: Map end

struct NodesMap{D} <: IndexMap
  indices::Vector{CartesianIndex{D}}
end

ReferenceFEs.num_nodes(a::NodesMap) = length(a.indices)
ReferenceFEs.num_dofs(a::NodesMap) = @notimplemented
Base.getindex(a::NodesMap,i::Integer) = a.indices[i]

function _index_inclusion(orders,α,β)
  N = num_vertices(SEGMENT) # 2
  if α ∈ β
    return N+1:orders[α]+1
  else
    return 1:N
  end
end

function push_entries!(vids,I::CartesianIndices{D},perm=1:D) where D
  ids = permutedims(collect(I),perm)
  for idi in ids
    push!(vids,idi)
  end
end

function compute_nodes_map(;
  polytope::Polytope{D}=QUAD,
  orders=tfill(1,Val(D))) where D

  vids = CartesianIndex{D}[]
  for d = 0:D
    I = collect(subsets(1:D,Val(d)))
    for i in I
      ij = CartesianIndices(ntuple(j->_index_inclusion(orders,j,i),D))
      push_entries!(vids,ij)
    end
  end
  return NodesMap(vids)
end

function trivial_nodes_map(;
  polytope::Polytope{D}=QUAD,
  orders=tfill(1,Val(D))) where D

  NodesMap(_get_terms(orders))
end

struct NodesAndComps2DofsMap{A,B} <: IndexMap
  nodes_map::A
  dofs_map::B
end

ReferenceFEs.num_nodes(a::NodesAndComps2DofsMap) = size(a.nodes_map,1)
ReferenceFEs.num_components(a::NodesAndComps2DofsMap) = size(a.nodes_map,2)
ReferenceFEs.num_dofs(a::NodesAndComps2DofsMap) = length(a.dofs_map)

function compute_nodes_and_comps_2_dof_map(
  nodes_map::NodesMap{D};
  T=Float64,
  orders=tfill(1,Val(D)),
  dofs_map=_get_terms(orders)) where D

  ncomps = num_components(T)
  local_nnodes = orders.+1
  global_nnodes = num_nodes(nodes_map)
  _nodes_map = nodes_map.indices

  nodes_map_comp = zeros(CartesianIndex{D},global_nnodes,ncomps)
  dofs_map_comp = zeros(CartesianIndex{D},global_nnodes,ncomps)
  @inbounds for comp in 1:ncomps
    for node in 1:global_nnodes
      nodes_map_comp[node,comp] = CartesianIndex((comp-1).*local_nnodes.+Tuple(_nodes_map[node]))
      dofs_map_comp[node,comp] = CartesianIndex((Tuple(dofs_map[node]).-1).*ncomps.+comp)
    end
  end
  NodesAndComps2DofsMap(nodes_map_comp,dofs_map_comp)
end

function compute_nodes_and_comps_2_dof_map(;
  polytope::Polytope{D}=QUAD,
  orders=tfill(1,Val(D)),
  kwargs...) where D

  nodes_map = compute_nodes_map(;polytope,orders)
  compute_nodes_and_comps_2_dof_map(nodes_map;orders,kwargs...)
end
