abstract type IndexMap <: Map end

struct NodesMap{D} <: IndexMap
  indices::Vector{CartesianIndex{D}}
end

ReferenceFEs.num_nodes(a::NodesMap) = length(a.indices)
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
  ndofs::Int
end

ReferenceFEs.num_nodes(a::NodesAndComps2DofsMap) = num_nodes(a.nodes_map)
ReferenceFEs.num_components(a::NodesAndComps2DofsMap) = Int(num_dofs(a)/num_nodes(a))
ReferenceFEs.num_dofs(a::NodesAndComps2DofsMap) = a.ndofs

function compute_nodes_and_comps_2_dof_map(
  nodes_map::NodesMap{D};
  orders=tfill(1,Val(D)),
  dofs_map=_get_terms(orders),
  ndofs=num_nodes(nodes_map)) where D

  NodesAndComps2DofsMap(nodes_map,dofs_map,ndofs)
end

function compute_nodes_and_comps_2_dof_map(;
  polytope::Polytope{D}=QUAD,
  orders=tfill(1,Val(D)),
  kwargs...) where D

  nodes_map = compute_nodes_map(;polytope,orders)
  compute_nodes_and_comps_2_dof_map(nodes_map;orders,kwargs...)
end
