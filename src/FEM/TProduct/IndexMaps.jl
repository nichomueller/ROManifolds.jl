abstract type IndexMap <: Map end

struct NodesMap{D,A} <: IndexMap
  indices::Vector{CartesianIndex{D}}
  rmatrix::A
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

function standard_nodes_map(;
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
  return vids
end

function trivial_nodes_map(;
  polytope::Polytope{D}=QUAD,
  orders=tfill(1,Val(D))) where D

  return _get_terms(orders)
end

function compute_nodes_map(;
  polytope::Polytope{D}=QUAD,
  orders=tfill(1,Val(D))
  ) where D

  indices = standard_nodes_map(;polytope,orders)
  ordered_indices = trivial_nodes_map(;polytope,orders)
  ordered2indices = zeros(Int,length(indices))
  tup = Tuple.(indices)
  for (i,id) in enumerate(ordered_indices)
    @inbounds ordered2indices[i] = findfirst([Tuple(id)] .== tup)
  end
  rmatrix = OneHotMatrix(ordered2indices,length(indices)) |> Matrix
  return NodesMap(indices,rmatrix)
end
