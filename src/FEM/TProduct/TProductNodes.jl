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

function indices_map(p::Polytope{D},orders::NTuple{D,Int}) where D
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

struct TensorProductNodes{D,T,A} <: AbstractVector{Point{D,T}}
  nodes::A
  nodes_map::Vector{CartesianIndex{D}}
  function TensorProductNodes(
    nodes::Tuple{Vararg{AbstractVector{Point{1,T}},D}},
    nodes_map::Vector{CartesianIndex{D}}) where {D,T}

    A = typeof(nodes)
    new{D,T,A}(nodes,nodes_map)
  end
end

Base.length(a::TensorProductNodes) = length(a.nodes_map)
Base.size(a::TensorProductNodes) = (length(a),)
Base.axes(a::TensorProductNodes) = (Base.OneTo(length(a)),)
Base.IndexStyle(::TensorProductNodes) = IndexLinear()

function Base.getindex(a::TensorProductNodes{D,T},i::Integer) where {D,T}
  entry = a.nodes_map[i]
  p = zero(Mutable(Point{D,T}))
  @inbounds for d in 1:D
    p[d] = a.nodes[d][entry[d]][1]
  end
  Point(p)
end

function Arrays.return_value(f::Field,x::TensorProductNodes)
  return_value(f,collect(x))
end

function Arrays.return_cache(f::Field,x::TensorProductNodes)
  reorder_x = collect(x)
  cache = return_cache(f,reorder_x)
  return cache,reorder_x
end

function Arrays.evaluate!(c,f::Field,x::TensorProductNodes)
  cache,reorder_x = c
  evaluate!(cache,f,reorder_x)
end
