abstract type IsotropyStyle end
struct Isotropic <: IsotropyStyle end
struct Anisotropic <: IsotropyStyle end

Isotropy(b::Bool) = b ? Isotropic() : Anisotropic()
Isotropy(a::NTuple{D}) where D = Isotropy(all([a[d] == a[1] for d = 1:D]))
Isotropy(a::AbstractVector) = Isotropy(all([a[d] == a[1] for d = eachindex(a)]))
Isotropy(::typeof(identity)) = Isotropy(true)
Isotropy(::Function) = Isotropy(false)

abstract type AbstractTensorProductPoints{D,I,T,N} <: AbstractArray{Point{D,T},N} end

get_factors(::AbstractTensorProductPoints) = @abstractmethod
get_index_map(::AbstractTensorProductPoints) = @abstractmethod
ReferenceFEs.num_dims(::AbstractTensorProductPoints{D}) where D = D
get_isotropy(::AbstractTensorProductPoints{D,I} where D) where I = I

struct TensorProductNodes{D,I,T,A} <: AbstractTensorProductPoints{D,I,T,1}
  nodes::A
  nodes_map::NodesMap{D}
  isotropy::I
  function TensorProductNodes{T}(nodes::A,nodes_map::NodesMap{D},isotropy::I) where {D,I,T,A}
    new{D,I,T,A}(nodes,nodes_map,isotropy)
  end
end

function TensorProductNodes(nodes::NTuple{D,AbstractVector{Point{1,T}}},args...) where {D,T}
  TensorProductNodes{T}(nodes,args...)
end

function TensorProductNodes(nodes::AbstractVector{<:AbstractVector{Point{1,T}}},args...) where T
  TensorProductNodes{T}(nodes,args...)
end

function TensorProductNodes(polytope::Polytope{D},orders) where D
  function _compute_1d_nodes(order)
    nodes,_ = compute_nodes(SEGMENT,(order,))
    return nodes
  end
  o1 = first(orders)
  isotropy = Isotropy(all(orders .== o1))
  nodes = isotropy==Isotropic() ? Fill(_compute_1d_nodes(o1),D) : map(_compute_1d_nodes,orders)
  indices_map = compute_nodes_map(;polytope,orders)
  TensorProductNodes(nodes,indices_map,isotropy)
end

function TensorProductNodes(polytope::Polytope{D},order::Integer=1) where D
  TensorProductNodes(polytope,tfill(order,Val(D)))
end

get_factors(a::TensorProductNodes) = a.nodes
get_index_map(a::TensorProductNodes) = a.nodes_map

ReferenceFEs.num_nodes(a::TensorProductNodes) = length(a.nodes_map.indices)

Base.length(a::TensorProductNodes) = num_nodes(a)
Base.size(a::TensorProductNodes) = (num_nodes(a),)
Base.axes(a::TensorProductNodes) = (Base.OneTo(num_nodes(a)),)
Base.IndexStyle(::TensorProductNodes) = IndexLinear()

function Base.getindex(a::TensorProductNodes{D,I,T},i::Integer) where {D,I,T}
  entry = a.nodes_map.indices[i]
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
